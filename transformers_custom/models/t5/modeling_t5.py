# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch T5 model. """

import copy
import math
import os
import warnings

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...file_utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
    )
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    )
from ...modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import logging
from ...utils.model_parallel_utils import assert_device_map, get_device_map
from .configuration_t5 import T5Config

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "T5Config"
_TOKENIZER_FOR_DOC = "T5Tokenizer"

####################################################
# This dict contains ids and associated url
# for the pretrained weights provided with the models
####################################################
T5_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "t5-small",
    "t5-base",
    "t5-large",
    "t5-3b",
    "t5-11b",
    # See all T5 models at https://huggingface.co/models?filter=t5
    ]


####################################################
# This is a conversion method from TF 1.0 to PyTorch
# More details: https://medium.com/huggingface/from-tensorflow-to-pytorch-265f40ef2a28
####################################################
def load_tf_weights_in_t5(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re
        
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
                "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
                "https://www.tensorflow.org/install/ for installation instructions."
                )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    tf_weights = {}
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        tf_weights[name] = array
    
    for txt_name in names:
        name = txt_name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
                n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
                for n in name
                ):
            logger.info("Skipping {}".format("/".join(name)))
            tf_weights.pop(txt_name, None)
            continue
        if "_slot_" in name[-1]:
            logger.info("Skipping {}".format("/".join(name)))
            tf_weights.pop(txt_name, None)
            continue
        pointer = model
        array = tf_weights[txt_name]
        
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] in ["kernel", "scale", "embedding"]:
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "self_attention":
                pointer = getattr(pointer, "layer")
                pointer = pointer[0]
            elif scope_names[0] == "enc_dec_attention":
                pointer = getattr(pointer, "layer")
                pointer = pointer[1]
            elif scope_names[0] == "dense_relu_dense":
                pointer = getattr(pointer, "layer")
                pointer = pointer[2]
            elif scope_names[0] == "rms_norm":
                if hasattr(pointer, "layer_norm"):
                    pointer = getattr(pointer, "layer_norm")
                elif hasattr(pointer, "final_layer_norm"):
                    pointer = getattr(pointer, "final_layer_norm")
            elif scope_names[0] == "scale":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            elif scope_names[0] == "decoder" and name[1] == "logits":
                continue
            elif scope_names[0] == "logits":
                pointer = getattr(pointer, "lm_head")
            elif scope_names[0] == "wi" and len(scope_names) > 1 and scope_names[1].isdigit():
                pointer = getattr(pointer, f"wi_{scope_names[1]}")
                continue
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if scope_names[0] not in ["kernel", "scale", "embedding"]:
            pointer = getattr(pointer, "weight")
        if scope_names[0] != "embedding":
            logger.info("Transposing numpy weight of shape {} for {}".format(array.shape, name))
            array = np.transpose(array)
        try:
            assert (
                    pointer.shape == array.shape
            ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array.astype(np.float32))
        tf_weights.pop(txt_name, None)
    
    logger.info("Weights not copied to PyTorch model: {}".format(", ".join(tf_weights.keys())))
    return model


####################################################
# PyTorch Models are constructed by sub-classing
# - torch.nn.Module for the layers and
# - PreTrainedModel for the models (it-self a sub-class of torch.nn.Module)
####################################################
PARALLELIZE_DOCSTRING = r"""
    This is an experimental feature and is a subject to change at a moment's notice.

    Uses a device map to distribute attention modules of the model across several devices. If no device map is given,
    it will evenly distribute blocks across all devices.

    Args:
        device_map (:obj:`Dict[int, list]`, optional, defaults to None):
            A dictionary that maps attention modules to devices. Note that the embedding module and LMHead are always
            automatically mapped to the first device (for esoteric reasons). That means that the first device should
            have fewer attention modules mapped to it than other devices. For reference, the t5 models have the
            following number of attention modules:

                - t5-small: 6
                - t5-base: 12
                - t5-large: 24
                - t5-3b: 24
                - t5-11b: 24

    Example::

            # Here is an example of a device map on a machine with 4 GPUs using t5-3b, which has a total of 24 attention modules:
            model = T5ForConditionalGeneration.from_pretrained('t5-3b')
            device_map = {0: [0, 1, 2],

                         1: [3, 4, 5, 6, 7, 8, 9],
                         2: [10, 11, 12, 13, 14, 15, 16],
                         3: [17, 18, 19, 20, 21, 22, 23]}
            model.parallelize(device_map)
"""
DEPARALLELIZE_DOCSTRING = r"""
    Moves the model to cpu from a model parallel state.

    Example::

        # On a 4 GPU machine with t5-3b:
        model = T5ForConditionalGeneration.from_pretrained('t5-3b')
        device_map = {0: [0, 1, 2],

                     1: [3, 4, 5, 6, 7, 8, 9],
                     2: [10, 11, 12, 13, 14, 15, 16],
                     3: [17, 18, 19, 20, 21, 22, 23]}
        model.parallelize(device_map) # Splits the model across several devices
        model.deparallelize() # Put the model back on cpu and cleans memory by calling torch.cuda.empty_cache()
"""


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states):
        # layer norm should always be calculated in float32
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states*torch.rsqrt(variance + self.variance_epsilon)
        
        # convert into float16 if necessary
        if self.weight.dtype == torch.float16:
            hidden_states = hidden_states.to(torch.float16)
        return self.weight*hidden_states


class T5DenseReluDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = F.relu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5DenseGatedGeluDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.gelu_act = ACT2FN["gelu_new"]
    
    def forward(self, hidden_states):
        hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu*hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5LayerFF(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.feed_forward_proj == "relu":
            self.DenseReluDense = T5DenseReluDense(config)
        elif config.feed_forward_proj == "gated-gelu":
            self.DenseReluDense = T5DenseGatedGeluDense(config)
        else:
            raise ValueError(
                    f"{self.config.feed_forward_proj} is not supported. Choose between `relu` and `gated-gelu`"
                    )
        
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class T5Attention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads*self.key_value_proj_dim
        
        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)
        
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()
    
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
                heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
                )
        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim*self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)
    
    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers
        .py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long)*num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)
        
        # half of the buckets are for exact increments in positions
        max_exact = num_buckets//2
        is_small = relative_position < max_exact
        
        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_postion_if_large = max_exact + (
                torch.log(relative_position.float()/max_exact)
                /math.log(max_distance/max_exact)
                *(num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
                relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
                )
        
        relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
        return relative_buckets
    
    def compute_bias(self, query_length, key_length):
        """ Compute binned relative position bias """
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
                relative_position,  # shape (query_length, key_length)
                bidirectional=(not self.is_decoder),
                num_buckets=self.relative_attention_num_buckets,
                )
        relative_position_bucket = relative_position_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values
    
    def forward(
            self,
            hidden_states,
            mask=None,
            key_value_states=None,
            position_bias=None,
            past_key_value=None,
            layer_head_mask=None,
            query_length=None,
            use_cache=False,
            output_attentions=False,
            ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # if key_value_states is None:
        #     print("key_value_states: ", key_value_states)
        # else:
        #     print("key_value_states.shape: ", key_value_states.shape)
        
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]
        real_seq_length = seq_length
        # print("real_seq_length A: ", real_seq_length)
        
        if past_key_value is not None:
            assert (
                    len(past_key_value) == 2
            ), "past_key_value should have 2 past states: keys and values. Got {} past states".format(
                    len(past_key_value)
                    )
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length
        
        # print("real_seq_length B: ", real_seq_length)
        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]
        
        # print("A key_length: ", key_length)
        # if key_value_states is None:
        #     print("A key_value_states: ", key_value_states)
        # else:
        #     print("A key_value_states.shape[1]: ", key_value_states.shape[1])
        
        def shape(states):
            """  projection """
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        
        def unshape(states):
            """  reshape """
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
        
        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """ projects hidden states correctly to key/query states """
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))
            
            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states
        
        # print("hidden_states.shape: ", hidden_states.shape)
        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        
        # get key/value states
        key_states = project(
                hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
                )
        value_states = project(
                hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
                )
        
        # print("query_states.shape: ", query_states.shape)
        # print("key_states.shape: ", key_states.shape)
        # compute scores
        scores = torch.matmul(
                query_states, key_states.transpose(3, 2)
                )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9
        
        if position_bias is None:
            # print("position_bias: ", position_bias)
            if not self.has_relative_attention_bias:
                # print("key_length: ", key_length)
                position_bias = torch.zeros(
                        (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                        )
                # torch.Size([1, 16, 1, 85])
                # print("position_bias A ", position_bias.shape)
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)
                # print("position_bias B ", position_bias.shape)
            
            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -seq_length:, :]
                # print("position_bias C ", position_bias.shape)
            
            if mask is not None:
                if mask.shape[-1] > position_bias.shape[-1]:
                    mask = mask[:, :, :, :position_bias.shape[-1]]
                # print("position_bias D before", position_bias.shape)
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)
                # print("position_bias D after", position_bias.shape)
                # print("mask D ", mask.shape)
        
        # else:
        # print("position_bias not None: ", position_bias.shape)
        
        # print("scores.shape: ", scores.shape) # err: scores.shape:  torch.Size([1, 16, 1, 1])
        # print("position_bias.shape: ", position_bias.shape) # err: position_bias.shape:  torch.Size([1, 16, 1, 85])
        scores += position_bias
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(
                scores
                )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = F.dropout(
                attn_weights, p=self.dropout, training=self.training
                )  # (batch_size, n_heads, seq_length, key_length)
        
        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights*layer_head_mask
        
        attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)
        
        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)
        
        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class T5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_bias=None,
            layer_head_mask=None,
            past_key_value=None,
            use_cache=False,
            output_attentions=False,
            ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
                normed_hidden_states,
                mask=attention_mask,
                position_bias=position_bias,
                layer_head_mask=layer_head_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
                )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5LayerCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.EncDecAttention = T5Attention(config, has_relative_attention_bias=False)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(
            self,
            hidden_states,
            key_value_states,
            attention_mask=None,
            position_bias=None,
            layer_head_mask=None,
            past_key_value=None,
            use_cache=False,
            query_length=None,
            output_attentions=False,
            ):
        # if position_bias is None:
        #     # print("T5LayerCrossAttention, position_bias: ", position_bias)
        # else:
        # print("T5LayerCrossAttention, position_bias.shape: ", position_bias.shape)
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
                normed_hidden_states,
                mask=attention_mask,
                key_value_states=key_value_states,
                position_bias=position_bias,
                layer_head_mask=layer_head_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                query_length=query_length,
                output_attentions=output_attentions,
                )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))
        
        self.layer.append(T5LayerFF(config))
    
    def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_bias=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            encoder_decoder_position_bias=None,
            layer_head_mask=None,
            encoder_layer_head_mask=None,
            past_key_value=None,
            use_cache=False,
            output_attentions=False,
            return_dict=True,
            ):
        
        if past_key_value is not None:
            assert self.is_decoder, "Only decoder can use `past_key_values`"
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4
            
            error_message = "There should be {} past states. 2 (past / key) for self attention.{} Got {} past key / value states".format(
                    expected_num_past_key_values,
                    "2 (past / key) for cross attention" if expected_num_past_key_values == 4 else "",
                    len(past_key_value),
                    )
            assert len(past_key_value) == expected_num_past_key_values, error_message
            
            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None
        
        self_attention_outputs = self.layer[0](
                hidden_states,
                attention_mask=attention_mask,
                position_bias=position_bias,
                layer_head_mask=layer_head_mask,
                past_key_value=self_attn_past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
                )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention output and relative position weights
        
        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        
        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None
            
            # print("before cross_attention_outputs, hidden_states.shape: ", hidden_states.shape)
            # print("before cross_attention_outputs, encoder_hidden_states.shape: ", encoder_hidden_states.shape)
            # print("before cross_attention_outputs, encoder_attention_mask.shape: ", encoder_attention_mask.shape)
            # if encoder_decoder_position_bias is None:
            #     # print("T5Block, before cross_attention_outputs, encoder_decoder_position_bias: ", encoder_decoder_position_bias)
            # else:
            # print("T5Block, before cross_attention_outputs, encoder_decoder_position_bias.shape: ", encoder_decoder_position_bias.shape)
            
            cross_attention_outputs = self.layer[1](
                    hidden_states,
                    key_value_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    position_bias=encoder_decoder_position_bias,
                    layer_head_mask=encoder_layer_head_mask,
                    past_key_value=cross_attn_past_key_value,
                    query_length=query_length,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    )
            hidden_states = cross_attention_outputs[0]
            
            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
            
            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]
            
            # Keep cross-attention output and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]
        
        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)
        
        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        
        outputs = (hidden_states,)
        
        outputs = outputs + (present_key_value_state,) + attention_outputs
        return outputs  # hidden-states, present_key_value_states, (self-attention weights), (self-attention position bias),
        # (cross-attention weights), (cross-attention position bias)


class T5PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    
    config_class = T5Config
    load_tf_weights = load_tf_weights_in_t5
    base_model_prefix = "transformer"
    is_parallelizable = True
    
    @property
    def dummy_inputs(self):
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {
            "decoder_input_ids"     : input_ids,
            "input_ids"             : input_ids,
            "decoder_attention_mask": input_mask,
            }
        return dummy_inputs
    
    def _init_weights(self, module):
        """ Initialize the weights """
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, T5LayerNorm):
            module.weight.data.fill_(factor*1.0)
        elif isinstance(module, (T5Model, T5ForConditionalGeneration, T5EncoderModel)):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            module.shared.weight.data.normal_(mean=0.0, std=factor*1.0)
        elif isinstance(module, T5DenseReluDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            module.wi.weight.data.normal_(mean=0.0, std=factor*((self.config.d_model) ** -0.5))
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor*((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5DenseGatedGeluDense):
            module.wi_0.weight.data.normal_(mean=0.0, std=factor*((self.config.d_model) ** -0.5))
            if hasattr(module.wi_0, "bias") and module.wi_0.bias is not None:
                module.wi_0.bias.data.zero_()
            module.wi_1.weight.data.normal_(mean=0.0, std=factor*((self.config.d_model) ** -0.5))
            if hasattr(module.wi_1, "bias") and module.wi_1.bias is not None:
                module.wi_1.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor*((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5Attention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention
            # .py#L136
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(mean=0.0, std=factor*((d_model*key_value_proj_dim) ** -0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor*(d_model ** -0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor*(d_model ** -0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor*((n_heads*key_value_proj_dim) ** -0.5))
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor*((d_model) ** -0.5))
    
    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id
        
        assert (
                decoder_start_token_id is not None
        ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more " \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "" \
           "information"
        
        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id
        
        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        
        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"
        
        return shifted_input_ids


class T5Stack(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)
        
        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        
        self.block = nn.ModuleList(
                [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
                )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # print("config.dropout_rate: ", config.dropout_rate)
        self.dropout = nn.Dropout(config.dropout_rate)
        
        self.init_weights()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
    
    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.block), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.block))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        # Load onto devices
        for k, v in self.device_map.items():
            for layer in v:
                cuda_device = "cuda:" + str(k)
                self.block[layer] = self.block[layer].to(cuda_device)
        
        # Set embed_tokens to first layer
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        # Set final layer norm to last device
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)
    
    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        for i in range(len(self.block)):
            self.block[i] = self.block[i].to("cpu")
        self.embed_tokens = self.embed_tokens.to("cpu")
        self.final_layer_norm = self.final_layer_norm.to("cpu")
        torch.cuda.empty_cache()
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings
    
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            inputs_logits=None,
            inputs_embeds=None,
            head_mask=None,
            encoder_head_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs
            ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                    f"You cannot specify both {err_msg_prefix}inputs and {err_msg_prefix}inputs_embeds at the same time"
                    )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        elif inputs_logits is not None:
            input_shape = inputs_logits.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}inputs or {err_msg_prefix}inputs_embeds")
        
        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            if input_ids is not None:
                inputs_embeds = self.embed_tokens(input_ids)
                # print("self.embed_tokens: ", self.embed_tokens)
                # print("self.embed_tokens.weight.shape: ", self.embed_tokens.weight.shape) # torch.Size([30080, 512/1024])
                # print("A1 inputs_embeds.shape: ", inputs_embeds.shape) # torch.Size([16, 85, 512/1024])
            elif inputs_logits is not None:
                # print("A2 inputs_logits: ", inputs_logits)
                # print("A2 inputs_logits.shape: ", inputs_logits.shape) # torch.Size([16, 85, 30080])
                # print("A2 inputs_logits sum .shape : ", torch.sum(inputs_logits, dim=-1).shape) # torch.Size([16, 85])
                # print("A2 self.embed_tokens.weight.shape: ", self.embed_tokens.weight.shape)
                inputs_probs = F.softmax(inputs_logits, dim=-1)
                # print("A2 inputs_probs.shape: ", inputs_probs.shape) # torch.Size([16, 85, 30080])
                # print("A2 inputs_probs sum .shape : ", torch.sum(inputs_probs, dim=-1).shape) # torch.Size([16, 85])
                inputs_embeds = torch.matmul(inputs_probs, self.embed_tokens.weight)
                # print("A2 inputs_embeds.shape: ", inputs_embeds.shape)
        
        batch_size, seq_length = input_shape
        
        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length
        # print("T5Stack, mask_seq_length", mask_seq_length)
        
        if use_cache is True:
            assert self.is_decoder, ":obj:`use_cache` can only be set to `True` if {} is used as a decoder".format(
                    self
                    )
        
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
            # print("T5Stack, attention_mask is None", attention_mask.shape)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                    batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
                    )
            
            # print("T5Stack, encoder_attention_mask A", encoder_attention_mask.shape)
            # print("T5Stack, encoder_hidden_states A", encoder_hidden_states.shape)
        # else:
        #     if encoder_attention_mask is None:
        #         # print("T5Stack, encoder_attention_mask B", encoder_attention_mask)
        #     else:
        # print("T5Stack, encoder_attention_mask B", encoder_attention_mask.shape)
        
        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None]*len(self.block)
        
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)
        # print("T5Stack, input_shape", input_shape)
        # print("T5Stack, extended_attention_mask", extended_attention_mask.shape)
        
        if self.is_decoder and encoder_attention_mask is not None:
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
            # print("T5Stack, encoder_attention_mask C", encoder_attention_mask.shape)
            # print("T5Stack, encoder_extended_attention_mask C", encoder_extended_attention_mask.shape)
            # print("T5Stack, encoder_attention_mask C", encoder_attention_mask)
            # print("T5Stack, encoder_extended_attention_mask C", encoder_extended_attention_mask)
        else:
            encoder_extended_attention_mask = None
        
        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        encoder_head_mask = self.get_head_mask(encoder_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None
        
        hidden_states = self.dropout(inputs_embeds)
        
        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            encoder_layer_head_mask = encoder_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if encoder_layer_head_mask is not None:
                    encoder_layer_head_mask = encoder_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            # print("T5Stack, extended_attention_mask B ", extended_attention_mask.shape)
            # if encoder_extended_attention_mask is None:
            #     # print("T5Stack, encoder_extended_attention_mask B ", encoder_extended_attention_mask)
            # else:
            # print("T5Stack, encoder_extended_attention_mask B ", encoder_extended_attention_mask.shape)
            layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    encoder_layer_head_mask=encoder_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    )
            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention weights), (self-attention position bias), (cross-attention weights),
            # (cross-attention position bias)
            hidden_states, present_key_value_state = layer_outputs[:2]
            
            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention weights),
            # (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)
            
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)
            
            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))
        
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        if not return_dict:
            return tuple(
                    v
                    for v in [
                        hidden_states,
                        present_key_value_states,
                        all_hidden_states,
                        all_attentions,
                        all_cross_attentions,
                        ]
                    if v is not None
                    )
        return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                past_key_values=present_key_value_states,
                hidden_states=all_hidden_states,
                attentions=all_attentions,
                cross_attentions=all_cross_attentions,
                )


T5_START_DOCSTRING = r"""

    The T5 model was proposed in `Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
    <https://arxiv.org/abs/1910.10683>`__ by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang,
    Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. It's an encoder decoder transformer pre-trained in a text-to-text
    denoising generative setting.

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.T5Config`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

T5_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
            should be able to pad the inputs on both the right and the left.

            Indices can be obtained using :class:`~transformers.T5Tokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            detail.

            `What are input IDs? <../glossary.html#input-ids>`__

            To know more on how to prepare :obj:`input_ids` for pretraining take a look a `T5 Training
            <./t5.html#training>`__.
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.BartTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__

            T5 uses the :obj:`pad_token_id` as the starting token for :obj:`decoder_input_ids` generation. If
            :obj:`past_key_values` is used, optionally only the last :obj:`decoder_input_ids` have to be input (see
            :obj:`past_key_values`).

            To know more on how to prepare :obj:`decoder_input_ids` for pretraining take a look at `T5 Training
            <./t5.html#training>`__. If :obj:`decoder_input_ids` and :obj:`decoder_inputs_embeds` are both unset,
            :obj:`decoder_input_ids` takes the value of :obj:`input_ids`.
        decoder_attention_mask (:obj:`torch.BoolTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Default behavior: generate a tensor that ignores pad tokens in :obj:`decoder_input_ids`. Causal mask will
            also be used by default.
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules in the encoder. Mask values selected in ``[0,
            1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        decoder_head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. in the decoder Mask values selected in ``[0,
            1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        encoder_outputs (:obj:`tuple(tuple(torch.FloatTensor)`, `optional`):
            Tuple consists of (:obj:`last_hidden_state`, :obj:`optional`: `hidden_states`, :obj:`optional`:
            `attentions`) :obj:`last_hidden_state` of shape :obj:`(batch_size, sequence_length, hidden_size)` is a
            sequence of hidden states at the output of the last layer of the encoder. Used in the cross-attention of
            the decoder.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of
        shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        decoder_inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, target_sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`decoder_input_ids` you can choose to directly pass an embedded
            representation. If :obj:`past_key_values` is used, optionally only the last :obj:`decoder_inputs_embeds`
            have to be input (see :obj:`past_key_values`). This is useful if you want more control over how to convert
            :obj:`decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

            If :obj:`decoder_input_ids` and :obj:`decoder_inputs_embeds` are both unset, :obj:`decoder_inputs_embeds`
            takes the value of :obj:`inputs_embeds`.

        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).

        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""

T5_ENCODER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
            should be able to pad the inputs on both the right and the left.

            Indices can be obtained using :class:`~transformers.T5Tokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            detail.

            To know more on how to prepare :obj:`input_ids` for pretraining take a look a `T5 Training
            <./t5.html#training>`__.
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""

# Warning messafe for FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""


@add_start_docstrings(
        "The bare T5 Model transformer outputting raw hidden-states" "without any specific head on top.",
        T5_START_DOCSTRING,
        )
class T5Model(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
        ]
    
    def __init__(self, config: T5Config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)
        
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)
        
        self.init_weights()
        
        # Model parallel
        self.model_parallel = False
        self.device_map = None
    
    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.model_parallel = True
    
    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()
    
    def get_input_embeddings(self):
        return self.shared
    
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)
    
    def get_encoder(self):
        return self.encoder
    
    def get_decoder(self):
        return self.decoder
    
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)
    
    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            ):
        r"""
        Returns:

        Example::

            >>> from transformers import T5Tokenizer, T5Model

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5Model.from_pretrained('t5-small')

            >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  #
            Batch size 1
            >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
            >>> output = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

            >>> last_hidden_states = output.last_hidden_state
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask
        
        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                    last_hidden_state=encoder_outputs[0],
                    hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                    attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                    )
        
        hidden_states = encoder_outputs[0]
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)
        
        # Decode
        decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                past_key_values=past_key_values,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
                head_mask=decoder_head_mask,
                encoder_head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                )
        
        if not return_dict:
            return decoder_outputs + encoder_outputs
        
        return Seq2SeqModelOutput(
                last_hidden_state=decoder_outputs.last_hidden_state,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
                )


# adapted from https://github.com/IBM/controlled-peptide-generation - start -
def kl_gaussianprior(mu, logvar):
    """ analytically compute kl divergence with unit gaussian. """
    return torch.mean(0.5*torch.sum((logvar.exp() + mu ** 2 - 1 - logvar), 1))


def kl_gaussian_sharedmu(mu, logvar):
    """ analytically compute kl divergence N(mu,sigma) with N(mu, I). """
    return torch.mean(0.5*torch.sum((logvar.exp() - 1 - logvar), 1))


cfgm_default = {
    # 'sigma'      : 32.0,  # kernel width: ~ O( sqrt(z_dim) ), 7.0 for z_dim=100
    'sigma'      : 14.0,  # kernel width: ~ O( sqrt(z_dim) ), 7.0 for z_dim=100
    # 'sigma': 7.0,  # kernel width: ~ O( sqrt(z_dim) ), 7.0 for z_dim=100
    'kernel'     : 'gaussian',
    # for method = rf
    'rf_dim'     : 500,
    'rf_resample': False
    }


def wae_mmd_gaussianprior(z, method='full_kernel', cfgm=cfgm_default):
    """ compute MMD with samples from unit gaussian.
    MMD parametrization from cfg loaded here."""
    z_prior = torch.randn_like(z)  # shape and device
    # cfgm = cfg.losses.wae_mmd
    
    if method == 'full_kernel':
        mmd_kwargs = {'sigma': cfgm['sigma'], 'kernel': cfgm['kernel']}
        return mmd_full_kernel(z, z_prior, **mmd_kwargs)
    else:
        mmd_kwargs = {**cfgm}  # shallow copy, all cfg params.
        # print("mmd_kwargs: ", mmd_kwargs)
        return mmd_rf(z, z_prior, **mmd_kwargs)


def mmd_full_kernel(z1, z2, **mmd_kwargs):
    K11 = compute_mmd_kernel(z1, z1, **mmd_kwargs)
    K22 = compute_mmd_kernel(z2, z2, **mmd_kwargs)
    K12 = compute_mmd_kernel(z1, z2, **mmd_kwargs)
    N = z1.size(0)
    assert N == z2.size(0), 'expected matching sizes z1 z2'
    H = K11 + K22 - K12*2  # gretton 2012 eq (4)
    H = H - torch.diag(H)  # unbiased: delete diagonal. Makes MMD^2_u negative! (typically)
    loss = 1./(N*(N - 1))*H.sum()
    return loss


def mmd_rf(z1, z2, **mmd_kwargs):
    mu1 = compute_mmd_mean_rf(z1, **mmd_kwargs)
    mu2 = compute_mmd_mean_rf(z2, **mmd_kwargs)
    loss = ((mu1 - mu2) ** 2).sum()
    return loss


rf = {}


def compute_mmd_mean_rf(z, sigma, kernel, rf_dim, rf_resample=False):
    # random features approx of gaussian kernel mmd.
    # rf_resample: keep fixed base of RF? or resample RF every time?
    # Then just loss = |mu_real - mu_fake|_H
    global rf
    if kernel == 'gaussian':
        if not kernel in rf or rf_resample:
            # sample rf if it's the first time or we want to resample every time
            rf_w = torch.randn((z.shape[1], rf_dim), device=z.device)
            rf_b = math.pi*2*torch.rand((rf_dim,), device=z.device)
            rf['gaussian'] = (rf_w, rf_b)
        else:
            rf_w, rf_b = rf['gaussian']
            assert rf_w.shape == (z.shape[1], rf_dim), 'not expecting z dim or rf_dim to change'
        z_rf = compute_gaussian_rf(z, rf_w, rf_b, sigma, rf_dim)
    else:  # kernel xxx
        raise ValueError('todo implement rf for kernel ' + kernel)
    mu_rf = z_rf.mean(0, keepdim=False)
    return mu_rf


def compute_gaussian_rf(z, rf_w, rf_b, sigma, rf_dim):
    z_emb = (z@rf_w)/sigma + rf_b
    z_emb = torch.cos(z_emb)*(2./rf_dim) ** 0.5
    return z_emb


def compute_mmd_kernel(x, y, sigma, kernel):
    """ x: (Nxd) y: (Mxd). sigma: kernel width """
    # adapted from https://discuss.pytorch.org/t/error-when-implementing-rbf-kernel-bandwidth-differentiation-in-pytorch/13542
    x_i = x.unsqueeze(1)
    y_j = y.unsqueeze(0)
    xmy = ((x_i - y_j) ** 2).sum(2)
    if kernel == "gaussian":
        K = torch.exp(- xmy/sigma ** 2)
    elif kernel == "laplace":
        K = torch.exp(- torch.sqrt(xmy + (sigma ** 2)))
    elif kernel == "energy":
        K = torch.pow(xmy + (sigma ** 2), -.25)
    return K


# adapted from https://github.com/IBM/controlled-peptide-generation - end -


@add_start_docstrings("""T5 Model with a `language modeling` head on top. """, T5_START_DOCSTRING)
class T5ForConditionalGenerationWithLatentSpace(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
        ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
        ]
    
    def __init__(
            self, config: T5Config,
            latent_pooler='cls',
            pool_enc_hidden_states_for_dec=True,
            mask_non_target_z_vector=False,
            separate_targetattr_head=False,
            do_mi=False,
            z_tar_vector_dim=1,
            latent_space_type='wae',  # ['plain', 'vae', 'wae', 'adversarial']
            wae_z_enc_type='deterministic',  # ['deterministic', 'stochastic', None] None is same as stochastic
            separate_latent_enc=False,
            separate_latent_dec=False,
            mmd_method='rf',  # defaults to random feature ,['rf', 'full_kernel']
            sigma_mmd=None,
            rf_dim_mmd=None,
            dim_target_kl=0.5,  # KL loss term threshold to stabilize vae training
            latent_size=1024,  # if None, will use hidden state as latent vector (z)
            ):
        super().__init__(config)
        
        print('==================== T5 Model T5ForConditionalGenerationWithLatentSpace Initialization start ====================')
        
        self.model_dim = config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)
        
        # latent space modules - end
        self.z_tar_vector_dim = z_tar_vector_dim
        self.z_nontar_vector_dim = self.model_dim - z_tar_vector_dim
        self.latent_space_type = latent_space_type
        self.wae_z_enc_type = wae_z_enc_type
        self.separate_latent_enc = separate_latent_enc
        self.separate_latent_dec = separate_latent_dec
        self.mmd_method = mmd_method
        self.sigma_mmd = sigma_mmd
        self.rf_dim_mmd = rf_dim_mmd
        self.dim_target_kl = dim_target_kl
        self.latent_size = latent_size
        self.latent_pooler = latent_pooler
        self.pool_enc_hidden_states_for_dec = pool_enc_hidden_states_for_dec
        self.mask_non_target_z_vector = mask_non_target_z_vector
        self.separate_targetattr_head = separate_targetattr_head
        self.do_mi = do_mi
        
        # 打印上面的参数
        print('========== Initialize T5ForConditionalGenerationWithLatentSpace ==========')
        print("latent_space_type: ", self.latent_space_type)
        print("wae_z_enc_type: ", self.wae_z_enc_type)
        print("separate_latent_enc: ", self.separate_latent_enc)
        print("separate_latent_dec: ", self.separate_latent_dec)
        print("mmd_method: ", self.mmd_method)
        print("sigma_mmd: ", self.sigma_mmd)
        print("rf_dim_mmd: ", self.rf_dim_mmd)
        print("dim_target_kl: ", self.dim_target_kl)
        print("latent_size: ", self.latent_size)
        print("latent_pooler: ", self.latent_pooler)
        print("pool_enc_hidden_states_for_dec: ", self.pool_enc_hidden_states_for_dec)
        print("mask_non_target_z_vector: ", self.mask_non_target_z_vector)
        print("separate_targetattr_head: ", self.separate_targetattr_head)
        print("do_mi: ", self.do_mi)
        print('========== Initialize T5ForConditionalGenerationWithLatentSpace ==========')
        
        if separate_targetattr_head:
            self.targetattr_head = nn.Linear(z_tar_vector_dim, 1, bias=False)
        
        if do_mi:
            mi_head_input_dim = self.model_dim - z_tar_vector_dim
            self.mi_head = nn.Linear(mi_head_input_dim, 1, bias=False)
        
        if self.latent_space_type in ['vae', 'wae']:
            if latent_size is None:
                raise ValueError("latent_size cannot be {} when latent_space_type is {}".format(latent_size, latent_space_type))
            if z_tar_vector_dim > latent_size:
                raise ValueError(
                        "z_tar_vector_dim of size {} cannot be larger than latent_size of size {} in vae mode".format(z_tar_vector_dim,
                                                                                                                      latent_size))
            if self.wae_z_enc_type == 'deterministic':
                self.vae_enc = nn.Linear(self.model_dim, self.latent_size, bias=False)
            else:
                self.vae_enc = nn.Linear(self.model_dim, 2*self.latent_size, bias=False)
            self.vae_dec = nn.Linear(self.latent_size, self.model_dim, bias=False)
        # latent space modules - end
        
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.init_weights()
        
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        print('==================== T5 Model T5ForConditionalGenerationWithLatentSpace Initialization end ====================')
    
    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        if self.separate_targetattr_head:
            self.targetattr_head = self.targetattr_head.to(self.decoder.first_device)
        
        if self.latent_space_type in ['vae', 'wae']:
            self.vae_enc = self.vae_enc.to(self.decoder.first_device)
            self.vae_dec = self.vae_dec.to(self.decoder.first_device)
        
        if self.do_mi:
            self.mi_head = self.mi_head.to(self.decoder.first_device)
        self.model_parallel = True
    
    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()
    
    def get_input_embeddings(self):
        return self.shared
    
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def get_encoder(self):
        return self.encoder
    
    def get_decoder(self):
        return self.decoder
    
    # reparameterization trick for vae
    def reparameterize(self, mu, logvar, nsamples=1):
        """sample from posterior Gaussian family
        Args:
            mu: Tensor
                Mean of gaussian distribution with shape (batch, nz)
            logvar: Tensor
                logvar of gaussian distibution with shape (batch, nz)
        Returns: Tensor
            Sampled z with shape (batch, nsamples, nz)
        """
        batch_size, nz = mu.size()
        std = logvar.mul(0.5).exp()
        
        # mu_expd = mu.expand(batch_size, nsamples, nz)
        # std_expd = std.expand(batch_size, nsamples, nz)
        mu_expd = mu.unsqueeze(1).expand(batch_size, nsamples, nz)
        std_expd = std.unsqueeze(1).expand(batch_size, nsamples, nz)
        
        eps = torch.zeros_like(std_expd).normal_()
        
        return mu_expd + torch.mul(eps, std_expd)
    
    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            inputs_logits=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            contrast_targets=None,
            return_contrastive_head_pred=True,
            train_mi_head_step=False,
            output_perturbed_z_dec_logits=False,
            z_tar_edit_before_dec=None,  # magnitude to shift  value_pred by
            z_tar_edit_noise_std_before_dec=None,
            return_only_value_pred=False,
            # mask_similar_contrast_label=True, # mask contrastive loss for sample pairs with similar label
            # return_same_label_loss=True,
            mask_similar_contrast_label=False,  # mask contrastive loss for sample pairs with similar label
            return_same_label_loss=False,
            ):
        """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``

        Returns:

        Examples::

            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')

            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> output = model(input_ids=input_ids, labels=labels)
            >>> loss = output.loss
            >>> logits = output.logits

            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids
            # Batch size 1
            >>> output = model.generate(input_ids)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # print('forward, use_cache', use_cache)  # True
        # print('forward, return_dict', return_dict)  # True, self.config.use_return_dict: True
        
        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            print('head mask function')  # haven't been here
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask
        
        # print('-'*100)
        # print("calling model forward")  # always here
        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            '''used in training'''
            '''The first forward use input_ids, while the second forward use inputs_logits'''
            # print("forward, encoder_outputs is None")  # always here when training
            # print('*'*100)
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    inputs_logits=inputs_logits,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            '''used in conditional generation'''
            '''No need to use encoder to encode input sequence each time, directly use the parameter encoder_outputs'''
            # print("z_tar_edit_before_dec: ", z_tar_edit_before_dec)
            # print("forward, encoder_outputs is not None")
            # print("forward, len(encoder_outputs): ", len(encoder_outputs))
            # print("forward, encoder_outputs[0].shape: ", encoder_outputs[0].shape)
            encoder_outputs = BaseModelOutput(
                    last_hidden_state=encoder_outputs[0],
                    hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                    attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                    )
        
        hidden_states = encoder_outputs[0]
        # print("before latent, hidden_states.shape: ", hidden_states.shape)
        # [batch, length, dim] torch.Size([16, 84 (原版是85 = <cls> + 84), 1024])
        '''Add latent space code here before decoder step: start'''
        '''debug 后发现需要对 pooled_hidden_states 进行.clone()，否则每一次decode都会累加edit的值'''
        
        # pool hidden state to form fixed-length z
        if self.latent_pooler == 'mean':
            pooled_hidden_states = torch.mean(hidden_states, dim=1, keepdim=True).clone()
        elif self.latent_pooler == 'max':
            pooled_hidden_states = torch.max(hidden_states, dim=1, keepdim=True)[0].clone()
        elif self.latent_pooler == 'cls':
            pooled_hidden_states = hidden_states[:, :1, :].clone()
        
        # vae vector sampling
        if self.latent_space_type in ['vae', 'wae']:
            # print("pooled_hidden_states.shape: ", pooled_hidden_states.shape)  # [16, 1, 1024]
            vae_enc_input = pooled_hidden_states.squeeze(dim=1)  # remove len dim
            # print("vae_enc_input.shape: ", vae_enc_input.shape)  # [16, 1024]
            
            if self.wae_z_enc_type == 'deterministic':
                if self.separate_latent_enc == True:
                    '''no use here, self.separate_latent_enc == False'''
                    if self.model_parallel:
                        self.vae_enc = self.vae_enc.to(self.pooled_hidden_states.device)
                    latent_vector = self.vae_enc(pooled_hidden_states)
                else:
                    '''use here'''
                    latent_vector = pooled_hidden_states  # [16, 1, 1024]
            else:
                '''no use here, self.wae_z_enc_type == deterministic'''
                if self.model_parallel:
                    self.vae_enc = self.vae_enc.to(self.vae_enc_input.device)
                mu, logvar = self.vae_enc(vae_enc_input).chunk(2, -1)
                latent_vector = self.reparameterize(mu, logvar, nsamples=1)
            # print("mu.shape: ", mu.shape)
            # print("logvar.shape: ", logvar.shape)
            # print("latent_vector.shape: ", latent_vector.shape)
            
            if self.latent_space_type == 'wae':
                cfgm = {**cfgm_default}
                if self.sigma_mmd is not None:
                    cfgm['sigma'] = self.sigma_mmd
                if self.rf_dim_mmd is not None:
                    cfgm['rf_dim'] = self.rf_dim_mmd
                z = latent_vector.squeeze(1)  # [16, 1024]
                wae_mmd_loss = wae_mmd_gaussianprior(z, method=self.mmd_method, cfgm=cfgm)
                z_regu_loss = wae_mmd_loss
            else:
                '''no use here, self.latent_space_type == wae'''
                kl_loss = 0.5*(mu.pow(2) + logvar.exp() - logvar - 1)
                kl_mask = (kl_loss > self.dim_target_kl).float()
                kl_loss = (kl_mask*kl_loss).sum(dim=1).mean()  # sum over dim and mean over batch
                z_regu_loss = kl_loss
            
            # latent logvar regularization
            if self.wae_z_enc_type != 'deterministic':
                '''no use here, self.wae_z_enc_type == deterministic'''
                z_logvar_L1 = logvar.abs().sum(1).mean(0)  # L1 in z-dim, mean over mb.
                z_logvar_KL_penalty = kl_gaussian_sharedmu(mu, logvar)
        else:
            '''no use here, self.latent_space_type == wae'''
            latent_vector = pooled_hidden_states
        
        # latent_vector: torch.Size([16, 1, 1024])
        # take first several dims as predicted contrast values, torch.Size([16, 1, z_tar_vector_dim])
        
        # latent_vector: [16, 1, 1024]
        z_tar = latent_vector[:, :, :self.z_tar_vector_dim]  # torch.Size([16, 1, 1]) or torch.Size([16, 1, 2])
        
        if self.do_mi:
            '''no use yet'''
            z_nontar = latent_vector[:, :, self.z_tar_vector_dim:]
            if self.model_parallel:
                self.mi_head = self.mi_head.to(self.z_nontar.device)
            
            if train_mi_head_step:
                z_nontar = z_nontar.detach()
            
            nontar_value_pred = self.mi_head(z_nontar)  # torch.Size([16, 1, 1])
            nontar_value_pred = torch.squeeze(nontar_value_pred, -1)  # torch.Size([16, 1])
            if contrast_targets is not None:
                if len(contrast_targets.shape) != 2:
                    contrast_targets = torch.unsqueeze(contrast_targets, dim=-1)
                contrast_labels = torch.sign(contrast_targets - contrast_targets.transpose(1, 0))*0.5 + 0.5
                contrastive_preds = F.logsigmoid(nontar_value_pred - nontar_value_pred.transpose(1, 0))
                inverse_preds = F.logsigmoid(-1*(nontar_value_pred - nontar_value_pred.transpose(1, 0)))
                if self.model_parallel:
                    contrast_labels = contrast_labels.to(contrastive_preds.device)
                losses = -contrast_labels*contrastive_preds - (1 - contrast_labels)*inverse_preds
                
                loss_mask = 1 - torch.eye(losses.shape[0], device=losses.device)
                nontar_contrastive_pred_loss = torch.sum(losses*loss_mask)/torch.sum(loss_mask)
                
                # print("nontar_contrastive_pred_loss: ", nontar_contrastive_pred_loss)
                if train_mi_head_step:
                    # print("nontar_contrastive_pred_loss: ", nontar_contrastive_pred_loss)
                    return nontar_contrastive_pred_loss
        
        '''no use yet'''
        if self.separate_targetattr_head:
            if self.model_parallel:
                self.targetattr_head = self.targetattr_head.to(z_tar.device)
            
            value_pred = self.targetattr_head(z_tar)  # torch.Size([16, 1, 1]) or torch.Size([16, 1, 2])
        else:
            '''use here'''
            value_pred = z_tar  # torch.Size([16, 1, 1]) or torch.Size([16, 1, 2]) or torch.Size([16, 1, 2])
        
        value_pred = torch.squeeze(value_pred, 1)  # torch.Size([16, 1]) or torch.Size([16, 2])
        
        if return_only_value_pred:
            '''no use here, used only when lambda_contrastive_perturb_cyc>0'''
            return value_pred
        
        if contrast_targets is not None:
            def compute_contrastive_loss(value_pred, targets):
                if len(targets.shape) != 2:
                    targets = torch.unsqueeze(targets, dim=-1)  # [16, 1]
                
                '''construct contrastive learning labels'''
                # targets.transpose(1, 0): [16, 1] -> [1, 16]
                contrast_labels = torch.sign(targets - targets.transpose(1, 0))*0.5 + 0.5  # torch.Size([16, 16])
                # when targets[i] > targets[j], contrast_labels[i, j] = 1
                # when targets[i] < targets[j], contrast_labels[i, j] = 0
                # when targets[i] = targets[j], contrast_labels[i, j] = 0.5
                
                '''compute contrastive learning predictions'''
                value_pred_diff = value_pred - value_pred.transpose(1, 0)  # torch.Size([16, 16])
                contrastive_preds = F.logsigmoid(value_pred_diff)  # torch.Size([16, 16]), all values range in (0, 1)
                inverse_preds = F.logsigmoid(-1*value_pred_diff)  # torch.Size([16, 16]), all values range in (0, 1)
                
                '''compute contrastive learning loss matrix'''
                if self.model_parallel:
                    contrast_labels = contrast_labels.to(contrastive_preds.device)
                losses = -contrast_labels*contrastive_preds - (1 - contrast_labels)*inverse_preds
                
                '''compute contrastive learning loss mask'''
                similar_label_mask = (contrast_labels != 0.5).float()  # mask diagonal values where contrast_labels[i, j] = 0.5
                self_mask = 1 - torch.eye(losses.shape[0], device=losses.device)  # torch.Size([16, 16])
                if mask_similar_contrast_label:
                    '''no use here, mask_similar_contrast_label == False'''
                    loss_mask = similar_label_mask
                else:
                    loss_mask = self_mask  # 对角线为0，其余为1
                
                '''compute final contrastive learning loss'''
                contrastive_pred_loss = torch.sum(losses*loss_mask)/torch.sum(loss_mask)
                
                same_label_loss = 0
                if return_same_label_loss:
                    '''no use here, return_same_label_loss == False'''
                    same_label_loss_mask = (1 - similar_label_mask)*self_mask
                    same_label_loss = torch.sum(torch.abs(value_pred_diff*same_label_loss_mask))/torch.sum(same_label_loss_mask)
                return contrastive_pred_loss, same_label_loss
            
            # print('value_pred.shape[-1]', value_pred.shape)
            # print('self.z_tar_vector_dim', self.z_tar_vector_dim)
            
            contrastive_pred_loss = 0
            same_label_loss = 0
            edit_property_dim = 0
            [ddG_targets, solubility_targets] = contrast_targets
            if ddG_targets is not None:
                ddG_value_pred = value_pred[:, edit_property_dim].unsqueeze(-1)
                ddG_loss, ddG_same_label_loss = compute_contrastive_loss(ddG_value_pred, ddG_targets)
                contrastive_pred_loss += ddG_loss
                same_label_loss += ddG_same_label_loss
                # print('ddG_loss', ddG_loss)
                edit_property_dim += 1
            if solubility_targets is not None:
                solubility_value_pred = value_pred[:, edit_property_dim].unsqueeze(-1)
                solubility_loss, solubility_same_label_loss = compute_contrastive_loss(solubility_value_pred, solubility_targets)
                contrastive_pred_loss += solubility_loss
                same_label_loss += solubility_same_label_loss
                # print('solubility_loss', solubility_loss)
                edit_property_dim += 1
            # print('contrastive_pred_loss', contrastive_pred_loss)
        
        # mask non-target z positions as zero
        if self.mask_non_target_z_vector:
            '''no use yet, mask_non_target_z_vector == False'''
            latent_vector[:, :, self.z_tar_vector_dim:] = 0
        
        '''used in conditional generation process'''
        if z_tar_edit_before_dec is not None:
            if self.z_tar_vector_dim == 1:  # scalar value
                if z_tar_edit_noise_std_before_dec is not None:
                    '''no use here, z_tar_edit_noise_std_before_dec == None'''
                    #  TODO: add noise to the edit vector/scalar
                    pass
                else:
                    # print("z_tar_edit_before_dec: ", z_tar_edit_before_dec)
                    # z_tar: torch.Size([16, 1, 1])
                    z_tar_edit_tensor = torch.full_like(z_tar, z_tar_edit_before_dec)
                
                # print("latent_vector before edit: ", latent_vector.shape, latent_vector[0, 0, :3])
                # latent_vector: [16, 1, 1024], z_tar_edit_tensor: [16, 1, 1]
                latent_vector[:, :, :self.z_tar_vector_dim] = z_tar + z_tar_edit_tensor
                # print("latent_vector after edit: ", latent_vector.shape, latent_vector[0, 0, :3])
            elif self.z_tar_vector_dim == 2:  # z edit is a vector
                assert type(z_tar_edit_before_dec) == list
                ddG_z_edit, solubility_z_edit = z_tar_edit_before_dec
                # z_tar: torch.Size([16, 1, 2])
                z_tar_edit_tensor = torch.full_like(z_tar, ddG_z_edit)
                z_tar_edit_tensor[:, :, 1] = solubility_z_edit
                # print("latent_vector before edit: ", latent_vector.shape, latent_vector[0, 0, :3])
                latent_vector[:, :, :self.z_tar_vector_dim] = z_tar + z_tar_edit_tensor
                # print("latent_vector after edit: ", latent_vector.shape, latent_vector[0, 0, :3])
            else:
                raise RuntimeError()
        
        if self.latent_space_type in ['vae', 'wae'] and self.separate_latent_dec != False:
            '''no use here, separate_latent_dec == False'''
            if self.model_parallel:
                self.vae_dec = self.vae_dec.to(latent_vector.device)
            pooled_hidden_states = self.vae_dec(latent_vector)
        else:
            '''use here'''
            pooled_hidden_states = latent_vector  # edited latent vector when generation, [16, 1, 1024]
        
        # print("hidden_states.shape: ", hidden_states.shape)  # torch.Size([64, 85, 1024]), 来自原始的encoder输出，池化前的结果
        # replace hidden_states with pooled_hidden_states to decode from the latent space
        if self.pool_enc_hidden_states_for_dec:
            '''use here'''
            hidden_states = pooled_hidden_states
        
        # print("pooled_hidden_states.shape: ", pooled_hidden_states.shape)  # torch.Size([64, 1, 1024]), 来自原池化后的结果
        # print("after latent, hidden_states.shape: ", hidden_states.shape)  # [batch, length, dim] torch.Size([16, 1, 1024])
        # Add latent space code here before decoder step: end
        
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
        
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)
        
        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]
        
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)
        
        decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                past_key_values=past_key_values,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
                head_mask=decoder_head_mask,
                encoder_head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                )
        
        sequence_output = decoder_outputs[0]
        # print("sequence_output.shape: ", sequence_output.shape) # torch.Size([16, 84, 1024])
        
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)
        
        # print('config.tie_word_embeddings', self.config.tie_word_embeddings) # False
        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer
            # /transformer.py#L586
            sequence_output = sequence_output*(self.model_dim ** -0.5)
        
        # print("sequence_output.shape: ", sequence_output.shape)  # torch.Size([64, 85, 1024]), [batch, length, dim]
        
        lm_logits = self.lm_head(sequence_output)
        # print("lm_logits: ", lm_logits)
        # print("lm_logits argmax: ", torch.argmax(lm_logits, dim=-1))
        # print("lm_logits.shape: ", lm_logits.shape)  # torch.Size([16, 84, 128])
        
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # lm_logits.view(-1, lm_logits.size(-1)): torch.Size([16 * 85, 128])
            # labels.view(-1): torch.Size([16 * 85])
            # print("labels: ", labels)  # tensor([[ 7, 11, 12,  ...,  4, 16,  1], [ 7, 11, 12,  ...,  4, 16,  1], ..., ]])
            # print("labels.shape: ", labels.shape)  # torch.Size([16, 85])
            # print("lm_logits.shape: ", lm_logits.shape)  # torch.Size([16, 85, 128])
        
        if not return_dict:
            '''it seems return_dict is always True, no matter when training or generation'''
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output
        
        seq2seqoutput = Seq2SeqLMOutput(
                loss=loss,
                logits=lm_logits,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
                )
        output = (seq2seqoutput,)
        if contrast_targets is not None:
            '''used in training'''
            if return_same_label_loss:
                '''not used here, return_same_label_loss == False'''
                output = output + (contrastive_pred_loss, value_pred, same_label_loss,)
            else:
                '''use here'''
                output = output + (contrastive_pred_loss, value_pred,)
            if self.do_mi:
                '''not used here, do_mi == False'''
                output = output + (nontar_contrastive_pred_loss,)
        else:
            '''used in generation'''
            if return_contrastive_head_pred:
                '''use here'''
                output = output + (value_pred,)
            elif not self.latent_space_type in ['vae', 'wae']:
                '''not used here'''
                return seq2seqoutput
        
        if self.latent_space_type in ['vae', 'wae']:
            if self.wae_z_enc_type == 'deterministic':
                '''use here'''
                z_regu_dict = {
                    'z_regu_loss': z_regu_loss,
                    }
            else:
                '''no use here'''
                z_regu_dict = {
                    'z_regu_loss'        : z_regu_loss,
                    'z_logvar_L1'        : z_logvar_L1,
                    'z_logvar_KL_penalty': z_logvar_KL_penalty,
                    }
            output = output + (z_regu_dict,)
        
        '''Training output: (seq2seqoutput, contrastive_pred_loss, value_pred, z_regu_dict)'''
        '''Generation output: (seq2seqoutput, value_pred, z_regu_dict)'''
        return output
    
    # TODO: edit to pass in latent_vector edit inputs
    def prepare_inputs_for_generation(
            self, input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
            ):
        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]
        
        return_dict = {
            "decoder_input_ids": input_ids,
            "past_key_values"  : past,
            "encoder_outputs"  : encoder_outputs,
            "attention_mask"   : attention_mask,
            "use_cache"        : use_cache,
            }
        
        if "z_tar_edit_before_dec" in kwargs:
            # print("z_tar_edit_before_dec in kwargs")
            return_dict["z_tar_edit_before_dec"] = kwargs["z_tar_edit_before_dec"]
        
        if "z_tar_edit_noise_std_before_dec" in kwargs:
            # print("z_tar_edit_noise_std_before_dec in kwargs")
            return_dict["z_tar_edit_noise_std_before_dec"] = kwargs["z_tar_edit_noise_std_before_dec"]
        
        return return_dict
    
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)
    
    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past
        
        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx),
                    )
            
            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)
            
            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


# @add_start_docstrings("""Discriminator with T5 encoder module. """, T5_START_DOCSTRING)
# @add_start_docstrings(
#     "The bare T5 Model transformer outputting encoder's raw hidden-states" "without any specific head on top.",
#     T5_START_DOCSTRING,
# )
class T5Discriminator(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
        ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
        ]
    
    def __init__(
            self, config,
            latent_pooler='mean',
            ):
        super().__init__(config)
        self.model_dim = config.d_model
        
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)
        
        # latent space modules - end
        self.latent_pooler = latent_pooler
        
        self.targetattr_head = nn.Linear(self.model_dim, 1, bias=False)
        
        self.init_weights()
        
        # Model parallel
        self.model_parallel = False
        self.device_map = None
    
    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        # print("self.device_map: ", self.device_map)
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.targetattr_head = self.targetattr_head.to(self.encoder.last_device)
        self.model_parallel = True
    
    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()
    
    def get_input_embeddings(self):
        return self.shared
    
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
    
    def get_encoder(self):
        return self.encoder
    
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)
    
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            inputs_logits=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            contrast_targets=None,
            mask_similar_contrast_label=False,  # mask contrastive loss for sample pairs with similar label
            return_same_label_loss=False,
            ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                inputs_logits=inputs_logits,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                )
        
        hidden_states = encoder_outputs[0]
        
        # pool hidden state to form fixed-length z
        if self.latent_pooler == 'mean':
            # print("self.latent_pooler == mean")
            pooled_hidden_states = torch.mean(hidden_states, dim=1, keepdim=True)
        elif self.latent_pooler == 'max':
            pooled_hidden_states = torch.max(hidden_states, dim=1, keepdim=True)[0]
        elif self.latent_pooler == 'cls':
            # get hidden state from <cls> position
            # print("self.latent_pooler == cls")
            pooled_hidden_states = hidden_states[:, :1, :]
        
        value_pred = self.targetattr_head(pooled_hidden_states)
        
        value_pred = torch.squeeze(value_pred, -1)  # torch.Size([16, 1])
        # print("value_pred.shape: ", value_pred.shape)
        
        if contrast_targets is not None:
            # print("A contrast_targets: ", contrast_targets)
            # print("A contrast_targets.shape: ", contrast_targets.shape)
            if len(contrast_targets.shape) != 2:
                contrast_targets = torch.unsqueeze(contrast_targets, dim=-1)
            # print("B contrast_targets.shape: ", contrast_targets.shape) # torch.Size([16, 1])
            contrast_labels = torch.sign(contrast_targets - contrast_targets.transpose(1, 0))*0.5 + 0.5
            # print("B contrast_targets: ", contrast_targets)
            # print("B contrast_labels: ", contrast_labels) #  tensor([[0.5000, 0.0000, 1.0000, 1.0000, 1.0000, 0.0000, 0.0000, 0.0000,
            # 1.0000,.. , 0.5 at the diagonal
            # print("B contrast_labels.shape: ", contrast_labels.shape) # torch.Size([16, 16])
            value_pred_diff = value_pred - value_pred.transpose(1, 0)
            contrastive_preds = F.logsigmoid(value_pred_diff)
            # print("B value_pred.shape: ", value_pred.shape)  # torch.Size([16, 1])
            # print("B value_pred.transpose(1,0).shape: ", value_pred.transpose(1,0).shape) # torch.Size([1, 16])
            # print("B contrastive_preds.shape: ", contrastive_preds.shape) # torch.Size([16, 16])
            inverse_preds = F.logsigmoid(-1*value_pred_diff)
            # print("B inverse_preds.shape: ", inverse_preds.shape)  # torch.Size([16, 16])
            # print("contrast_labels.device: ", contrast_labels.device)
            # print("contrastive_preds.device: ", contrastive_preds.device)
            # print("inverse_preds.device: ", inverse_preds.device)
            # print("same device: ", inverse_preds.device == contrastive_preds.device)
            if self.model_parallel:
                contrast_labels = contrast_labels.to(contrastive_preds.device)
            losses = -contrast_labels*contrastive_preds - (1 - contrast_labels)*inverse_preds
            
            similar_label_mask = (contrast_labels != 0.5).float()
            self_mask = 1 - torch.eye(losses.shape[0], device=losses.device)
            if mask_similar_contrast_label:
                # print("similar_label_mask: ", similar_label_mask)
                loss_mask = similar_label_mask
            else:
                loss_mask = self_mask
                # print("loss_mask: ", loss_mask)
                # print("B self_mask: ", loss_mask)  # tensor([[0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,  , 0 at the diagonal
            # print("loss_mask: ", loss_mask)
            # print("B loss_mask.shape: ", loss_mask.shape)  # torch.Size([16, 16])
            # print("B losses.shape: ", losses.shape)  # torch.Size([16, 16])
            contrastive_pred_loss = torch.sum(losses*loss_mask)/torch.sum(loss_mask)
            
            output = (contrastive_pred_loss, value_pred)
            
            if return_same_label_loss:
                same_label_loss_mask = (1 - similar_label_mask)*self_mask
                same_label_loss = torch.sum(torch.abs(value_pred_diff*same_label_loss_mask))/torch.sum(same_label_loss_mask)
                
                output = output + (same_label_loss,)
        else:
            output = (value_pred,)
        
        return output


@add_start_docstrings("""T5 Model with a `language modeling` head on top. """, T5_START_DOCSTRING)
class T5ForConditionalGeneration(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
        ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
        ]
    
    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model
        
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)
        
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)
        
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        self.init_weights()
        
        # Model parallel
        self.model_parallel = False
        self.device_map = None
    
    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True
    
    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()
    
    def get_input_embeddings(self):
        return self.shared
    
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def get_encoder(self):
        return self.encoder
    
    def get_decoder(self):
        return self.decoder
    
    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``

        Returns:

        Examples::

            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')

            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> output = model(input_ids=input_ids, labels=labels)
            >>> loss = output.loss
            >>> logits = output.logits

            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids
            # Batch size 1
            >>> output = model.generate(input_ids)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask
        
        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                    last_hidden_state=encoder_outputs[0],
                    hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                    attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                    )
        
        hidden_states = encoder_outputs[0]
        
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
        
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)
        
        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]
        
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)
        
        # Decode
        decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                past_key_values=past_key_values,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
                head_mask=decoder_head_mask,
                encoder_head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                )
        
        sequence_output = decoder_outputs[0]
        
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)
        
        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer
            # /transformer.py#L586
            sequence_output = sequence_output*(self.model_dim ** -0.5)
        
        lm_logits = self.lm_head(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow
            #  /layers.py#L666
        
        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output
        
        return Seq2SeqLMOutput(
                loss=loss,
                logits=lm_logits,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
                )
    
    def prepare_inputs_for_generation(
            self, input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
            ):
        
        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]
        
        return {
            "decoder_input_ids": input_ids,
            "past_key_values"  : past,
            "encoder_outputs"  : encoder_outputs,
            "attention_mask"   : attention_mask,
            "use_cache"        : use_cache,
            }
    
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)
    
    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past
        
        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx),
                    )
            
            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)
            
            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


@add_start_docstrings(
        "The bare T5 Model transformer outputting encoder's raw hidden-states" "without any specific head on top.",
        T5_START_DOCSTRING,
        )
class T5EncoderModel(T5PreTrainedModel):
    authorized_missing_keys = [
        r"encoder\.embed_tokens\.weight",
        ]
    
    def __init__(self, config: T5Config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        
        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)
        
        self.init_weights()
        
        # Model parallel
        self.model_parallel = False
        self.device_map = None
    
    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.model_parallel = True
    
    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()
    
    def get_input_embeddings(self):
        return self.shared
    
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
    
    def get_encoder(self):
        return self.encoder
    
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)
    
    @add_start_docstrings_to_model_forward(T5_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            ):
        r"""
        Returns:

        Example::

            >>> from transformers import T5Tokenizer, T5EncoderModel
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5EncoderModel.from_pretrained('t5-small')
            >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  #
            Batch size 1
            >>> output = model(input_ids=input_ids)
            >>> last_hidden_states = output.last_hidden_state
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                )
        
        return encoder_outputs
