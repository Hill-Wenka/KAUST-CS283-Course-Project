'''
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

# 获取项目根目录
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print("sys.path: ", sys.path)

import re
import shutil
import torch
import argparse
import torch.nn.functional as F
import numpy as np
import random
import scipy
import typing
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW, get_scheduler
from transformers import T5Tokenizer
from transformers_custom import MT5ForConditionalGenerationWithLatentSpace, MT5Config

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

# argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--seed', action='store', type=int, default=42, help='random seed')
parser.add_argument('--data_dir', action='store', type=str, help='input df filename',
                    default="/home/hew/storage/storage/genhance/data/")
parser.add_argument('--pretrained_dir', action='store', type=str, help='dir path for pretrained progeny weights',
                    default="Rostlab/prot_t5_xl_uniref50")
parser.add_argument('--cache_dir', action='store', type=str, help='pretrained model cache path',
                    default='/home/hew/storage/storage/genhance/pretrained/')
parser.add_argument('--src_json', action='store', type=str, help='model config json file',
                    default='/home/hew/python/genhance/temp/config.json')
parser.add_argument('--output_dir', action='store', type=str, default="/home/hew/storage/storage/genhance/ckpts/congen1/results/",
                    help='input df filename')
parser.add_argument('--num_train_epochs', action='store', type=int, default=12)
parser.add_argument('--per_device_train_batch_size', action='store', type=int, default=16)
parser.add_argument('--per_device_eval_batch_size', action='store', type=int, default=64)
parser.add_argument('--warmup_steps', action='store', type=int, default=500)
parser.add_argument('--weight_decay', action='store', type=float, default=0.01)
parser.add_argument('--logging_dir', action='store', type=str, default='/scratch/hew/genhance/ckpts/congen3/log')
parser.add_argument('--save_total_limit', action='store', type=int, default=30)
parser.add_argument('--save_steps', action='store', type=int, default=10000)
parser.add_argument('--logging_steps', action='store', type=int, default=500)
parser.add_argument('--eval_steps', action='store', type=int, default=1000)
parser.add_argument('--num_warmup_steps', action='store', type=int, default=0)

parser.add_argument('--lr', action='store', type=float, default=5e-05, help='learning rate')
parser.add_argument('--train_ratio', action='store', type=float, default=1.0)
parser.add_argument('--train_split_name', action='store', type=str, default="train")
parser.add_argument('--eval_split_name', action='store', type=str, default="valid")

# latent space args
parser.add_argument('--latent_pooler', action='store', type=str, default="mean", choices=['mean', 'max', 'cls'],
                    help='op to pool encoder hidden states')
parser.add_argument('--pool_enc_hidden_states_for_dec', action='store_true')
parser.add_argument('--mask_non_target_z_vector', action='store_true')
parser.add_argument('--lambda_contrastive', action='store', type=float, default=1.0)
parser.add_argument('--lambda_contrastive_cyc', action='store', type=float, default=0.0)
parser.add_argument('--contrastive_cyc_start_step', action='store', type=int, default=-1,
                    help='Step index to start contrastive_cyc loss minimization')

parser.add_argument('--lambda_contrastive_perturb_cyc', action='store', type=float, default=0.0)
parser.add_argument('--contrastive_perturb_cyc_start_step', action='store', type=int, default=-1,
                    help='Step index to start contrastive_perturb_cyc loss minimization')
parser.add_argument('--pc_perturb', action='store', type=float, default=-0.25, help='Perturbation for contrastive_perturb_cyc loss')
parser.add_argument('--pc_perturb_type', action='store', type=str, default='std', choices=['std', 'fixed'],
                    help='type of z perturbation for perturb cycle contrastive loss')

parser.add_argument('--separate_targetattr_head', action='store_true')
parser.add_argument('--z_tar_vector_dim', action='store', type=int, default=1)
parser.add_argument('--do_mi', action='store_true')
parser.add_argument('--lambda_mi_head_loss', action='store', type=float, default=1.0)

# vae/wae args
parser.add_argument('--dim_target_kl', action='store', type=float, default=0.5)
parser.add_argument("--beta", type=float, default=1.0,
                    help="The weighting hyper-parameter of the KL term in VAE")
parser.add_argument("--lambda_logvar_L1", type=float, default=0.0,
                    help="Regularizing term to prevent z_logvar from being too large, recommended to be 0")
parser.add_argument("--lambda_logvar_KL", type=float, default=0.0,
                    help="Regularizing term to prevent z_logvar from diminishing, recommended to be 1e-3")
parser.add_argument("--use_beta_schedule", action='store_true', help="Use cyclical beta schedule for vae/wae.")
parser.add_argument("--beta_ratio_increase", default=0.25, type=float,
                    help="Learning schedule, the percentage for the annealing stage.")
parser.add_argument("--beta_ratio_zero", default=0.25, type=float,
                    help="Learning schedule, the percentage for the pure auto-encoding stage.")
parser.add_argument('--beta_start_step', action='store', type=int, default=-1, help='Step index to start z_regu_loss minimization')

parser.add_argument('--latent_space_type', action='store', type=str, default="plain", choices=['plain', 'vae', 'wae', 'adversarial'],
                    help='type of latent space')
parser.add_argument('--latent_size', action='store', type=int, default=None,
                    help='use None to use pooled enc hidden state as latent vector')

parser.add_argument('--no_separate_latent_enc', action='store_false', dest='separate_latent_enc', default=True)
parser.add_argument('--no_separate_latent_dec', action='store_false', dest='separate_latent_dec', default=True)

# wae only args
parser.add_argument('--wae_z_enc_type', action='store', type=str, default=None, choices=['deterministic', 'stochastic'],
                    help='type of wae encoder')
parser.add_argument('--mmd_method', action='store', type=str, default="rf", choices=['rf', 'full_kernel'],
                    help='random feature approx or full kernel for mmd computation')
parser.add_argument("--sigma_mmd", type=float, default=None,
                    help="use None for default, RBF kernel width: ~ O( sqrt(z_dim) ), 7.0 for z_dim=100")
parser.add_argument("--rf_dim_mmd", type=int, default=None,
                    help="Dim of random features")

# self-defined args
parser.add_argument('--property', type=str, default='ddG', choices=['ddG', 'solubility', 'ddG_solubility'],
                    help='property to be optimized')
parser.add_argument('--num_layers', type=int, default=6, help='number of used layers in the encoder')
parser.add_argument('--num_decoder_layers', type=int, default=6, help='number of used layers in the decoder')
parser.add_argument('--cuda_device', type=str, default='0', help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--datafile', type=str, default="train_new_tophalf_ddG.pkl", help='pkl data file')

args = parser.parse_args()


class PKLDFDatasetForGen(Dataset):
    def __init__(
            self,
            data_file: typing.Union[str, Path],
            in_memory: bool = False,
            split: str = 'train',
            train_ratio: float = 1,
            data_subset='full'
            ):
        
        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)
        
        df = pd.read_pickle(data_file)
        
        if train_ratio != 1:
            shuffled_df = df.sort_index()
            train_num_samples = int(len(shuffled_df)*train_ratio)
            if split == 'train':
                final_df = shuffled_df.iloc[:train_num_samples]
            elif split == 'valid':
                final_df = shuffled_df.iloc[train_num_samples:]
            else:
                final_df = df
        else:
            final_df = df
        
        print("split: ", split)
        print("data_file: ", data_file)
        print("len(final_df): ", len(final_df))
        
        self.df = final_df
        num_examples = len(final_df)
        self._num_examples = num_examples
        
        if in_memory:
            cache = [None]*num_examples
            self._cache = cache
        
        self._in_memory = in_memory
    
    def __len__(self) -> int:
        return self._num_examples
    
    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)
        if self._in_memory and self._cache[index] is not None:
            item = self._cache[index]
        else:
            row = self.df.iloc[index]
            item = {}
            item['ddG'] = row['ddG']
            item['solubility'] = row['solubility']
            item['input_ids'] = row['MT_seq']
            item['labels'] = row['MT_seq']
            item['id'] = str(index)
            if self._in_memory:
                self._cache[index] = item
        return item


class CustomStabilityDatasetForGenLatentSpace(Dataset):
    def __init__(
            self,
            data_path: typing.Union[str, Path],
            split: str,
            tokenizer,
            in_memory: bool = False,
            train_ratio: float = 1,
            data_subset='full'
            ):
        
        self.tokenizer = tokenizer
        
        if split == 'valid':
            file_prefix = 'train'
        else:
            file_prefix = split
        
        data_path = Path(data_path)
        # data_file = f'{file_prefix}_tophalf_ddG_solubility.pkl'  # top half
        # data_file = f'{file_prefix}_new_tophalf_ddG_solubility.pkl'  # top half
        data_file = args.datafile
        self.data = PKLDFDatasetForGen(data_path/data_file, in_memory, split, train_ratio, data_subset)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int):
        item = self.data[index]
        input_ids = item['input_ids']
        labels = item['labels']
        if args.property == 'ddG':
            ddG = item['ddG']
            solubility = None
        elif args.property == 'solubility':
            ddG = None
            solubility = item['solubility']
        elif args.property == 'ddG_solubility':
            ddG = item['ddG']
            solubility = item['solubility']
        else:
            raise RuntimeError('invalid optimization property')
        return input_ids, labels, ddG, solubility
    
    def collate_fn(self, batch: typing.List[typing.Tuple[typing.Any, ...]]) -> typing.Dict[str, torch.Tensor]:
        input_ids, labels, ddG, solubility = tuple(zip(*batch))
        prefix = "<cls> "
        input_ids = [prefix + " ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in input_ids]
        labels = [prefix + " ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in labels]
        input_ids = self.tokenizer.batch_encode_plus(input_ids, add_special_tokens=True, padding="longest", return_tensors='pt')[
            'input_ids']
        labels = self.tokenizer.batch_encode_plus(labels, add_special_tokens=True, padding="longest", return_tensors='pt')['input_ids']
        ddG = torch.Tensor(ddG) if None not in ddG else None
        solubility = torch.Tensor(solubility) if None not in solubility else None
        return {'input_ids': input_ids, 'labels': labels, 'ddG': ddG, 'solubility': solubility}


def spearmanr(target, prediction):
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return scipy.stats.spearmanr(target_array, prediction_array).correlation


def evaluate(model, eval_loader, do_mi=False, do_spearmanr=True, latent_space_type='wae', return_pred=False):
    eval_contrastive_loss_total = 0
    eval_lm_loss_total = 0
    if do_mi:
        eval_mi_head_loss_total = 0
    if latent_space_type in ['vae', 'wae']:
        eval_z_regu_loss_total = 0
    model.eval()
    num_eval_batch = 0
    
    ddG_preds = []
    ddG_targs = []
    solubility_preds = []
    solubility_targs = []
    
    with torch.no_grad():
        for step, batch in enumerate(tqdm(eval_loader)):
            input_ids = batch['input_ids'].to(model.device)
            labels = batch['labels'].to(model.device)
            ddG_targets = batch['ddG'].to(model.device) if batch['ddG'] is not None else None
            solubility_targets = batch['solubility'].to(model.device) if batch['solubility'] is not None else None
            contrast_targets = [ddG_targets, solubility_targets]
            
            if do_mi:
                model_outputs = model(input_ids, labels=labels, contrast_targets=contrast_targets)
                outputs, contrastive_loss, contrastive_value, mi_head_loss = model_outputs[0], model_outputs[1], model_outputs[2], \
                    model_outputs[3]
                eval_mi_head_loss_total = eval_mi_head_loss_total + mi_head_loss
            else:
                model_outputs = model(input_ids, labels=labels, contrast_targets=contrast_targets)
                outputs, contrastive_loss, contrastive_value = model_outputs[0], model_outputs[1], model_outputs[2]
            
            if latent_space_type in ['vae', 'wae']:
                z_regu_output = model_outputs[-1]
                if type(z_regu_output) is dict:
                    z_regu_loss = z_regu_output['z_regu_loss']
                else:
                    z_regu_loss = z_regu_output
            
            # contrastive_value: [batch_size, 1] or [batch_size, 2], ddG_targets: [batch_size]
            if args.property == 'ddG':
                for pred, target in zip(contrastive_value.squeeze().cpu().numpy(), ddG_targets.cpu().numpy()):
                    ddG_targs.append(target)
                    ddG_preds.append(pred)
            elif args.property == 'solubility':
                for pred, target in zip(contrastive_value.squeeze().cpu().numpy(), solubility_targets.cpu().numpy()):
                    solubility_targs.append(target)
                    solubility_preds.append(pred)
            elif args.property == 'ddG_solubility':
                for pred, target in zip(contrastive_value[:, 0].squeeze().cpu().numpy(), ddG_targets.cpu().numpy()):
                    ddG_targs.append(target)
                    ddG_preds.append(pred)
                for pred, target in zip(contrastive_value[:, 1].squeeze().cpu().numpy(), solubility_targets.cpu().numpy()):
                    solubility_targs.append(target)
                    solubility_preds.append(pred)
            
            lm_loss = outputs.loss
            
            eval_contrastive_loss_total = eval_contrastive_loss_total + contrastive_loss
            eval_lm_loss_total = eval_lm_loss_total + lm_loss
            
            if latent_space_type in ['vae', 'wae']:
                eval_z_regu_loss_total = eval_z_regu_loss_total + z_regu_loss
            
            num_eval_batch += 1
    
    eval_lm_loss = eval_lm_loss_total/num_eval_batch
    eval_contrastive_loss = eval_contrastive_loss_total/num_eval_batch
    eval_output = {
        "lm_loss"         : eval_lm_loss,
        "contrastive_loss": eval_contrastive_loss,
        }
    
    if do_mi:
        eval_mi_head_loss_total = eval_mi_head_loss_total/num_eval_batch
        eval_output['mi_head_loss'] = eval_mi_head_loss_total
    
    if latent_space_type in ['vae', 'wae']:
        eval_z_regu_loss_total = eval_z_regu_loss_total/num_eval_batch
        eval_output['z_regu_loss'] = eval_z_regu_loss_total
    
    if do_spearmanr:
        ddG_spearmanr = 0
        solubility_spearmanr = 0
        if len(ddG_targs) > 0 and len(ddG_preds) > 0:
            ddG_spearmanr = spearmanr(ddG_targs, ddG_preds)
        if len(solubility_targs) > 0 and len(solubility_preds) > 0:
            solubility_spearmanr = spearmanr(solubility_targs, solubility_preds)
        
        print()
        print("========== evaluation finished ==========")
        print("ddG_spearmanr: ", ddG_spearmanr)
        print("solubility_spearmanr: ", solubility_spearmanr)
        print("========== evaluation finished ==========")
        eval_output['ddG_spearmanr'] = ddG_spearmanr
        eval_output['solubility_spearmanr'] = solubility_spearmanr
    
    if return_pred:
        eval_output['ddG_preds'] = ddG_preds
        eval_output['ddG_targs'] = ddG_targs
        eval_output['solubility_preds'] = solubility_preds
        eval_output['solubility_targs'] = solubility_targs
    
    return eval_output


def frange_cycle_zero_linear(n_iter, start=0.0, stop=1.0, n_cycle=4, ratio_increase=0.5, ratio_zero=0.3):
    L = np.ones(n_iter)*stop
    period = n_iter/n_cycle
    step = (stop - start)/(period*ratio_increase)  # linear schedule
    
    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i + c*period) < n_iter):
            if i < period*ratio_zero:
                L[int(i + c*period)] = start
            else:
                L[int(i + c*period)] = v
                v += step
            i += 1
    return L


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("mkdir: ", path)


if args.logging_dir is None:
    args.logging_dir = args.output_dir

print("args: ", args)
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device

if args.property == 'ddG':
    args.z_tar_vector_dim = 1
elif args.property == 'solubility':
    args.z_tar_vector_dim = 1
elif args.property == 'ddG_solubility':
    args.z_tar_vector_dim = 2
else:
    raise ValueError(args.property)

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
logging_dir = Path(args.logging_dir)
logging_dir.mkdir(parents=True, exist_ok=True)
tb_writer = SummaryWriter(logging_dir)

latent_space_type = args.latent_space_type
wae_z_enc_type = args.wae_z_enc_type
latent_space_args = {
    'latent_pooler'                 : args.latent_pooler,
    'pool_enc_hidden_states_for_dec': args.pool_enc_hidden_states_for_dec,
    'mask_non_target_z_vector'      : args.mask_non_target_z_vector,
    'separate_targetattr_head'      : args.separate_targetattr_head,
    'z_tar_vector_dim'              : args.z_tar_vector_dim,
    'do_mi'                         : args.do_mi,
    'latent_space_type'             : args.latent_space_type,
    'separate_latent_enc'           : args.separate_latent_enc,
    'separate_latent_dec'           : args.separate_latent_dec,
    'wae_z_enc_type'                : args.wae_z_enc_type,
    'latent_size'                   : args.latent_size,
    'dim_target_kl'                 : args.dim_target_kl,
    'mmd_method'                    : args.mmd_method,
    'sigma_mmd'                     : args.sigma_mmd,
    'rf_dim_mmd'                    : args.rf_dim_mmd
    }
print("latent_space_args: ", latent_space_args)

# load dataset
tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50', cache_dir=args.cache_dir)
tokenizer.add_special_tokens({"cls_token": "<cls>"})
assert tokenizer.cls_token == "<cls>"

train_dataset = CustomStabilityDatasetForGenLatentSpace(args.data_dir, args.train_split_name, train_ratio=args.train_ratio,
                                                        tokenizer=tokenizer)
eval_dataset = CustomStabilityDatasetForGenLatentSpace(args.data_dir, args.eval_split_name, train_ratio=args.train_ratio,
                                                       tokenizer=tokenizer)
# Train data set-up
train_loader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True,
                          num_workers=0, collate_fn=train_dataset.collate_fn)
# Eval data set-up
eval_loader = DataLoader(eval_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False,
                         num_workers=0, collate_fn=train_dataset.collate_fn)
num_training_steps = args.num_train_epochs*len(train_dataset)//args.per_device_train_batch_size

model = MT5ForConditionalGenerationWithLatentSpace.from_pretrained(args.pretrained_dir,
                                                                   cache_dir=args.cache_dir,
                                                                   num_layers=args.num_layers,
                                                                   num_decoder_layers=args.num_decoder_layers,
                                                                   **latent_space_args)
print('='*50, 'parallelize', '='*50)
model.parallelize()
model.resize_token_embeddings(len(tokenizer))

# optimizer set up
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {
        'params'      : [p for n, p in model.named_parameters() if ('mi_head' not in n and not any(nd in n for nd in no_decay))],
        'weight_decay': 0.01
        },
    {
        'params'      : [p for n, p in model.named_parameters() if ('mi_head' not in n and any(nd in n for nd in no_decay))],
        'weight_decay': 0.0
        }
    ]
optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

# lr scheduling
lr_scheduler = get_scheduler(
        'linear',
        optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=num_training_steps,
        )

if args.do_mi:
    mi_optimizer_grouped_parameters = [
        {
            'params'      : [p for n, p in model.named_parameters() if ('mi_head' in n and not any(nd in n for nd in no_decay))],
            'weight_decay': 0.01
            },
        {
            'params'      : [p for n, p in model.named_parameters() if ('mi_head' in n and any(nd in n for nd in no_decay))],
            'weight_decay': 0.0
            }
        ]
    
    mi_optimizer = AdamW(mi_optimizer_grouped_parameters, lr=args.lr)
    mi_lr_scheduler = get_scheduler(
            'linear',
            mi_optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=num_training_steps,
            )

start_epoch = 1
global_step = 0
n_iter = int(args.num_train_epochs*len(train_loader))
print("len_train_loader: ", len(train_loader))
print("num_train_epochs: ", args.num_train_epochs)
print("n_iter: ", n_iter)

beta_t_list = frange_cycle_zero_linear(n_iter, start=0.0, stop=args.beta, n_cycle=10, ratio_increase=args.beta_ratio_increase,
                                       ratio_zero=args.beta_ratio_zero)
if args.beta_start_step > 0:
    beta_t_list[:args.beta_start_step] = 0

model.train()
for epoch in tqdm(range(start_epoch, args.num_train_epochs + 1)):
    print('start_epoch:', start_epoch, 'current epoch:', epoch, 'global_step:', global_step)
    for step, batch in enumerate(tqdm(train_loader)):
        input_ids = batch['input_ids'].to(model.device)
        labels = batch['labels'].to(model.device)
        ddG_targets = batch['ddG'].to(model.device) if batch['ddG'] is not None else None
        solubility_targets = batch['solubility'].to(model.device) if batch['solubility'] is not None else None
        contrast_targets = [ddG_targets, solubility_targets]
        model.zero_grad()
        
        ''' no use of mi_head'''
        if args.do_mi:
            # train mi_head
            mi_head_loss = model(input_ids, labels=labels, contrast_targets=contrast_targets, train_mi_head_step=True)
            mi_head_loss.backward()
            mi_optimizer.step()
            mi_lr_scheduler.step()
            
            model.zero_grad()
            # train generator model
            model_outputs = model(input_ids, labels=labels, contrast_targets=contrast_targets)
            outputs, contrastive_loss, contrastive_value, mi_head_loss = model_outputs[0], model_outputs[1], model_outputs[2], \
                model_outputs[3]
            lm_loss = outputs.loss
            
            total_loss = lm_loss
            if args.lambda_contrastive > 0:
                # set to same device
                if total_loss.device != contrastive_loss.device:
                    contrastive_loss = contrastive_loss.to(total_loss.device)
                if total_loss.device != mi_head_loss.device:
                    mi_head_loss = mi_head_loss.to(total_loss.device)
                
                total_loss = total_loss + args.lambda_contrastive*contrastive_loss - args.lambda_mi_head_loss*mi_head_loss  # generator
                # optimized to increase mi_head_loss
        else:
            model_outputs = model(input_ids, labels=labels, contrast_targets=contrast_targets)
            outputs, contrastive_loss, contrastive_value = model_outputs[0], model_outputs[1], model_outputs[2]
            lm_loss = outputs.loss
            
            '''reconstruction loss'''
            total_loss = lm_loss
            
            '''contrastive loss'''
            if args.lambda_contrastive > 0:
                if total_loss.device != contrastive_loss.device:
                    contrastive_loss = contrastive_loss.to(total_loss.device)
                total_loss = total_loss + args.lambda_contrastive*contrastive_loss
        
        if latent_space_type in ['vae', 'wae']:
            z_regu_output = model_outputs[-1]
            if type(z_regu_output) is dict:
                z_regu_loss = z_regu_output['z_regu_loss']
            else:
                z_regu_loss = z_regu_output
            
            if args.use_beta_schedule and global_step < len(beta_t_list):
                '''no use here, use_beta_schedule == False'''
                beta_z_regu = beta_t_list[global_step]
            else:
                if args.beta_start_step > 0 and global_step < args.beta_start_step:
                    beta_z_regu = 0
                else:
                    beta_z_regu = args.beta  # constant value
            if beta_z_regu > 0:
                if total_loss.device != z_regu_loss.device:
                    z_regu_loss = z_regu_loss.to(total_loss.device)
                
                '''smoothness loss'''
                total_loss = total_loss + beta_z_regu*z_regu_loss
            
            if wae_z_enc_type != 'deterministic':
                '''no use here, wae_z_enc_type == deterministic'''
                z_logvar_L1 = z_regu_output['z_logvar_L1']
                if args.lambda_logvar_L1 > 0 and beta_z_regu > 0:
                    if total_loss.device != z_logvar_L1.device:
                        z_logvar_L1 = z_logvar_L1.to(total_loss.device)
                    
                    # prevent z_logvar from being too large
                    total_loss = total_loss + beta_z_regu*args.lambda_logvar_L1*z_logvar_L1
                
                z_logvar_KL_penalty = z_regu_output['z_logvar_KL_penalty']
                if args.lambda_logvar_KL > 0 and beta_z_regu > 0:
                    if total_loss.device != z_logvar_KL_penalty.device:
                        z_logvar_KL_penalty = z_logvar_KL_penalty.to(total_loss.device)
                    
                    # prevent z_logvar from diminishing
                    total_loss = total_loss + beta_z_regu*args.lambda_logvar_KL*z_logvar_KL_penalty
        
        # perturb cycle consistency loss
        '''no use here, args.lambda_contrastive_perturb_cyc == 0.0'''
        if args.lambda_contrastive_perturb_cyc > 0 and global_step > args.contrastive_perturb_cyc_start_step:
            if args.pc_perturb_type == 'std':
                contrastive_value_std = torch.std(contrastive_value)
                pc_perturb = args.pc_perturb*contrastive_value_std.item()
            elif args.pc_perturb_type == 'static':
                pc_perturb = args.pc_perturb
            
            gen_output = model.generate(input_ids, max_length=input_ids.shape[-1] + 1, return_dict_in_generate=True, output_scores=True,
                                        z_tar_edit_before_dec=pc_perturb)  # change z_tar_edit_before_dec
            gen_logits = torch.stack(gen_output.scores, dim=1)
            pc_gen_value_pred = model(inputs_logits=gen_logits, return_only_value_pred=True)
            
            if len(ddG_targets.shape) != 2:
                ddG_targets = torch.unsqueeze(ddG_targets, dim=-1)
            pc_gen_ddG_targets = ddG_targets + pc_perturb
            pc_ddG_labels = torch.sign(ddG_targets - pc_gen_ddG_targets)*0.5 + 0.5
            contrastive_preds = F.logsigmoid(contrastive_value - pc_gen_value_pred)
            inverse_preds = F.logsigmoid(-1*(contrastive_value - pc_gen_value_pred))
            if pc_ddG_labels.device != contrastive_preds.device:
                pc_ddG_labels = pc_ddG_labels.to(contrastive_preds.device)
            pc_losses = -pc_ddG_labels*contrastive_preds - (1 - pc_ddG_labels)*inverse_preds
            contrastive_perturb_cyc_loss = pc_losses.mean()
            
            if total_loss.device != contrastive_perturb_cyc_loss.device:
                contrastive_perturb_cyc_loss = contrastive_perturb_cyc_loss.to(total_loss.device)
            total_loss = total_loss + args.lambda_contrastive_perturb_cyc*contrastive_perturb_cyc_loss
        
        if args.lambda_contrastive_cyc > 0 and global_step > args.contrastive_cyc_start_step:
            model_outputs_2nd_forward = model(inputs_logits=outputs.logits, labels=labels, contrast_targets=contrast_targets)
            contrastive_cyc_loss = model_outputs_2nd_forward[1]
            
            if total_loss.device != contrastive_cyc_loss.device:
                contrastive_cyc_loss = contrastive_cyc_loss.to(total_loss.device)
            
            '''cycle consistency loss'''
            total_loss = total_loss + args.lambda_contrastive_cyc*contrastive_cyc_loss
        
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()
        global_step += 1
        
        if global_step%args.logging_steps == 0:
            tb_writer.add_scalar("TRAIN/lr", lr_scheduler.get_last_lr()[0], global_step)
            tb_writer.add_scalar("TRAIN/contrastive_loss", contrastive_loss, global_step)
            tb_writer.add_scalar("TRAIN/lm_loss", lm_loss, global_step)
            if args.do_mi:
                tb_writer.add_scalar("TRAIN/mi_head_loss", mi_head_loss, global_step)
            if latent_space_type in ['vae', 'wae']:
                tb_writer.add_scalar("TRAIN/z_regu_loss", z_regu_loss, global_step)
                tb_writer.add_scalar("TRAIN/beta_z_regu", beta_z_regu, global_step)
                if wae_z_enc_type != 'deterministic':
                    tb_writer.add_scalar("TRAIN/z_logvar_L1", z_logvar_L1, global_step)
                    tb_writer.add_scalar("TRAIN/z_logvar_KL_penalty", z_logvar_KL_penalty, global_step)
            
            if args.lambda_contrastive_cyc > 0 and global_step > args.contrastive_cyc_start_step:
                tb_writer.add_scalar("TRAIN/contrastive_cyc_loss", contrastive_cyc_loss, global_step)
            
            if args.lambda_contrastive_perturb_cyc > 0 and global_step > args.contrastive_perturb_cyc_start_step:
                tb_writer.add_scalar("TRAIN/contrastive_perturb_cyc_loss", contrastive_perturb_cyc_loss, global_step)
        
        if global_step%args.eval_steps == 0:
            model.eval()
            eval_output = evaluate(model, eval_loader, do_mi=args.do_mi, latent_space_type=latent_space_type)
            eval_lm_loss, eval_contrastive_loss, eval_ddG_spearmanr, eval_solubility_spearmanr = eval_output['lm_loss'], eval_output[
                'contrastive_loss'], eval_output['ddG_spearmanr'], eval_output['solubility_spearmanr']
            tb_writer.add_scalar("EVAL/lm_loss", eval_lm_loss, global_step)
            tb_writer.add_scalar("EVAL/contrastive_loss", eval_contrastive_loss, global_step)
            tb_writer.add_scalar("EVAL/ddG_spearmanr", eval_ddG_spearmanr, global_step)
            tb_writer.add_scalar("EVAL/solubility_spearmanr", eval_solubility_spearmanr, global_step)
            if args.do_mi:
                eval_mi_head_loss = eval_output['mi_head_loss']
                tb_writer.add_scalar("EVAL/mi_head_loss", eval_mi_head_loss, global_step)
            if latent_space_type in ['vae', 'wae']:
                eval_z_regu_loss = eval_output['z_regu_loss']
                tb_writer.add_scalar("EVAL/z_regu_loss", eval_z_regu_loss, global_step)
            model.train()
        
        if global_step%args.save_steps == 0:
            print()
            print('========== save step checkpoint ==========')
            weights_name = "pytorch_model.bin"
            ckpt_saved_dir = os.path.join(output_dir, f'step_{global_step}')
            mkdir(ckpt_saved_dir)
            saved_weights_file = os.path.join(ckpt_saved_dir, weights_name)
            torch.save(model.state_dict(), saved_weights_file)
            torch.save(optimizer.state_dict(), ckpt_saved_dir + "/optimizer.pt")
            torch.save(lr_scheduler.state_dict(), ckpt_saved_dir + "/scheduler.pt")
            torch.save(args, ckpt_saved_dir + "/training_args.bin")
            ckpt_info = {'epoch': epoch, 'global_step': global_step}
            torch.save(ckpt_info, ckpt_saved_dir + "/ckpt_info.pt")
            shutil.copy(args.src_json, ckpt_saved_dir)

'''
Training Finished
'''

# Final log step
tb_writer.add_scalar("TRAIN/lr", lr_scheduler.get_last_lr()[0], global_step)
tb_writer.add_scalar("TRAIN/contrastive_loss", contrastive_loss, global_step)
tb_writer.add_scalar("TRAIN/lm_loss", lm_loss, global_step)
if args.do_mi:
    tb_writer.add_scalar("TRAIN/mi_head_loss", mi_head_loss, global_step)

if latent_space_type in ['vae', 'wae']:
    tb_writer.add_scalar("TRAIN/z_regu_loss", z_regu_loss, global_step)
    tb_writer.add_scalar("TRAIN/beta_z_regu", beta_z_regu, global_step)
    if wae_z_enc_type != 'deterministic':
        tb_writer.add_scalar("TRAIN/z_logvar_L1", z_logvar_L1, global_step)
        tb_writer.add_scalar("TRAIN/z_logvar_KL_penalty", z_logvar_KL_penalty, global_step)

if args.lambda_contrastive_cyc > 0 and global_step > args.contrastive_cyc_start_step:
    tb_writer.add_scalar("TRAIN/contrastive_cyc_loss", contrastive_cyc_loss, global_step)

if args.lambda_contrastive_perturb_cyc > 0 and global_step > args.contrastive_perturb_cyc_start_step:
    tb_writer.add_scalar("TRAIN/contrastive_perturb_cyc_loss", contrastive_perturb_cyc_loss, global_step)

'''
Final evaluation
'''
model.eval()
eval_output = evaluate(model, eval_loader, do_mi=args.do_mi, latent_space_type=latent_space_type)
eval_lm_loss, eval_contrastive_loss, eval_ddG_spearmanr, eval_solubility_spearmanr = eval_output['lm_loss'], eval_output[
    'contrastive_loss'], eval_output['ddG_spearmanr'], eval_output['solubility_spearmanr']
tb_writer.add_scalar("EVAL/lm_loss", eval_lm_loss, global_step)
tb_writer.add_scalar("EVAL/contrastive_loss", eval_contrastive_loss, global_step)
tb_writer.add_scalar("EVAL/ddG_spearmanr", eval_ddG_spearmanr, global_step)
tb_writer.add_scalar("EVAL/solubility_spearmanr", eval_solubility_spearmanr, global_step)
if args.do_mi:
    eval_mi_head_loss = eval_output['mi_head_loss']
    tb_writer.add_scalar("EVAL/mi_head_loss", eval_mi_head_loss, global_step)
if latent_space_type in ['vae', 'wae']:
    eval_z_regu_loss = eval_output['z_regu_loss']
    tb_writer.add_scalar("EVAL/z_regu_loss", eval_z_regu_loss, global_step)

results_txt_name = "eval_results.txt"
results_path = output_dir/results_txt_name
with open(results_path, "w") as writer:
    for key in sorted(eval_output.keys()):
        writer.write("%s = %s\n"%(key, str(eval_output[key])))

weights_name = "pytorch_model.bin"
saved_weights_file = output_dir/weights_name
torch.save(model.state_dict(), saved_weights_file)
torch.save(optimizer.state_dict(), output_dir/"optimizer.pt")
torch.save(lr_scheduler.state_dict(), output_dir/"scheduler.pt")
torch.save(args, output_dir/"training_args.bin")
ckpt_info = {'epoch': epoch, 'global_step': global_step}
torch.save(ckpt_info, output_dir/"ckpt_info.pt")
shutil.copy(args.src_json, output_dir)
print('Training finished')
