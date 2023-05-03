'''
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print("sys.path: ", sys.path)

import re
import shutil
import torch
import numpy as np
import random
import pandas as pd
import argparse
import pickle
import time
from transformers import T5Tokenizer
from transformers_custom import MT5ForConditionalGenerationWithLatentSpace
from pathlib import Path
from tqdm import tqdm

# argparse
parser = argparse.ArgumentParser()

parser.add_argument('--seed', action='store', type=int, default=42, help='random seed')
parser.add_argument('--num_generations', action='store', type=int, default=None, help='(min) number of generation')
parser.add_argument('--generation_output_dir', action='store', type=str, default="")
parser.add_argument('--prepend_output_name', action='store', type=str, default="")
parser.add_argument('--unique_gen', action='store_true')
parser.add_argument('--ckpt_file', type=str, default=None, help='generation checkpoint pkl file')

# generator args
parser.add_argument('--gen_pretrained_dir', action='store', type=str, default="")
parser.add_argument('--temperature_init', action='store', type=float, default=1.0)
# parser.add_argument('--temperature_multiple', action='store', type=float, default=1.2) # update per epoch
parser.add_argument('--temperature_multiple', action='store', type=float, default=1.004)  # update per step
parser.add_argument('--patience', action='store', type=int, default=50,
                    help='number of repeats before increasing temperature values for gen decoding')
parser.add_argument('--batch_repeat_threshold', action='store', type=int, default=4)
parser.add_argument('--gen_batch_size', action='store', type=int, default=200)
parser.add_argument('--gen_save_interval', action='store', type=int, default=100, help='interval to save generations')
parser.add_argument('--skip_gen', action='store_true')

parser.add_argument('--gen_token_len', action='store', type=int, default=83 + 2 + 1, help='len to check for generated tokens')

# new controlled gen args
parser.add_argument('--input_data_dir', action='store', type=str, default="/home/hew/storage/storage/genhance/data",
                    help='data for generator input seqs')
parser.add_argument('--src_config_json', action='store', type=str,
                    default="/home/hew/python/genhance/temp/config.json")
parser.add_argument('--topk_as_input', action='store', type=int, default=12500,
                    help='top K most stable sequences to use input for generation')
parser.add_argument('--num_gen_samples_per_input', action='store', type=int, default=20, help='number of generation per input sequence')

# latent space args
parser.add_argument('--z_tar_vector_dim', action='store', type=int, default=1)
parser.add_argument('--z_tar_edit_before_dec', action='store', type=float, nargs='+', default=None,
                    help='perturbation to latent vector z_tar')

# self-defined args
parser.add_argument('--property', type=str, default='ddG', choices=['ddG', 'solubility', 'ddG_solubility'],
                    help='property to be optimized')
parser.add_argument('--cuda_device', type=str, default='0', help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--datafile', type=str, default="train_new_tophalf_ddG.pkl", help='pkl data file')

args = parser.parse_args()

'''resume args'''
print("args: ", args)
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
print('='*100)
# 除了以上参数之外，几乎都要沿用ckpt_args，保持一致性
ckpt_args = torch.load(args.gen_pretrained_dir + '/training_args.bin')
print("ckpt_args: ", ckpt_args)
'''resume args'''

print('='*100)
if ckpt_args.property != args.property:
    print('change optimization property from {} to {}'.format(ckpt_args.property, args.property))
    ckpt_args.property = args.property

if ckpt_args.property == 'ddG':
    args.z_tar_vector_dim = 1
    ckpt_args.z_tar_vector_dim = 1
elif ckpt_args.property == 'solubility':
    args.z_tar_vector_dim = 1
    ckpt_args.z_tar_vector_dim = 1
elif ckpt_args.property == 'ddG_solubility':
    args.z_tar_vector_dim = 2
    ckpt_args.z_tar_vector_dim = 2
else:
    raise ValueError(ckpt_args.property)
print('updated property: ', ckpt_args.property)
print('updated z_tar_vector_dim: ', ckpt_args.z_tar_vector_dim)

# assert len(args.z_tar_edit_before_dec) == ckpt_args.z_tar_vector_dim
print('z_tar_edit_before_dec', type(args.z_tar_edit_before_dec), args.z_tar_edit_before_dec)
if type(args.z_tar_edit_before_dec) == list and len(args.z_tar_edit_before_dec) == 1:
    args.z_tar_edit_before_dec = args.z_tar_edit_before_dec[0]
print('z_tar_edit_before_dec: ', args.z_tar_edit_before_dec)
print('='*100)

latent_space_args = {
    'latent_pooler'                 : ckpt_args.latent_pooler,
    'pool_enc_hidden_states_for_dec': ckpt_args.pool_enc_hidden_states_for_dec,
    'mask_non_target_z_vector'      : ckpt_args.mask_non_target_z_vector,
    'separate_targetattr_head'      : ckpt_args.separate_targetattr_head,
    'z_tar_vector_dim'              : ckpt_args.z_tar_vector_dim,  # can only change this
    'do_mi'                         : ckpt_args.do_mi,
    'latent_space_type'             : ckpt_args.latent_space_type,
    'latent_size'                   : ckpt_args.latent_size,
    'separate_latent_enc'           : ckpt_args.separate_latent_enc,
    'separate_latent_dec'           : ckpt_args.separate_latent_dec,
    'wae_z_enc_type'                : ckpt_args.wae_z_enc_type,
    }

if not os.path.isfile(os.path.join(args.gen_pretrained_dir, 'config.json')):
    shutil.copy(args.src_config_json, args.gen_pretrained_dir)

output_dir = Path(args.generation_output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

wt_seq = 'STIEEQAKTFLDKFNHEAEDLFYQSSLASWNYNTNITEENVQNMNNAGDKWSAFLKEQSTLAQMYPLQEIQNLTVKLQLQALQ'
constant_region = 'NTNITEEN'

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

# Set up generator model
tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", cache_dir=ckpt_args.cache_dir)
tokenizer.add_special_tokens({"cls_token": "<cls>"})
assert tokenizer.cls_token == "<cls>"

gen_model = MT5ForConditionalGenerationWithLatentSpace.from_pretrained(args.gen_pretrained_dir,
                                                                       cache_dir=ckpt_args.cache_dir,
                                                                       num_layers=ckpt_args.num_layers,
                                                                       num_decoder_layers=ckpt_args.num_decoder_layers if
                                                                       'num_decoder_layers' in ckpt_args else
                                                                       ckpt_args.num_decode_layers,
                                                                       **latent_space_args)
gen_model.parallelize()
gen_model.resize_token_embeddings(len(tokenizer))

# Set up input data
input_data_path = Path(args.input_data_dir)
# input_data_file = f'train_new_tophalf_ddG_solubility.pkl'
input_data_file = args.datafile
input_data_file = input_data_path/input_data_file
input_data_df = pd.read_pickle(input_data_file)
train_seq_list = input_data_df['MT_seq'].tolist()

if 'ddG' in args.property:
    print("ddG stats of input data")
    print("min: ", np.min(input_data_df['ddG']))
    print("mean: ", np.mean(input_data_df['ddG']))
    print("median: ", np.median(input_data_df['ddG']))
    print("max: ", np.max(input_data_df['ddG']))
    sorted_input_df = input_data_df.sort_values(by='ddG', ascending=True)
    gen_input_df = sorted_input_df.iloc[:args.topk_as_input]
if 'solubility' in args.property:
    print("solubility stats of input data")
    print("min: ", np.min(input_data_df['solubility']))
    print("mean: ", np.mean(input_data_df['solubility']))
    print("median: ", np.median(input_data_df['solubility']))
    print("max: ", np.max(input_data_df['solubility']))
    sorted_input_df = input_data_df.sort_values(by='solubility', ascending=False)
    gen_input_df = sorted_input_df.iloc[:args.topk_as_input]
if 'ddG_solubility' in args.property:
    print("avg rank stats of input data")
    print("min: ", np.min(input_data_df['avg_rank']))
    print("mean: ", np.mean(input_data_df['avg_rank']))
    print("median: ", np.median(input_data_df['avg_rank']))
    print("max: ", np.max(input_data_df['avg_rank']))
    sorted_input_df = input_data_df.sort_values(by='avg_rank', ascending=True)
    gen_input_df = sorted_input_df.iloc[:args.topk_as_input]

print(sorted_input_df.head())

# gen code - start
if args.num_generations is None:
    args.num_generations = args.topk_as_input*args.num_gen_samples_per_input
num_unique_seqs_per_batch = args.gen_batch_size//args.num_gen_samples_per_input
num_batch = len(gen_input_df)//num_unique_seqs_per_batch
if len(gen_input_df)%num_unique_seqs_per_batch != 0:
    num_batch += 1

print("="*100)
print('generation conifg:')
print('num_generations: ', args.num_generations)
print('topk_as_input: ', args.topk_as_input)
print('num_gen_samples_per_input: ', args.num_gen_samples_per_input)
print('num_unique_seqs_per_batch: ', num_unique_seqs_per_batch)
print('num_batch: ', num_batch)
print("="*100)

output_seq_list = []
input_seq_list = []
output_tensor_list = []
repeat_list = []
in_train_data_list = []
unique_n_notrain_list = []
start_time = time.time()
prev_save_path = None
repeat_seq_count = 0
in_train_count = 0
temperature = args.temperature_init
generation_rounds_done = 0
num_cls_seq = 0
num_fail_seq = 0
num_no_cons_seq = 0
bad_words_ids = [0, 2] + list(range(23, 127 + 1))  # <cls>和</s>必须保留，因为train的时候包含，所以生成的序列跟train的序列分布式一样的
bad_words_ids = [[x] for x in bad_words_ids]

min_increase_value = args.gen_batch_size/2
increase_queue = [min_increase_value]*10

if not args.skip_gen:
    print("start generation")
    if args.ckpt_file is not None:
        print("resume ckpt data")
        with open(os.path.join(args.ckpt_file), 'rb') as f:
            saved_dict = pickle.load(f)
        output_seq_list = saved_dict['output_seq_list']
        input_seq_list = saved_dict['input_seq_list']
        output_tensor_list = saved_dict['output_tensor_list']
        repeat_list = saved_dict['repeat_list']
        in_train_data_list = saved_dict['in_train_data_list']
        unique_n_notrain_list = [True]*len(output_seq_list)  # all unique by default since args.unique_gen is True
    
    gen_model.eval()
    while args.unique_gen and np.sum(unique_n_notrain_list) < args.num_generations:
        if generation_rounds_done > 0:
            print('='*100)
            print("New generation round, temperature: ", temperature, "num_unique_n_notrain_list: ", np.sum(unique_n_notrain_list))
            print('='*100)
        
        for batch_ind in tqdm(range(num_batch)):
            batch_seqs = gen_input_df[batch_ind*num_unique_seqs_per_batch: (batch_ind + 1)*num_unique_seqs_per_batch]['MT_seq']
            batch_input_ids = []
            batch_input_seqs = []
            for seq in batch_seqs:
                batch_input_seqs = batch_input_seqs + [seq]*args.num_gen_samples_per_input
                seq = '<cls> ' + " ".join(list(re.sub(r"[UZOB]", "X", seq)))
                input_ids = tokenizer.encode(seq, return_tensors='pt').to(gen_model.device)
                repeated_input_ids = input_ids.repeat((args.num_gen_samples_per_input, 1))
                batch_input_ids.append(repeated_input_ids)
            
            batch_input_ids = torch.cat(batch_input_ids, dim=0)
            
            # print("batch_input_ids.shape: ", batch_input_ids.shape)  # torch.Size([200, 85])
            # print(batch_input_ids[0, :]) # '<cls> '有没有空格不影响结果, [128, ... (83 AA tokens) , 1]
            gen_output = gen_model.generate(batch_input_ids,
                                            min_length=83 + 2 + 1,
                                            max_length=83 + 2 + 1,
                                            do_sample=True,
                                            temperature=temperature,
                                            bad_words_ids=bad_words_ids,
                                            z_tar_edit_before_dec=args.z_tar_edit_before_dec)
            # print("gen_output.shape: ", gen_output.shape)  # torch.Size([batch_size, 84]), 84 = <pad> + 83 amino acid tokens
            
            batch_valid_seqs = 0
            for seq_ind, gen_seq in enumerate(gen_output.cpu().numpy()):
                unique_n_notrain = True
                repeat = False
                in_train_data = False
                
                tokens = tokenizer.convert_ids_to_tokens(gen_seq.tolist())  # list of tokens
                # print("len(tokens): ", len(tokens)) # 84, <pad> + 83 amino acid tokens
                if tokens == None or len(tokens) != args.gen_token_len:
                    continue
                
                str_token_seq = "".join(tokens[2:-1]).replace('▁', '')
                # 头两个和结尾一个的token是<pad><cls>和</s>对应的位置，无论生成的序列是否是特殊token，都要去掉
                
                if constant_region not in str_token_seq:
                    num_no_cons_seq += 1
                
                if num_fail_seq%2000 == 0 or np.sum(unique_n_notrain_list)%2000 == 0:
                    print("num failed gen: {},".format(num_fail_seq),
                          "num no constance gen: {},".format(num_no_cons_seq),
                          "repeat gen: {},".format(repeat_seq_count),
                          "in train gen: {},".format(in_train_count),
                          "num valid gen: {},".format(np.sum(unique_n_notrain_list)),
                          "gen/total: {:.2f}%".format(np.sum(unique_n_notrain_list)/args.num_generations*100))
                    print('[example gen seq]:', str_token_seq)
                    print('[ wilde type seq]:', wt_seq)
                
                if str_token_seq in output_seq_list:
                    repeat_seq_count += 1
                    repeat = True
                    unique_n_notrain = False
                
                if str_token_seq in train_seq_list:
                    in_train_count += 1
                    in_train_data = True
                    unique_n_notrain = False
                
                if args.unique_gen and not unique_n_notrain:
                    num_fail_seq += 1
                    continue
                
                unique_n_notrain_list.append(unique_n_notrain)
                repeat_list.append(repeat)
                in_train_data_list.append(in_train_data)
                
                input_seq_str = batch_input_seqs[seq_ind]
                input_seq_list.append(input_seq_str)
                output_seq_list.append(str_token_seq)
                
                seq_tensor = gen_output[seq_ind].detach().cpu()
                output_tensor_list.append(seq_tensor)
                
                batch_valid_seqs += 1
            
            increase_queue.append(batch_valid_seqs)
            
            temp_avg_increase = np.mean(increase_queue[-10:])
            if temp_avg_increase < min_increase_value:
                temperature = temperature*args.temperature_multiple
                print('temp_avg_increase: {} < {}, increase temperature to: {}'.format(temp_avg_increase, min_increase_value, temperature))
            if batch_ind%10 == 0:
                print('current temperature:', temperature)
            
            if batch_ind%args.gen_save_interval == 0 and batch_ind != 0:
                save_path = os.path.join(args.generation_output_dir,
                                         "{}-gens-{}-{}.pkl".format(args.prepend_output_name,
                                                                    len(output_seq_list),
                                                                    args.num_generations))
                saved_dict = {
                    'output_seq_list': output_seq_list, "input_seq_list": input_seq_list, "output_tensor_list": output_tensor_list,
                    'repeat_list'    : repeat_list, 'in_train_data_list': in_train_data_list
                    }
                with open(save_path, 'wb') as f:
                    pickle.dump(saved_dict, f)
                cur_time = time.time()
                
                print('='*50, 'interval save', '='*50)
                print("generated #", len(output_seq_list))
                print("Time taken so far: {} hours".format((cur_time - start_time)/3600))
                print('='*50, 'interval save', '='*50)
                
                if prev_save_path is not None:
                    os.remove(prev_save_path)
                prev_save_path = save_path
            
            if args.unique_gen and np.sum(unique_n_notrain_list) > args.num_generations:
                break
        generation_rounds_done += 1
    
    '''Save Final Data'''
    save_path = os.path.join(args.generation_output_dir, "{}-gens-{}.pkl".format(args.prepend_output_name,
                                                                                 args.num_generations))
    saved_dict = {
        'output_seq_list': output_seq_list, "input_seq_list": input_seq_list, "output_tensor_list": output_tensor_list,
        'repeat_list'    : repeat_list, 'in_train_data_list': in_train_data_list
        }
    with open(save_path, 'wb') as f:
        pickle.dump(saved_dict, f)
    
    if prev_save_path is not None:
        os.remove(prev_save_path)
else:
    print("skip generation and load from saved pkl")
    save_path = os.path.join(args.generation_output_dir, "{}-gens-{}.pkl".format(args.prepend_output_name,
                                                                                 args.num_generations))
    with open(save_path, 'rb') as f:
        saved_dict = pickle.load(f)
    output_seq_list = saved_dict['output_seq_list']
    input_seq_list = saved_dict['input_seq_list']
    output_tensor_list = saved_dict['output_tensor_list']
    repeat_list = saved_dict['repeat_list']
    in_train_data_list = saved_dict['in_train_data_list']
    # temperature = saved_dict['temperature']

print()
print('='*100)
print('Generation Done, Forwards Pass on Latent Head')
print('='*100)
print()

gen_tensors = torch.stack(output_tensor_list, dim=0)
print("gen_tensors.shape: ", gen_tensors.shape)

# Latent Head inference - start
ddG_latent_head_pred_list = []
solubility_latent_head_pred_list = []

num_disc_batch = len(gen_tensors)//args.gen_batch_size
if len(gen_tensors)%args.gen_batch_size != 0:
    num_disc_batch += 1
print('num_disc_batch', num_disc_batch)

start_time = time.time()
gen_model.eval()
with torch.no_grad():
    for batch_ind in tqdm(range(num_disc_batch)):
        gen_tensor_batch = gen_tensors[batch_ind*args.gen_batch_size: (batch_ind + 1)*args.gen_batch_size, 1:]
        gen_tensor_batch = gen_tensor_batch.to(gen_model.device)
        model_outputs = gen_model(gen_tensor_batch, labels=gen_tensor_batch)
        contrastive_value = model_outputs[1]
        if args.property == 'ddG':
            ddG_latent_head_pred_list.append(contrastive_value[:, 0].squeeze().cpu().numpy())
            solubility_latent_head_pred_list.append([None]*contrastive_value.shape[0])
        elif args.property == 'solubility':
            ddG_latent_head_pred_list.append([None]*contrastive_value.shape[0])
            solubility_latent_head_pred_list.append(contrastive_value[:, 0].squeeze().cpu().numpy())
        else:
            ddG_latent_head_pred_list.append(contrastive_value[:, 0].squeeze().cpu().numpy())
            solubility_latent_head_pred_list.append(contrastive_value[:, 1].squeeze().cpu().numpy())

'''Save Final Data'''
ddG_latent_head_pred_list = np.concatenate(ddG_latent_head_pred_list, axis=None).tolist()
solubility_latent_head_pred_list = np.concatenate(solubility_latent_head_pred_list, axis=None).tolist()
save_path = os.path.join(args.generation_output_dir, "{}-latent-{}.pkl".format(args.prepend_output_name,
                                                                               args.num_generations))
with open(save_path, 'wb') as f:
    pickle.dump({'ddG_latent_pred': ddG_latent_head_pred_list, 'solubility_latent_pred': solubility_latent_head_pred_list}, f)

# Latent Head inference - end

print('ddG_latent_head_pred_list', len(ddG_latent_head_pred_list))
print('solubility_latent_head_pred_list', len(solubility_latent_head_pred_list))
print('output_seq_list', len(output_seq_list))

# Save generated samples into TSV file
# PDB, Chain, Start_index, WT_seq, MT_seq
PDB = 'template2.pdb'
Chain = 'A'
Start_index = 19
WT_seq = 'STIEEQAKTFLDKFNHEAEDLFYQSSLASWNYNTNITEENVQNMNNAGDKWSAFLKEQSTLAQMYPLQEIQNLTVKLQLQALQ'

df = pd.DataFrame()
df['MT_seq'] = output_seq_list
df['ddG_latent_pred'] = ddG_latent_head_pred_list
df['solubility_latent_pred'] = solubility_latent_head_pred_list
df['gen_input_seq'] = input_seq_list
df['PDB'] = PDB
df['Chain'] = Chain
df['Start_index'] = Start_index
df['WT_seq'] = WT_seq
df['repeated_gen'] = repeat_list
df['in_train_data_gen'] = in_train_data_list

# Latent head-predicted most stable ones first
if args.property == 'ddG':
    df = df.sort_values(by='ddG_latent_pred', ascending=True)
elif args.property == 'solubility':
    df = df.sort_values(by='solubility_latent_pred', ascending=False)
else:
    df['ddG_latent_pred_rank'] = df['ddG_latent_pred'].rank(ascending=True)
    df['solubility_latent_pred_rank'] = df['solubility_latent_pred'].rank(ascending=False)
    df['avg_latent_head_pred_rank'] = (df['ddG_latent_pred_rank'] + df['solubility_latent_pred_rank'])/2
    df = df.sort_values(by='avg_latent_head_pred_rank', ascending=True)
tsv_name = os.path.join(args.generation_output_dir, "{}-gens-{}.tsv".format(args.prepend_output_name,
                                                                            args.num_generations))
df.to_csv(tsv_name, sep="\t", index=False)
