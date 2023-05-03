import re
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, T5Tokenizer, AlbertTokenizer

print('Loading model')
model_name = 'prot_t5_xl_uniref50'
# model_name = 'prot_t5_base_mt_uniref50'
# AutoTokenizer = AutoTokenizer.from_pretrained(f"Rostlab/{model_name}", cache_dir='/scratch/hew/genhance/pretrained/')
AlbertTokenizer = AlbertTokenizer.from_pretrained(f"Rostlab/{model_name}", cache_dir='/scratch/hew/genhance/pretrained/')
T5Tokenizer = T5Tokenizer.from_pretrained(f"Rostlab/{model_name}", cache_dir='/scratch/hew/genhance/pretrained/')
# model = AutoModelForSeq2SeqLM.from_pretrained(f"Rostlab/{model_name}", cache_dir='/scratch/hew/genhance/pretrained/')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device', device)

print('tokenization')
# print(AutoTokenizer.get_vocab())
# print(AlbertTokenizer.get_vocab())

T5Tokenizer.add_special_tokens({"cls_token": "<cls>"})
T5Tokenizer.add_special_tokens({'bos_token': '[BOS]', 'eos_token': '[EOS]'})
assert T5Tokenizer.cls_token == "<cls>"

print('T5Tokenizer.all_special_tokens', T5Tokenizer.all_special_tokens)
print('T5Tokenizer.all_special_ids', T5Tokenizer.all_special_ids)
print('T5Tokenizer.all_special_tokens', T5Tokenizer.all_special_tokens)
print('T5Tokenizer.cls_token', T5Tokenizer.cls_token)
print('T5Tokenizer.get_vocab()', T5Tokenizer.get_vocab())

sequence_examples = ["PRTEINO", "SEQWENCE"]
# this will replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
prefix = "<cls> "
# prefix = " "
sequence_examples = [prefix + " ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

# tokenize sequences and pad up to the longest sequence in the batch
AlbertTokenizer_ids = AlbertTokenizer.batch_encode_plus(sequence_examples, add_special_tokens=True, padding="longest")
T5Tokenizer_ids = T5Tokenizer.batch_encode_plus(sequence_examples, add_special_tokens=True, padding="longest")
input_ids = torch.tensor(T5Tokenizer_ids['input_ids']).to(device)
print(input_ids)

attention_mask = torch.tensor(T5Tokenizer_ids['attention_mask']).to(device)

print('AlbertTokenizer_ids', AlbertTokenizer_ids)
print('T5Tokenizer_ids', T5Tokenizer_ids)

cls_tokens = torch.ones([input_ids.shape[0], 1], dtype=torch.long).to(device)
print('cls_tokens', cls_tokens)

# model.resize_token_embeddings(len(T5Tokenizer))
#
# # generate embeddings
# model = model.to(device)
# with torch.no_grad():
#     embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask,
#                            decoder_input_ids=torch.tensor([[128], [128]], device=device))
#
# # extract embeddings for the first ([0,:]) sequence in the batch while removing padded & special tokens ([0,:7])
# emb_0 = embedding_repr.encoder_last_hidden_state[0, :7]  # shape (7 x 1024)
# print(f"Shape of per-residue embedding of first sequences: {emb_0.shape}")
# # do the same for the second ([1,:]) sequence in the batch while taking into account different sequence lengths ([1,:8])
# emb_1 = embedding_repr.encoder_last_hidden_state[1, :8]  # shape (8 x 1024)
#
# # if you want to derive a single representation (per-protein embedding) for the whole protein
# emb_0_per_protein = emb_0.mean(dim=0)  # shape (1024)
#
# print(f"Shape of per-protein embedding of first sequences: {emb_0_per_protein.shape}")
