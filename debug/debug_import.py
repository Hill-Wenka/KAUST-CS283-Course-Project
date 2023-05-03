import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("sys.path: ", sys.path)

from transformers import MT5ForConditionalGeneration, MT5Config, Trainer, TrainingArguments, T5Tokenizer
from transformers_custom import MT5ForConditionalGeneration

print(1.001 ** (209*1), 1.001 ** (209*2), 1.001 ** (209*3), 1.001 ** (209*4))
print(1.002 ** (209*1), 1.002 ** (209*2), 1.002 ** (209*3), 1.002 ** (209*4))
print(1.003 ** (209*1), 1.003 ** (209*2), 1.003 ** (209*3), 1.003 ** (209*4))
print(1.004 ** (209*1), 1.004 ** (209*2), 1.004 ** (209*3), 1.004 ** (209*4))
print(1.005 ** (209*1), 1.005 ** (209*2), 1.005 ** (209*3), 1.005 ** (209*4))
