import json
import argparse
from tqdm import tqdm

from datasets import load_dataset
from tokenizers import SentencePieceBPETokenizer
from transformers import LlamaTokenizerFast, TrainingArguments, AutoTokenizer

mixtral_tokenizer = AutoTokenizer.from_pretrained("new-llama-tokenizer")
data_text="key => value <sep> key2 => value2"
print(f"Tokenized by Mixtral tokenizer: {mixtral_tokenizer.tokenize(data_text)}")

        