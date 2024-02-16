import numpy as np
import torch
import argparse
import os
import random
import re
import json
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from tqdm import tqdm


# shell friendly skill names
EOF_STRINGS = ["### Response:", "\n###", "\nQUESTION", "\n---", "\nANSWER", "<|endoftext|>", "<|EOT|>"]
SEED = 17

parser = argparse.ArgumentParser()
parser.add_argument("--run_name", default='local-test', type=str)
parser.add_argument("--model_name", type=str, help="the model name", default="dsc-6.7b-instruct")
parser.add_argument("--do_sample", type=bool, default=True, help="Enable sampling")
parser.add_argument("--temperature", type=float, default=0.6, help="Sampling softmax temperature")
parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate")
parser.add_argument("--max_new_tokens", type=int, default=800, help="The maximum number of tokens to generate")
parser.add_argument("--checkpoint_path", type=str, help="the checkpoint path", default="./finetuned_models/run_name")
parser.add_argument("--load_in_8bit", action='store_true', help="load the model in 8 bits precision", default=True)
parser.add_argument("--trust_remote_code", type=bool, default=True, help="Enable `trust_remote_code`")
parser.add_argument("--skill", type=str, default="all", help="the skill to test on", choices=SKILL_LIST)
args = parser.parse_args()


def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)



# Creates the finalized prompt
# Using "### Instruction" and "### Response" for DeepSeek Coder
def create_prompt(question):
    dsc_header = "You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer."
    instruction = "Write a Python program that solves the following question."
    res = "{} \n\n### Instruction: {} \nQuestion: {} \n\n### Response:\n".format(dsc_header, instruction, question)
    return res


# gets generations, handles input encoding and output decoding
def predict(model, tokenizer, gen_config, prompt):
    set_random_seed(SEED)
    res = []



if __name__ == "__main__":

    # Load the model:

    # Load the test data:

    