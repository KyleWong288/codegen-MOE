import argparse
import json
import os
import numpy as np
import random
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

parser = argparse.ArgumentParser()
parser.add_argument("--run_name", default='mistral_test', type=str)
parser.add_argument("--model_path", default='./finetuned_models/mistral_test/checkpoint-1000', type=str)
args = parser.parse_args()


def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


if __name__ == "__main__":

    test_file = "./data/test.jsonl"
    output_file = f"./output/{args.run_name}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load the model and tokenizer:
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # Load the test data:
    test_data = load_dataset("json", data_files=test_file, split="train", download_mode="force_redownload")
    labels = test_data["labels"]
    texts = test_data["text"]

    # Evaluate:
    # each result struct has {idx, gt labels, output labels}
    res = {"results": []}
    set_random_seed(17)
    tokenized_input = tokenizer(input, truncation=True, return_tensors="pt")
    with torch.no_grad():
        output = model(**tokenized_input)
    print(output)