import argparse
import json
import numpy as np
import os
import random
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--run_name", default='roberta_test', type=str)
parser.add_argument("--model_path", default='./finetuned_models/roberta_test', type=str)
args = parser.parse_args()


def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


# extracts the highest output logits
def get_labels(probs):
    res = []
    if args.eval_type == 0:
        pass
    else:
        res = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:args.eval_type]
    return res


# eval_type is the fixed number of output labels you want, 0 for any amount
if __name__ == "__main__":

    test_file = "./data/test.jsonl"
    output_file = f"./output/{args.run_name}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load the model and tokenizer:
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", trust_remote_code=True)

    # Load the test data:
    test_data = load_dataset("json", data_files=test_file, split="train", download_mode="force_redownload")
    labels = test_data["label"]
    texts = test_data["text"]
    
    # Evaluate:
    # each result struct has {idx, gt labels, output softmax}
    res = {"results": []}
    set_random_seed(17)
    for i in tqdm(range(len(texts))):
        gt_labels = labels[i]
        input = texts[i]
        tokenized_input = tokenizer(input, truncation=True, return_tensors="pt")
        with torch.no_grad():
            output = model(**tokenized_input)
        pred_probs = torch.softmax(output.logits, dim=1).tolist()
        res["results"].append({"index": i, "gt": gt_labels, "softmax": pred_probs})
        

    with open(output_file, 'w') as file:
        json.dump(res, file, indent=4)
