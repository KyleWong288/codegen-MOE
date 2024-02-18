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


if __name__ == "__main__":

    test_file = "./data/test.jsonl"
    output_file = f"./eval_results/{args.run_name}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load the model and tokenizer:
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", trust_remote_code=True)

    # Load the test data:
    test_data = load_dataset("json", data_files=test_file, split="train", download_mode="force_redownload")
    labels = test_data["label"]
    texts = test_data["text"]
    
    # Evaluate:
    res = {"acc": 0, "results": []}
    set_random_seed(17)
    correct = 0
    for i in tqdm(range(len(texts))):
        label_list = labels[i]
        input = texts[i]
        tokenized_input = tokenizer(input, truncation=True, return_tensors="pt")
        with torch.no_grad():
            output = model(**tokenized_input)
        pred_probs = torch.softmax(output.logits, dim=1)
        pred_label = pred_probs.argmax().item()
        if pred_label in label_list:
            correct += 1
        res["results"].append({"index": i, "pred": pred_label, "gt": label_list})

    acc = correct / len(texts)
    print("ACCURACY:", acc)
    res["acc"] = acc

    with open(output_file, 'w') as f:
        json.dump(res, f, indent=4)




    