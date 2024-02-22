import ast
import json
import os
import random
from datasets import load_dataset

SEED = 17

# Converts string representation of a list into an actual list
def convert_str_list(input):
    res = ast.literal_eval(input)
    return res


# Only uses the question for gpt
def convert_dataset(dataset, use_multi_label=False):
    all_data = []
    for record in dataset:
        data = {}
        data["labels"] = convert_str_list(record["skill_types"])
        data["text"] = record["question"]
        if len(data["labels"]) == 0:
            continue
        all_data.append(data)
    return all_data


# Saves testing data for finetuned model inference
def save_dataset_test(all_data, output_dir):
    random.seed(SEED)
    random.shuffle(all_data)

    test_file = os.path.join(output_dir, "test.jsonl")
    os.makedirs(os.path.dirname(test_file), exist_ok=True)

    with open(test_file, 'w') as file:
        for data in all_data:
            json_line = json.dumps(data)
            file.write(json_line + '\n')


# Creates the test set json file for GPT
if __name__ == "__main__":

    output_dir = "./data/"
    skills = [
        "Amortized analysis",
        "Bit manipulation",
        "Complete search",
        "Data structures",
        "Dynamic programming",
        "Greedy algorithms",
        "Range queries",
        "Sorting"
    ]
    target_difficulties = ["EASY", "MEDIUM", "MEDIUM_HARD"]


    test_data = load_dataset('BAAI/TACO', split='test', skills=skills)
    test_data = test_data.filter(lambda example: example["difficulty"] in target_difficulties)
    data = convert_dataset(test_data, use_multi_label=True)
    save_dataset_test(data, output_dir)

    print("Datasets successfully converted!")