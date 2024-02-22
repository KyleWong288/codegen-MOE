import ast
import json
import os
import random
from datasets import load_dataset

SEED = 17
LABEL2ID = {
    "Amortized analysis": 0,
    "Bit manipulation": 1,
    "Complete search": 2,
    "Data structures": 3,
    "Dynamic programming": 4,
    "Greedy algorithms": 5,
    "Range queries": 6,
    "Sorting": 7,
}


# Converts string representation of a list into an actual list
def convert_str_list(input):
    res = ast.literal_eval(input)
    return res


# Converts string skill list to a binary vector of labels
def make_labels(skill_list):
    res = [0] * 8
    for skill in skill_list:
        res[LABEL2ID[skill]] = 1
    return res


# Creates the finalized prompt
# Roberta only gets the question and default instruction
# TODO: experiment with removing the examples in the prompt
def create_prompt(question):
    dsc_header = "You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer."
    instruction = "Write a Python program that solves the following question."
    res = "{} \n\n### Instruction: {} \nQuestion: {} \n\n### Response:\n".format(dsc_header, instruction, question)
    return res


# Creates dataset of {prompt text, binary list of labels}
def convert_dataset(dataset, use_multi_label=False):
    all_data = []
    for record in dataset:
        data = {}
        question = record["question"]
        skill_list = convert_str_list(record["skill_types"])
        if len(skill_list) == 0:
            continue
        data["labels"] = make_labels(skill_list)
        data["text"] = create_prompt(question)
        all_data.append(data)
    return all_data


# splits dataset into train and eval and saves to output dir
def save_dataset_split(data, output_dir, split_ratio=0.9):
    random.seed(SEED)
    random.shuffle(data)
    split = int(len(data) * split_ratio)

    train_data = data[:split]
    dev_data = data[split:]
    train_file = os.path.join(output_dir, "train.jsonl")
    dev_file = os.path.join(output_dir, "dev.jsonl")
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    os.makedirs(os.path.dirname(train_file), exist_ok=True)
    os.makedirs(os.path.dirname(dev_file), exist_ok=True)

    with open(train_file, 'w') as file:
        for data in train_data:
            json_line = json.dumps(data)
            file.write(json_line + '\n')

    with open(dev_file, 'w') as file:
        for data in dev_data:
            json_line = json.dumps(data)
            file.write(json_line + '\n')


# saves testing data for finetuned model inference
def save_dataset_test(all_data, output_dir):
    random.seed(SEED)
    random.shuffle(all_data)

    test_file = os.path.join(output_dir, "test.jsonl")
    os.makedirs(os.path.dirname(test_file), exist_ok=True)

    with open(test_file, 'w') as file:
        for data in all_data:
            json_line = json.dumps(data)
            file.write(json_line + '\n')


# Splits dataset by tag/skill_type
if __name__ == "__main__":

    output_dir = "./data/"
    train_dev_ratio = 0.9

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

    train_data = load_dataset('BAAI/TACO', split='train', skills=skills)
    data = convert_dataset(train_data)
    save_dataset_split(data, output_dir, train_dev_ratio)

    test_data = load_dataset('BAAI/TACO', split='test', skills=skills)
    data = convert_dataset(test_data, use_multi_label=True)
    save_dataset_test(data, output_dir)

    print("Datasets successfully converted!")