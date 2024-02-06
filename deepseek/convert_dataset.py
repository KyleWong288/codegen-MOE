import ast
import json
import os
import random
from datasets import load_dataset

SEED = 17
EOT_TOKEN = "<|EOT|>"

# Converts string representation of a list into an actual list
def convert_str_list(input):
    res = ast.literal_eval(input)
    return res


# Creates the finalized prompt
# Using "### Instruction" and "### Response" for DeepSeek Coder
# TODO: experiment with removing the examples in the prompt
def create_prompt(question, answer, skill=None):
    dsc_statement = "You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer."
    instruction = "Write a Python program that solves the following question."
    if skill == "Sorting":
        instruction += " Your program should use sorting."
    elif skill == "Greedy algorithms":
        instruction += " Your program should use a greedy algorithm."
    elif skill == "Data structures":
        instruction += " Your program should use data structures."
    res = "{} \n\n### Instruction: {} \nQuestion: {} \n\n### Response:\n{} \n{}".format(dsc_statement, instruction, question, answer, EOT_TOKEN)
    return res


# Gets the min length solution, helps reduce max_seq_len
def min_len_answer(answers):
    if not answers:
        return None
    res = answers[0]
    for ans in answers:
        if (len(ans) < len(res)):
            res = ans
    return res


# DS Coder only works with instruction and output
# Groups instruction and question into instruction
# TODO: EDA for more data
def convert_dataset(dataset, target_size, skill=None):
    all_data = []

    for record in dataset:
        data = {}
        data["question"] = record["question"]
        data["answer"] = convert_str_list(record["solutions"])
        data["answer"] = min_len_answer(data["answer"])
        if not data["answer"]:
            continue
        data["skill_types"] = convert_str_list(record["skill_types"])
        if skill and skill not in data["skill_types"]:
            continue
        data["tags"] = convert_str_list(record["tags"])
        data["text"] = create_prompt(data["question"], data["answer"], skill)
        all_data.append(data)

    if len(all_data) > target_size:
        random.seed(SEED)
        random.shuffle(all_data)
        all_data = all_data[:target_size]
    
    return all_data


def save_dataset(data, output_dir, split_ratio=0.9):
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


# Splits dataset by tag/skill_type
if __name__ == "__main__":

    data_dir = "../dsc_limit_data"
    train_dev_ratio = 0.9
    target_size = 2000

    skills = ["Sorting", "Greedy algorithms", "Data structures"]
    train_data = load_dataset('BAAI/TACO', split='train', skills=skills)
    
    splits = {"Sorting": "sorting", 
              "Greedy algorithms": "greedy", 
              "Data structures": "data_structures",
              None: "all"}
    
    for skill, value in splits.items():
        output_dir = os.path.join(data_dir, value)
        data = convert_dataset(train_data, target_size, skill)
        save_dataset(data, output_dir, train_dev_ratio)
        print("Wrote to", value)
        print("SAMPLE:")
        print(data[0]["text"])

    print("Datasets successfully converted!")