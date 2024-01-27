import ast
import json
import os
import random
from datasets import load_dataset

# Converts string representation of a list into an actual list
def convert_str_list(input):
    res = ast.literal_eval(input)
    return res


# Creates the finalized prompt
# Using "Question/Answer" instead of "Input/Output" because input/output is also used for test cases 
# TODO: experiment with removing the examples in the prompt
# TODO: experiment without the '###'
def create_prompt(question, answer, skill=None):
    instruction = "Write a Python program that solves the following question."
    if skill == "Sorting":
        instruction += " Your program should use sorting."
    elif skill == "Greedy algorithms":
        instruction += " Your program should use a greedy algorithm."
    elif skill == "Data structures":
        instruction += " Your program should use data structures."
    res = "### Instruction: {} \n\n ### Question: {} \n\n ### Answer:\n{}".format(instruction, question, answer)
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


# Keys: ['question', 'solutions', 'starter_code', 'input_output', 'difficulty', 'raw_tags', 'name', 'source', 'tags', 'skill_types', 'url', 'Expected Auxiliary Space', 'time_limit', 'date', 'picture_num', 'memory_limit', 'Expected Time Complexity']
# Questions have multiple solutions, just use the first for now
# Creates a 90/10 train/dev split
# TODO: EDA potential for making a larger dataset
def dataset_to_jsonl(dataset, output_dir, split_ratio=0.9, skill=None):
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

    random.seed(17)
    random.shuffle(all_data)
    split = int(len(all_data) * split_ratio)

    train_data = all_data[:split]
    dev_data = all_data[split:]
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

    data_dir = "../poc_data"
    train_dev_ratio = 0.9

    skills = ["Sorting", "Greedy algorithms", "Data structures"]
    train_data = load_dataset('BAAI/TACO', split='train', skills=skills)
    
    splits = {"Sorting": "sorting", 
              "Greedy algorithms": "greedy", 
              "Data structures": "data_structures",
              None: "all"}
    
    for category, value in splits.items():
        output_dir = os.path.join(data_dir, value)
        dataset_to_jsonl(train_data, output_dir, train_dev_ratio, category)
        print("Wrote to", value)

    print("Datasets successfully converted!")