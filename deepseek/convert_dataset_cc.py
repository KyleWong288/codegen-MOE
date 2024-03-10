import ast
import json
import os
import random
import re
from datasets import load_dataset
from collections import defaultdict

SEED = 17
MAX_QUESTION_LEN = 2000

def get_python_solution(solutions, max_length=2000):
    python_solutions = []
    for sol in solutions:
        if len(sol) > max_length:
            continue
        if "input()" in sol and "#include" not in sol:
            python_solutions.append(sol)
    if not python_solutions:
        return None
    idx = random.randint(0, len(python_solutions)-1)
    return python_solutions[idx]


def convert_dataset(data, output_file):

    used = 0

    with open(output_file, "w") as file:
        for sample in data:
            json_obj = {
                "name": sample["name"],
                "description": sample["description"],
                "solution": get_python_solution(sample["solutions"]["solution"])
            }
            if len(json_obj["description"]) < MAX_QUESTION_LEN and json_obj["solution"]:
                json_line = json.dumps(json_obj, indent=4)
                file.write(json_line + "\n")
                used += 1

    print(f"USED: {used}/{len(data)}")


if __name__ == "__main__":

    random.seed(SEED)
    dataset = load_dataset("deepmind/code_contests")
    
    # Write to train
    train_data = dataset["train"]
    train_file = "../dsc_data_code_contests/train.jsonl"
    convert_dataset(train_data, train_file)

    # Write to validation
    dev_data = dataset["valid"]
    dev_file = "../dsc_data_code_contests/dev.jsonl"
    convert_dataset(dev_data, dev_file)

    # Write to test
    test_data = dataset["test"]
    test_file = "../dsc_data_code_contests/test.jsonl"
    convert_dataset(test_data, test_file)