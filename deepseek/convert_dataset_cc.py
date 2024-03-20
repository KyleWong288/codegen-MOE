import ast
import json
import os
import random
import re
from datasets import load_dataset
from collections import defaultdict

SEED = 17
MAX_QUESTION_LEN = 2000
EOT_TOKEN = "<|EOT|>"

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


# Creates the finalized prompt for dsc-instruct, using "### Instruction" and "### Response" for DeepSeek Coder
def create_prompt_dsc_instruct(question, answer):
    dsc_header = "You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer."
    instruction = "Write a Python program that solves the following question."
    res = "{} \n\n### Instruction: {}\nQuestion: {}\n\n### Response:\n{}".format(dsc_header, instruction, question, answer)
    return res


# Creates the finalized prompt dsc-base and llama
def create_prompt(question, answer):
    instruction = "Write a Python program that solves the following question."
    res = "### Instruction: {}\nQuestion: {}\n\n### Response:\n{}".format(instruction, question, answer)
    return res


# Puts the code in the dsc format
# Format is ```python <code> ```, and use 4 spaces instead of \t
def reformat_code(code):
    spacing = "    "
    code = code.replace("\t", spacing)
    res = f"```python\n{code}\n```\n"
    res += EOT_TOKEN
    return res


def convert_dataset(data, output_file):

    used = 0

    with open(output_file, "w") as file:
        for sample in data:
            json_obj = {
                "name": sample["name"],
                "text": None,
                "question": sample["description"],
                "answer": get_python_solution(sample["solutions"]["solution"])
            }
            if len(json_obj["question"]) > MAX_QUESTION_LEN or not json_obj["answer"]:
                continue
            json_obj["answer"] = reformat_code(json_obj["answer"])
            json_obj["text"] = create_prompt(json_obj["question"], json_obj["answer"])
            file.write(json.dumps(json_obj) + "\n")
            used += 1
            if used == 1:
                print(json_obj["text"])

    print(f"USED: {used}/{len(data)}")


if __name__ == "__main__":

    random.seed(SEED)
    dataset = load_dataset("deepmind/code_contests")
    
    # Write to train
    train_data = dataset["train"]
    train_file = "../dsc_data_base_code_contests/train.jsonl"
    os.makedirs(os.path.dirname(train_file), exist_ok=True)
    convert_dataset(train_data, train_file)

    # Write to validation
    dev_data = dataset["valid"]
    dev_file = "../dsc_data_base_code_contests/dev.jsonl"
    os.makedirs(os.path.dirname(dev_file), exist_ok=True)
    convert_dataset(dev_data, dev_file)

    # Write to test
    test_data = dataset["test"]
    test_file = "../dsc_data_base_code_contests/test.jsonl"
    os.makedirs(os.path.dirname(test_file), exist_ok=True)
    convert_dataset(test_data, test_file)