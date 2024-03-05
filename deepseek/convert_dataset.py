import ast
import json
import os
import random
import re
from datasets import load_dataset
from collections import defaultdict

SEED = 17
EOT_TOKEN = "<|EOT|>"
SKILL_INSTRUCTION_MAP = {
    "Amortized analysis": " Your program should use techniques that reduce time complexity, such as two pointers or sliding window.",
    "Bit manipulation": " Your program should use bitwise operations.",
    "Complete search": " Your program should use complete search or backtracking.",
    "Data structures": " Your program should use data structures.", 
    "Dynamic programming": " Your program should use dynamic programming.",
    "Greedy algorithms": " Your program should use a greedy algorithm.",
    "Range queries": " Your program should involve range queries, using prefix sums or a segment tree.",
    "Sorting": " Your program should use sorting.",
    None: "",
}

# Converts string representation of a list into an actual list
def convert_str_list(input):
    res = ast.literal_eval(input)
    return res


# Creates the finalized prompt, using "### Instruction" and "### Response" for DeepSeek Coder
def create_prompt(question, answer, skill=None):
    dsc_header = "You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer."
    instruction = "Write a Python program that solves the following question."
    instruction += SKILL_INSTRUCTION_MAP[skill]
    res = "{} \n\n### Instruction: {} \nQuestion: {} \n\n### Response:\n{}".format(dsc_header, instruction, question, answer)
    return res


# For the non-instruction tuned model
def create_prompt_base(question, answer, skill=None):
    instruction = "Write a Python program that solves the following question."
    instruction += SKILL_INSTRUCTION_MAP[skill]
    res = "### Instruction: {} \nQuestion: {} \n\n### Response:\n{}".format(instruction, question, answer)
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


# Picks a random answer under a certain length, always choosing the min len answer may create bias to shorter/simpler code
def pick_random_answer(answers, max_chars=2000):
    short_answers = [ans for ans in answers if len(ans) < max_chars]
    if not short_answers:
        return None
    idx = random.randint(0, len(short_answers)-1)
    return short_answers[idx]


# Erases the test cases and explanations from the question
def clean_question(question):
    res = question

    # String match clean
    targets = ["\nExamples", "\nExample 1:", "\n## Examples", "\nExample:", "\nExample\n", "\nExample \n", "\nExample :", "\nSample Input", "\nSAMPLE INPUT"]
    for target in targets:
        index = res.find(target)
        if index != -1:
            res = res[:index]

    # Regex clean
    pat0 = r'\n-+\s*Example'
    pat1 = r'\n-+\s*Examples'
    pat2 = r'\n-+\s*Sample Input'
    pat3 = r'\n-+\s*Example Input'
    regex_patterns = [pat0, pat1, pat2, pat3]
    for pattern in regex_patterns:
        res = re.split(pattern, res, maxsplit=1)[0]

    return res


# Puts the code in the dsc format
# Format is ```python <code> ```, and use 4 spaces instead of \t
def reformat_code(code):
    code += f"\n{EOT_TOKEN}"
    spacing = "    "
    code = code.replace("\t", spacing)
    res = f"```python\n{code}\n```"
    return res


# DS Coder only works with instruction and output
# Groups instruction and question into instruction
# TODO: EDA for more data??
def convert_dataset(dataset, target_size, skill=None):
    all_data = []

    for record in dataset:
        data = {"text": None}
        data["question"] = clean_question(record["question"])
        data["answer"] = convert_str_list(record["solutions"])
        data["answer"] = pick_random_answer(data["answer"])
        if not data["answer"]:
            continue
        data["answer"] = reformat_code(data["answer"])
        data["skill_types"] = convert_str_list(record["skill_types"])
        data["tags"] = convert_str_list(record["tags"])
        data["text"] = create_prompt_base(data["question"], data["answer"], skill)
        all_data.append(data)

    if len(all_data) > target_size:
        random.seed(SEED)
        random.shuffle(all_data)
        all_data = all_data[:target_size]
    
    return all_data


def save_split_dataset(data, output_dir, split_ratio=0.9):
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

    data_dir = "../dsc_data"
    train_dev_ratio = 0.9
    target_size = 4000

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

    # splits = {
    #     "Amortized analysis": "amortized",
    #     "Bit manipulation": "bit_manipulation",
    #     "Complete search": "complete_search",
    #     "Data structures": "data_structures",
    #     "Dynamic programming": "dp",
    #     "Greedy algorithms": "greedy",
    #     "Range queries": "range_queries",
    #     "Sorting": "sorting",
    #     None: "all"
    # }

    splits = {
        None: "all"
    }
    
    for skill, value in splits.items():
        output_dir = os.path.join(data_dir, value)
        filtered_data = train_data.filter(lambda example: skill in example["skill_types"]) if skill else train_data
        data = convert_dataset(filtered_data, target_size, skill)
        save_split_dataset(data, output_dir, train_dev_ratio)
        print("Wrote to", value)
        print("SAMPLE:")
        print(data[0]["text"])

    print("Datasets successfully converted!")