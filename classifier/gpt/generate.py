import argparse
import json
import numpy as np
import os
import random
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--run_name", default='gpt_test_1', type=str)
args = parser.parse_args()
SKILL_LIST = [
        "Amortized analysis",
        "Bit manipulation",
        "Complete search",
        "Data structures",
        "Dynamic programming",
        "Greedy algorithms",
        "Range queries",
        "Sorting" 
    ]


def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


# Returns a list of any labels mentioned in the generated text
def extract_labels(text):
    res = []
    text = text.lower()
    for skill in SKILL_LIST:
        label = skill.lower()
        if label in text:
            res.append(skill)
    return res


if __name__ == "__main__":

    test_file = "./data/test.jsonl"
    output_file = f"./eval_results/{args.run_name}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load the test data:
    test_data = load_dataset("json", data_files=test_file, split="train", download_mode="force_redownload")
    labels = test_data["labels"]
    questions = test_data["text"]

    # Load client:
    load_dotenv()
    client = OpenAI()
    set_random_seed(17)
    
    # Each result is a struct of {ground truth label list, answer parsed from response, generated response}
    output = {"accuracy": 0, "results": []}
    premise1 = "You are given a programming question and a list of eight programming topics. Read the question, and choose one topic that you think is the most relevant for solving the question. Do not solve the question. Simply state your choice."
    premise2 = "You are given a programming question and a list of eight programming topics. Read the question, and choose up to 2 topics that you think are extremely relevant for solving the question. Do not solve the question. Simply state your choices."
    premise3 = "You are given a programming question and a list of eight programming topics. Read the question, and choose up to 3 topics that you think are extremely relevant for solving the question. Do not solve the question. Simply state your choices."
    skill_list_str = "[Amortized analysis, Bit manipulation, Complete search, Data structures, Dynamic programming, Greedy algorithms, Range queries, Sorting]"

    # Get the generations:
    for i in tqdm(range(len(questions))):
        question = questions[i]
        prompt = f"{premise1}\nList of topics: {skill_list_str}\nQuestion: {question}\n"
        print(prompt)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an intelligent expert at solving programming questions."},
                {"role": "user", "content": prompt} 
            ],
            max_tokens=40
        )

        generation = response.choices[0].message.content
        answers = extract_labels(generation)
        result = {"labels": labels[i], "answers": answers, "generation": generation}
        output["results"].append(result)

    # Evaluate accuracy:
    acc = 0
    for result in output["results"]:
        gt_labels = result["labels"]
        pred_labels = result["answers"]
        intersection = [label for label in pred_labels if label in gt_labels]
        acc += (len(intersection) > 0)
    acc /= len(output["results"])
    output["accuracy"] = acc

    # Save output file
    with open(output_file, 'w') as file:
        json.dump(output, file, indent=4)

    print("DONE EVALUATING!")

