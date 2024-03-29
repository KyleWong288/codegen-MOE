import argparse
import json
import numpy as np
import os
import random
import re
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from tqdm import tqdm


DSC_MODEL_DICT = {
    "dsc-6.7b-instruct": "deepseek-ai/deepseek-coder-6.7b-instruct",
    "dsc-6.7b-base": "deepseek-ai/deepseek-coder-6.7b-base",
}
# shell friendly skill names
SKILL_LIST = ["amortized", "bit_maniuplation", "complete_search", "data_structures", "dp", "greedy", "range_queries", "sorting"]
SKILL_MAP = {
    "amortized": ["Amortized analysis"],
    "bit_maniuplation": ["Bit manipulation"], 
    "complete_search": ["Complete search"], 
    "data_structures": ["Data structures"], 
    "dp": ["Dynamic programming"], 
    "greedy": ["Greedy algorithms"], 
    "range_queries": ["Range queries"], 
    "sorting": ["Sorting"]
}
SKILL_INSTRUCTION_MAP = {
    "amortized": " Your program should use techniques that reduce time complexity, such as two pointers or sliding window.",
    "bit_manipulation": " Your program should use bitwise operations.",
    "complete_search": " Your program should use complete search or backtracking.",
    "data_structures": " Your program should use data structures.", 
    "dp": " Your program should use dynamic programming.",
    "greedy": " Your program should use a greedy algorithm.",
    "range_queries": " Your program should involve range queries, using prefix sums or a segment tree.",
    "sorting": " Your program should use sorting.",
    None: "",
}
EOF_STRINGS = ["### Response:", "\n###", "\nQUESTION", "\n---", "\nANSWER", "<|endoftext|>", "<|EOT|>"]
SEED = 17
NUM_SAMPLES = 20

parser = argparse.ArgumentParser()
parser.add_argument("--run_name", default='local-test', type=str)
parser.add_argument("--checkpoint", default='checkpoint-0', type=str)
parser.add_argument("--model_name", type=str, help="the model name", default="dsc-6.7b-base")
parser.add_argument("--do_sample", type=bool, default=True, help="Enable sampling")
parser.add_argument("--temperature", type=float, default=0.6, help="Sampling softmax temperature")
parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate")
parser.add_argument("--max_new_tokens", type=int, default=800, help="The maximum number of tokens to generate")
parser.add_argument("--checkpoint_path", type=str, help="the checkpoint path", default="./finetuned_models/run_name")
parser.add_argument("--load_in_8bit", action='store_true', help="load the model in 8 bits precision", default=True)
parser.add_argument("--trust_remote_code", type=bool, default=True, help="Enable `trust_remote_code`")
parser.add_argument("--skill", type=str, default="all", help="the skill to test on", choices=SKILL_LIST + ["all"])
parser.add_argument("--use_base_model", type=bool, default=False, help="Uses the base model out of box from hf")
args = parser.parse_args()


def load_pretrained_dsc_model(args):
    model = AutoModelForCausalLM.from_pretrained(
        DSC_MODEL_DICT[args.model_name],
        device_map = "auto",
        trust_remote_code = args.trust_remote_code,
        torch_dtype = torch.bfloat16
    )
    print(f"Base Model {args.model_name} loaded")
    return model


def load_finetuned_dsc_model(args):
    base_model = load_pretrained_dsc_model(args)
    ft_model = PeftModel.from_pretrained(base_model, args.checkpoint_path)
    print(f"Finetuned Model {args.checkpoint_path} loaded")
    return ft_model


def clean_no_format(text):
    pattern = '|'.join(re.escape(s) for s in EOF_STRINGS)
    match = re.search(pattern, text)
    if match:
        return text[:match.start()]
    else:
        return text


def clean_dsc_format(text):
    # Extracts the code block in ``` ``` and erases the "python\n"
    pattern = r"```(.*?)```"
    prefix = "python\n"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        res = match.group(1)
        if res.startswith(prefix):
            res = res[len(prefix):]
        return res
    else:
        return text


def clean_base_model2(text):
    p1 = "\n```python\n"
    p2 = "```python\n"
    prefixes = [p1, p2]
    for pref in prefixes:
        if text.startswith(pref):
            text = text[len(pref):]
    return text


def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


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


# Creates the prompt for a fine-tuned model
# Using "### Instruction" and "### Response" for DeepSeek Coder
def create_prompt(question):
    dsc_header = "You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer."
    instruction = "Write a Python program that solves the following question."
    if "all" not in args.run_name:
        instruction += SKILL_INSTRUCTION_MAP[args.skill]
    question = clean_question(question)
    res = "{} \n\n### Instruction: {} \nQuestion: {} \n\n### Response:\n".format(dsc_header, instruction, question)
    return res


# Creates the prompt for the instruction-tuned dsc model
# Using "### Instruction" and "### Response" for DeepSeek Coder
def create_prompt_instruct(question):
    dsc_header = "You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer."
    instruction = "Write a Python program that solves the following question. Only output your code, and do not provide an explanation."
    # question = clean_question(question)
    res = "{} \n\n### Instruction: {} \nQuestion: {} \n\n### Response:\n".format(dsc_header, instruction, question)
    return res


# Creates the prompt for the base dsc model
def create_prompt_base(question):
    instruction = "Write a Python program that solves the following question."
    # question = clean_question(question)
    res = "### Instruction: {} \nQuestion: {} \n\n### Response:\n".format(instruction, question)
    return res



# Gets NUM_SAMPLES generations, handles input encoding and output decoding
def predict(model, tokenizer, gen_config, prompt):
    
    res = []

    input = tokenizer(prompt, truncation=False, return_tensors="pt").to(model.device)
    input_ids = input["input_ids"]
    raw_text_len = len(input_ids[0])

    for i in tqdm(range(NUM_SAMPLES)):
        set_random_seed(i)
        
        with torch.no_grad():
            output = model.generate(
                **input,
                generation_config=gen_config,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=32021
            )
            output = tokenizer.decode(output[0][raw_text_len:], skip_special_tokens=True)
            clean_code = clean_dsc_format(output)
            clean_code = clean_base_model2(clean_code)
            res.append(clean_code)
            if i == 0:
                print(clean_code)
            
    
    return res


if __name__ == "__main__":

    # Load model and set up gen config
    generation_config = GenerationConfig(
        do_sample = args.do_sample,
        temperature = args.temperature,
        top_p = args.top_p,
        num_return_sequences = args.num_return_sequences, 
        max_new_tokens = args.max_new_tokens
    )
    model = load_pretrained_dsc_model(args) if args.use_base_model else load_finetuned_dsc_model(args)
    tokenizer = AutoTokenizer.from_pretrained(DSC_MODEL_DICT[args.model_name])
    tokenizer.pad_token = tokenizer.eos_token

    # Load test data and set up evaluation parameters
    if args.skill == "all":
        skills = [s for sublist in SKILL_MAP.values() for s in sublist]
    else:
        skills = SKILL_MAP[args.skill]
    print(skills)
    target_difficulties = ["EASY"]
    test_data = load_dataset('BAAI/TACO', split='test', skills=skills)
    test_data = test_data.filter(lambda example: example["difficulty"] in target_difficulties)
    test_data = test_data.filter(lambda example: len(example["starter_code"]) == 0)
    test_data = test_data.select(range(50))
    print("Skills tested:", skills)
    print("Difficulties test:", target_difficulties)
    
    # Configure output file
    if args.use_base_model:
        output_file = f"./output/{args.model_name}/{args.skill}/EASY/base.json"
    else:
        output_file = f"./output/{args.model_name}/{args.skill}/EASY/{args.run_name}-{args.checkpoint}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    output = []
    print("Writing to output file:", output_file)

    # Generate output
    for idx, sample in enumerate(test_data):
        print(f"ON {idx} OF {len(test_data)}")
        prompt = create_prompt_base(sample["question"])
        print(prompt)
        results = {"task_id": idx, "prompt": prompt}
        generations = predict(model, tokenizer, generation_config, prompt)
        results["output"] = generations
        output.append(results)

        with open(output_file, 'w') as file:
            json.dump(output, file, indent=4)

    print("DONE GENERATING!")
    