import numpy as np
import torch
import argparse
import os
import random
import re
import json
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from tqdm import tqdm


DSC_MODEL_DICT = {
    "dsc-6.7b-instruct": "deepseek-ai/deepseek-coder-6.7b-instruct",
}
SKILL_DICT = {
    "sorting": ["Sorting"],
    "greedy": ["Greedy"],
    "data_structures": ["Data structures"],
    "all": ["Sorting", "Greedy algorithms", "Data structures"]
}
EOF_STRINGS = ["### Response:", "\n###", "\nQUESTION", "\n---", "\nANSWER", "<|endoftext|>", "<|EOT|>"]
SEED = 17

parser = argparse.ArgumentParser()
parser.add_argument("--run_name", default='local-test', type=str)
parser.add_argument("--model_name", type=str, help="the model name", default="dsc-6.7b-instruct")
parser.add_argument("--do_sample", type=bool, default=True, help="Enable sampling")
parser.add_argument("--temperature", type=float, default=0.6, help="Sampling softmax temperature")
parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
parser.add_argument("--num_return_sequences", type=int, default=20, help="The number of samples to generate")
parser.add_argument("--max_new_tokens", type=int, default=800, help="The maximum number of tokens to generate")
parser.add_argument("--checkpoint_path", type=str, help="the checkpoint path", default="./finetuned_models/run_name")
parser.add_argument("--load_in_8bit", action='store_true', help="load the model in 8 bits precision", default=True)
parser.add_argument("--trust_remote_code", type=bool, default=True, help="Enable `trust_remote_code`")
parser.add_argument("--skill", type=str, default="all", help="the skill to test on")
args = parser.parse_args()


def load_pretrained_dsc_model(args):
    model = AutoModelForCausalLM.from_pretrained(
        DSC_MODEL_DICT[args.model_name],
        device_map = "auto",
        trust_remote_code = args.trust_remote_code,
        torch_dtype = torch.bfloat16
    )
    return model


def load_finetuned_dsc_model(args):
    base_model = AutoModelForCausalLM.from_pretrained(
        DSC_MODEL_DICT[args.model_name],
        device_map = "auto",
        trust_remote_code = args.trust_remote_code,
        torch_dtype = torch.bfloat16
    )
    print(f"Base Model {args.model_name} loaded")
    ft_model = PeftModel.from_pretrained(base_model, args.checkpoint_path)
    print(f"PEFT Model {args.checkpoint_path} loaded")
    return ft_model


def truncate_after_eof_strings(text):
    pattern = '|'.join(re.escape(s) for s in EOF_STRINGS)
    match = re.search(pattern, text)
    if match:
        return text[:match.start()]
    else:
        return text


def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


# Creates the finalized prompt
# Using "### Instruction" and "### Response" for DeepSeek Coder
def create_prompt(question, skill=None):
    dsc_header = "You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer."
    instruction = "Write a Python program that solves the following question."
    if skill == "Sorting":
        instruction += " Your program should use sorting."
    elif skill == "Greedy algorithms":
        instruction += " Your program should use a greedy algorithm."
    elif skill == "Data structures":
        instruction += " Your program should use data structures."
    res = "{} \n\n### Instruction: {} \nQuestion: {} \n\n### Response:\n".format(dsc_header, instruction, question)
    return res


# gets generations, handles input encoding and output decoding
def predict(model, tokenizer, gen_config, prompt):
    set_random_seed(SEED)
    res = []

    input = tokenizer(prompt, truncation=False, return_tensors="pt").to(model.device)
    input_ids = input["input_ids"]
    raw_text_len = len(input_ids[0])
    with torch.no_grad():
        output = model.generate(
            **input,
            generation_config=gen_config,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=32021
        )
        for out in output:
            res.append(tokenizer.decode(out[raw_text_len:], skip_special_tokens=True))
    
    return res


# naively truncates eot tokens
def clean_generations(generations):
    res = []
    for gen in generations:
        clean_code = truncate_after_eof_strings(gen)
        res.append(clean_code)
    return res


if __name__ == "__main__":

    # Load model and set up model gen config
    generation_config = GenerationConfig(
        do_sample = args.do_sample,
        temperature = args.temperature,
        top_p = args.top_p,
        num_return_sequences = args.num_return_sequences, 
        max_new_tokens = args.max_new_tokens
    )
    model = load_finetuned_dsc_model(args)
    tokenizer = AutoTokenizer.from_pretrained(DSC_MODEL_DICT[args.model_name])
    tokenizer.pad_token = tokenizer.eos_token

    # Load test data and set up evaluation parameters
    skills = SKILL_DICT[args.skill]
    test_data = load_dataset('BAAI/TACO', split='test', skills=skills)
    print("Skills tested:", skills)
    
    output_file = f"./output/{args.model_name}/{args.run_name}_late.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    output = []

    target_difficulties = ["EASY", "MEDIUM", "MEDIUM_HARD"]

    cnt = 0

    for idx, sample in enumerate(test_data):
        if sample["difficulty"] not in target_difficulties:
            continue
        print("ON:", idx)
        prompt = create_prompt(sample["question"])
        print(prompt)
        results = {"task_id": idx, "prompt": prompt}

        generations = predict(model, tokenizer, generation_config, prompt)
        generations = clean_generations(generations)
        results["output"] = generations
        output.append(results)

        cnt += 1
        if cnt >= 2:
            break

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=4)
    