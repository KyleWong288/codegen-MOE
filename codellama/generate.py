import numpy as np
import torch
import argparse
import os
import random
import re
import json
from datasets import load_dataset, load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from tqdm import tqdm


LLAMA_MODEL_DICT = {
    "codellama": "codellama/CodeLlama-7b-hf",
    "codellama_python": "codellama/CodeLlama-7b-Python-hf"
}
EOF_STRINGS = ["\nQUESTION", "\n---", "\nANSWER", "<|endoftext|>"]


parser = argparse.ArgumentParser()
parser.add_argument("--run_name", default='local-test', type=str)
parser.add_argument("--model_name", type=str, help="the model name", default="codellama_python")
parser.add_argument("--do_sample", type=bool, default=True, help="Enable sampling")
parser.add_argument("--temperature", type=float, default=0.6, help="Sampling softmax temperature")
parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate")
parser.add_argument("--max_new_tokens", type=int, default=800, help="The maximum number of tokens to generate")
parser.add_argument("--base_model_name", type=str, help="the base model name", default="codellama/CodeLlama-7b-Python-hf")
parser.add_argument("--checkpoint_path", type=str, help="the checkpoint path", default="./../checkpoints/Llama-2-7b-chat-hf")
parser.add_argument("--load_in_8bit", action='store_true', help="load the model in 8 bits precision", default=True)
parser.add_argument("--trust_remote_code", type=bool, default=True, help="Enable `trust_remote_code`")
args = parser.parse_args()


def load_pretrained_llama_model(args):
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL_DICT[args.model_name],
        quantization_config = BitsAndBytesConfig(load_in_8bit = True),
        device_map = "auto",
        trust_remote_code = args.trust_remote_code,
        torch_dtype = torch.bfloat16
    )
    return model


def load_finetuned_llama_model(args):
    base_model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL_DICT[args.model_name],
        quantization_config = BitsAndBytesConfig(load_in_8bit = True),
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


def decode(tokenizer, raw_text_len, output):
    sents = []
    for tokens in output:
        tokens = tokens.cpu().numpy().tolist()
        sent = tokenizer.decode(tokens[raw_text_len:], skip_special_tokens=True)
        sents.append(sent)
    return sents


def predict(model, tokenizer, gen_config, prompt, seed):
    set_random_seed(seed)
    input = tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda:0")
    input_ids = input["input_ids"]
    raw_text_len = len(input_ids[0])
    with torch.no_grad():
        output = model.generate(**input, generation_config=gen_config, pad_token_id=tokenizer.eos_token_id)
        output = decode(tokenizer, raw_text_len, output)
    return output[0]


def create_prompt(question, skill=None):
    instruction = "Write a Python program that solves the following question."
    res = "### Instruction: {} \n\n ### Question: {} \n\n ### Answer:\n".format(instruction, question)
    return res


if __name__ == "__main__":

    # Load model and set up generation config
    generation_config = GenerationConfig(
        do_sample = args.do_sample,
        temperature = args.temperature,
        top_p = args.top_p,
        num_return_sequences = args.num_return_sequences, 
        max_new_tokens = args.max_new_tokens
    )
    model = load_finetuned_llama_model(args)
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_DICT[args.model_name])
    tokenizer.pad_token = tokenizer.eos_token

    # Load test data and set up evaluation parameters
    skills = ["Sorting", "Greedy algorithms", "Data structures"]
    test_data = load_dataset('BAAI/TACO', split='test', skills=skills)
    print(len(test_data))
    
    output_file = f"./output/{args.model_name}/{args.run_name}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    output = []

    n_samples = 20

    # mini test
    # question = "Complete the method which accepts an array of integers, and returns one of the following:\n\n* `\"yes, ascending\"` - if the numbers in the array are sorted in an ascending order\n* `\"yes, descending\"` - if the numbers in the array are sorted in a descending order\n* `\"no\"` - otherwise\n\n\nYou can assume the array will always be valid, and there will always be one correct answer"
    # prompt = create_prompt(question)
    # answer = predict(model, tokenizer, generation_config, prompt, 42)
    # answer = truncate_after_eof_strings(answer)
    # print("Answer:\n", answer)
    
    for idx, sample in enumerate(test_data):
        if sample["difficulty"] != "EASY":
            continue
        print("ON:", idx)
        prompt = create_prompt(sample["question"])
        results = {"task_id": idx, "prompt": prompt}
        generations = []
        for i in tqdm(range(n_samples)):
            seed = i
            generation = predict(model, tokenizer, generation_config, prompt, seed)
            clean_code = truncate_after_eof_strings(generation)
            generations.append(clean_code)
        results["output"] = generations
        output.append(results)

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=4)
    