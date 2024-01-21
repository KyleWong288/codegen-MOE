import numpy as np
import torch
import argparse
import random
import re
import json
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from tqdm import tqdm


LLAMA_MODEL_DICT = {
    "code-llama": "codellama/CodeLlama-7b-hf"
}
EOF_STRINGS = ["\nQUESTION", "\n---", "\nANSWER", "<|endoftext|>"]


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, help="the model name", default="code-llama")
parser.add_argument("--do_sample", type=bool, default=True, help="Enable sampling")
parser.add_argument("--temperature", type=float, default=0.2, help="Sampling softmax temperature")
parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate")
parser.add_argument("--max_new_tokens", type=int, default=512, help="The maximum number of tokens to generate")
parser.add_argument("--base_model_name", type=str, help="the base model name", default="meta-llama/Llama-2-7b-chat-hf")
parser.add_argument("--checkpoint_path", type=str, help="the checkpoint path", default="./../checkpoints/Llama-2-7b-chat-hf")
parser.add_argument("--load_in_8bit", action='store_true', help="load the model in 8 bits precision", default=True)
parser.add_argument("--trust_remote_code", type=bool, default=True, help="Enable `trust_remote_code`")
args = parser.parse_args()


def load_llama_model(args):
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL_DICT[args.model_name],
        quantization_config = BitsAndBytesConfig(load_in_8bit = True),
        device_map = "auto",
        trust_remote_code = args.trust_remote_code,
        torch_dtype = torch.bfloat16
    )
    return model


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


if __name__ == "__main__":

    # Load model and set up generation config
    generation_config = GenerationConfig(
        do_sample = args.do_sample,
        temperature = args.temperature,
        top_p = args.top_p,
        num_return_sequences = args.num_return_sequences, 
        max_new_tokens = args.max_new_tokens
    )
    model = load_llama_model(args)
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_DICT[args.model_name])
    tokenizer.pad_token = tokenizer.eos_token

    # Load test data and set up evaluation parameters
    
    difficulties = ["EASY"]
    test_data = load_dataset('BAAI/TACO', split='test', difficulties=difficulties)
    
    output_file = 'generation.json'
    output = []

    test_data = test_data.select(list(range(50)))
    print(len(test_data))

    n_samples = 100

    for idx, sample in tqdm(enumerate(test_data)):
        prompt = "\nQUESTION:\n"
        prompt += sample["question"]
        starter_code = None if len(sample["starter_code"]) == 0 else sample["starter_code"]
        try:
            input_outpout = json.loads(sample["input_output"])
            fn_name = (
                None if not input_outpout.get("fn_name") else input_outpout["fn_name"]
            )
        except ValueError:
            fn_name = None
        if starter_code:
            prompt += starter_code
        if (not fn_name) and (not starter_code):
            call_format = "\nUse Standard Input format"
            prompt += call_format
        else:
            call_format = "\nUse Call-Based format"
            prompt += call_format
        prompt += "\nANSWER:\n"
        results = {"task_id": idx, "prompt": prompt}
        generations = []
        for i in range(n_samples):
            seed = i
            generation = predict(model, tokenizer, generation_config, prompt, seed)
            clean_code = truncate_after_eof_strings(generation)
            generations.append(clean_code)
        results["output"] = generations
        output.append(results)

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=4)
    

