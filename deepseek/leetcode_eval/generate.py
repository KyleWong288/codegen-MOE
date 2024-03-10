import argparse
import json
import numpy as np
import os
import random
import re
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from tqdm import tqdm

VERSION = "20240121-Jul"
DSC_MODEL_DICT = {
    "dsc-6.7b-instruct": "deepseek-ai/deepseek-coder-6.7b-instruct",
    "dsc-6.7b-base": "deepseek-ai/deepseek-coder-6.7b-base",
}
NUM_SAMPLES = 20


def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def load_pretrained_dsc_model(args):
    model = AutoModelForCausalLM.from_pretrained(
        DSC_MODEL_DICT[args.model_name],
        device_map = "auto",
        trust_remote_code = True,
        torch_dtype = torch.bfloat16
    )
    print(f"Base Model {args.model_name} loaded")
    return model


def load_finetuned_dsc_model(args):
    base_model = load_pretrained_dsc_model(args)
    ft_model = PeftModel.from_pretrained(base_model, args.checkpoint_path)
    print(f"Finetuned Model {args.checkpoint_path} loaded")
    return ft_model


def create_prompt(question):
    instruction = "Write a Python program that solves the following question."
    if args.use_base_model:
        instruction += " Only write the python program. Do not explain your code."
    res = "### Instruction: {} \nQuestion: {} \n\n### Response:\n".format(instruction, question)
    return res


# Gets one generation, handles input encoding and output decoding
def predict(model, tokenizer, gen_config, prompt, question_idx):
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
        output = tokenizer.decode(output[0][raw_text_len:], skip_special_tokens=True)
    
    return output


def main(args):
    
    # Load model and set up gen config
    model = load_pretrained_dsc_model(args) if args.use_base_model else load_finetuned_dsc_model(args)
    generation_config = GenerationConfig(
        do_sample = True,
        temperature = args.temperature,
        top_p = args.top_p,
        max_new_tokens = args.max_new_tokens
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(DSC_MODEL_DICT[args.model_name])
    tokenizer.pad_token = tokenizer.eos_token

    # Load test data
    test_data = [json.loads(line) for line in open(args.data_path).readlines()]
    print("USING TEST DATA:", args.data_path)

    # Configure output file
    if args.use_base_model:
        output_dir = f"./output/{args.model_name}/base"
    else:
        output_dir = f"./output/{args.model_name}/{args.run_name}-{args.checkpoint}"
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    print("OUTPUT DIRECTORY:", output_dir)

    # Generate output
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    for i in tqdm(range(NUM_SAMPLES)):
        set_random_seed(i)
        outputs = []
        output_file = f"{output_dir}/sample_{i}.jsonl"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        for idx, sample in enumerate(test_data):
            print(f"ON {idx} OF {len(test_data)}")
            prompt = create_prompt(sample["prompt_sft"])
            if i == 0:
                print(prompt)
            generation = predict(model, tokenizer, generation_config, prompt, idx)
            if i == 0:
                print(generation)
            output = sample
            output["output"] = generation
            outputs.append(output)

            with open(output_file, 'w') as file:
                json.dump(outputs, file, indent=4)

    print("DONE GENERATING!")
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default='local-test', type=str)
    parser.add_argument("--checkpoint", default='checkpoint-0', type=str)
    parser.add_argument('--model_name', type=str, default='dsc-6.7b-instruct')
    parser.add_argument('--data_path', type=str, default="./test_data/20240121-Jul_50.jsonl")
    parser.add_argument("--temperature", type=float, default=0.4, help="Sampling softmax temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="The maximum number of tokens to generate")
    parser.add_argument("--checkpoint_path", type=str, help="the checkpoint path", default="./finetuned_models/run_name")
    parser.add_argument("--load_in_8bit", action='store_true', help="load the model in 8 bits precision", default=True)
    parser.add_argument("--use_base_model", type=bool, default=False, help="Uses the base model out of box from hf")
    args = parser.parse_args()

    main(args)
