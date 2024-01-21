import os
import datasets

lora_module_dict = {
    'codellama': [
        'q_proj', 'k_proj', 'v_proj',
        'o_proj', 'gate_proj', 'up_proj', 'down_proj',
    ],
    'codellama_python': [
        'q_proj', 'k_proj', 'v_proj',
        'o_proj', 'gate_proj', 'up_proj', 'down_proj',
    ],
    'llama2-7b': [
        'q_proj', 'k_proj', 'v_proj',
        'o_proj', 'gate_proj', 'up_proj', 'down_proj',
    ],
}


def tokenize(args, tokenizer, feature):
    
    prompt_ids = tokenizer.encode(
        feature['prompt'].strip(), padding=False,
        max_length=args.max_length, truncation=True
    )
    
    target_ids = tokenizer.encode(
        feature['answer'].strip(), padding=False,
        max_length=args.max_length, truncation=True, add_special_tokens=False
    )
    
    input_ids = prompt_ids + target_ids
    exceed_max_length = len(input_ids) >= args.max_length
    
     # Add EOS Token
    if input_ids[-1] != tokenizer.eos_token_id and not exceed_max_length:
        input_ids.append(tokenizer.eos_token_id)
    
    label_ids = [tokenizer.pad_token_id] * len(prompt_ids) + input_ids[len(prompt_ids):]
    
    return {
        "input_ids": input_ids,
        "labels": label_ids,
        "exceed_max_length": exceed_max_length
    }


def parse_model_name(name, from_remote=False):
    if name == "codellama":
        return "codellama/CodeLlama-7b-hf"
    elif name == "codellama_python":
        return "codellama/CodeLlama-7b-Python-hf"
    elif name == "llama2-7b":
        return "meta-llama/Llama-2-7b-chat-hf"
    else:
        raise ValueError(f"Undefined base model: {name}")
        
    
def load_dataset(dataset_dir):
    train_file = os.path.join(dataset_dir, "train.jsonl")
    dev_file = os.path.join(dataset_dir, "dev.jsonl")
    train_data = datasets.load_dataset("json", data_files=train_file, split="train", download_mode="force_redownload")
    test_data = datasets.load_dataset("json", data_files=dev_file, split="train", download_mode="force_redownload")
    return train_data, test_data
