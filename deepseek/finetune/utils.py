import os
import datasets


lora_module_dict = {
    'dsc-6.7b-instruct': [
        'q_proj', 'k_proj', 'v_proj',
        'o_proj', 'gate_proj', 'up_proj', 'down_proj',
    ],
}


def parse_model_name(name, from_remote=False):
    if name == "dsc-6.7b-instruct":
        return "deepseek-ai/deepseek-coder-6.7b-instruct"
    else:
        raise ValueError(f"Undefined base model: {name}")


def load_dataset(dataset_dir):
    train_file = os.path.join(dataset_dir, "train.jsonl")
    dev_file = os.path.join(dataset_dir, "dev.jsonl")
    train_data = datasets.load_dataset("json", data_files=train_file, split="train", download_mode="force_redownload")
    test_data = datasets.load_dataset("json", data_files=dev_file, split="train", download_mode="force_redownload")
    return train_data, test_data