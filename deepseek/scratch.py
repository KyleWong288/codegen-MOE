import json
from datasets import load_dataset

# Specify the name of the dataset you want to download
dataset_name = "deepmind/code_contests"
# Download the dataset
dataset = load_dataset(dataset_name)
train_data = dataset["train"]

# Write path
output_file = "./cc_data/code_contests.jsonl"

with open(output_file, "w") as file:
    for i in range(2):
        json_entry = json.dumps(train_data[i])
        file.write(json_entry + "\n")