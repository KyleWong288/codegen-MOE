import json
import os
from datasets import load_dataset

# Specify the name of the dataset you want to download
dataset_name = "deepmind/code_contests"
# Download the dataset
dataset = load_dataset(dataset_name)
test_data = dataset["test"]

# Write path
output_file = "./cc_data/code_contests.jsonl"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, "w") as file:
    for i in range(5):
        json_obj = test_data[i]
        data = {
            "name": json_obj["name"],
            "generated_tests": json_obj["generated_tests"]
        }
        file.write(json.dumps(data) + "\n")