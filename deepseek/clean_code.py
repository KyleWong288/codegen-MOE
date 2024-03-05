import json
import re

def clean_base_model(text):
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


def clean_base_model3(text):
    eot = "<|EOT|>"
    idx = text.find(eot)
    if idx != -1:
        text = text[:idx]
    return text


if __name__ == "__main__":
    
    input_file = "./output/dsc-6.7b-base/all/EASY/ft-checkpoint-1000.json"
    output_file = "./output/dsc-6.7b-base/all/EASY/ft-checkpoint-1000-clean.json"
    res = []

    with open(input_file, 'r') as file:
        json_data = json.load(file)

    for i in range(len(json_data)):
        old_data = json_data[i]
        data = {}
        data["task_id"] = old_data["task_id"]
        data["prompt"] = old_data["prompt"]
        data["output"] = []
        for old_code in old_data["output"]:
            clean_code = clean_base_model3(old_code)
            data["output"].append(clean_code)
        res.append(data)

    with open(output_file, 'w') as file:
        json.dump(res, file, indent=4)

    print("DONE")