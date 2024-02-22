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


if __name__ == "__main__":
    
    input_file = "./output/dsc-6.7b-instruct/data_structures/base_copy.json"
    output_file = "./output/dsc-6.7b-instruct/data_structures/clean_base.json"
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
            clean_code = clean_base_model(old_code)
            data["output"].append(clean_code)
        res.append(data)

    with open(output_file, 'w') as file:
        json.dump(res, file, indent=4)

    print("DONE")