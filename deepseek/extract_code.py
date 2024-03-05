import json
import os

def load_generation(input_file):
    generations = {}
    with open(input_file, 'r') as f:
        results = json.load(f)
        for _, res in enumerate(results):
            task_id = res['task_id']
            output = res['output']
            generations[task_id] = output
    return generations

if __name__ == "__main__":
    input_file = "./output/dsc-6.7b-instruct/all/4000_dsc_all_format.json"
    generations = load_generation(input_file)
    code = generations[24][0]
    
