import argparse
import os
import re
import json
from pathlib import Path
from collections import defaultdict
from eval_utils.evaluation import evaluate_functional_correctness


VERSION = "20240121-Jul"
NUM_SAMPLES = 20

def extract_python_code(generation: str):
    generation = generation.replace("[PYTHON]", '```python').replace("[/PYTHON]", '```')
    if '```python' in generation:
        p_code = re.compile(r'```python\n(.*?)\n```', flags=re.DOTALL)
        matches = p_code.findall(generation)
        code_block = matches[0] if matches else generation
        return code_block
    else:
        codelist = re.split("\ndef|\nclass|\nif|\n#|\nprint", generation)
        return codelist[0]
    

def extract_python_code_2(text: str):
    # remove python prefix
    p1 = "\n```python\n"
    p2 = "```python\n"
    prefixes = [p1, p2]
    for pref in prefixes:
        idx = text.find(pref)
        if idx != -1:
            text = text[len(pref):]
    # remove eot token
    eot = "<|EOT|>"
    idx = text.find(eot)
    if idx != -1:
        text = text[:idx]
    return text


def main(args):

    # Load questions
    test_data = [json.loads(line) for line in open(args.data_path).readlines()]
    id2problems = { x['task_id']: x for x in test_data }

    for sample_num in range(NUM_SAMPLES):

        # Load results
        generation_file = os.path.join(args.generation_dir, f"sample_{sample_num}.jsonl")
        results = []
        with open(generation_file, 'r') as f:
            json_objects = json.load(f)
            for obj in json_objects:
                results.append(obj)

        # Extract the clean code
        for result in results:
            result["generation"] = extract_python_code(result["output"])
            # result["generation"] = extract_python_code_2(result["output"])

        # Write to result file
        result_file = os.path.join(args.result_dir, f"sample_{sample_num}.jsonl")
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        with open(result_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

        # Evaluate
        score = evaluate_functional_correctness(
            input_file=result_file,
            tmp_dir=args.temp_dir,
            problem_file=args.data_path,
            result_path=result_file
        )

        hardness_results = defaultdict(int)
        for result in [json.loads(line) for line in open(result_file, 'r')]:
            problem = id2problems[result['task_id']]

            hardness = problem['meta']['difficulty']
            hardness_results[hardness] += 1
            hardness_results[hardness + "_correct"] += result['passed']

        print("="*100)
        print("Evaluate {} over.".format(generation_file))
        print("Pass@1: {:.3f}".format(score["pass@1"]))
        for key in ["Easy", "Medium", "Hard"]:
            if key.endswith("_correct"):
                continue
            acc = hardness_results[key+"_correct"] / hardness_results[key]
            print("{}: {:.3f}({}/{})".format(key, acc, hardness_results[key+"_correct"],  hardness_results[key]))
        
        print(score)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--generation_dir", type=str, required=True)
    parser.add_argument("--result_dir", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--temp_dir", type=str, default="output/temp")
    args = parser.parse_args()

    main(args)
