import json
import os
import numpy as np

def load_generation(input_file):
    generations = {}
    with open(input_file, 'r') as f:
        results = json.load(f)
        for _, res in enumerate(results):
            task_id = res['task_id']
            output = res['output']
            generations[task_id] = output
    return generations

def load_gen2(input_file):
    results = []
    with open(input_file, 'r') as f:
        json_objects = json.load(f)
        for obj in json_objects:
            results.append(obj)
    print(results[0])


def compute_pass_at_k(num_correct):

    def estimate_pass_at_k(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
    
    pass1_all = [round(estimate_pass_at_k(20, c, 1), 8) for c in num_correct]
    pass10_all = [round(estimate_pass_at_k(20, c, 10), 8) for c in num_correct]
    pass1 = np.array(pass1_all).mean()
    pass10 = np.array(pass10_all).mean()

    res = {
            "pass@1": pass1,
            "pass@10": pass10,
            "detail": {
                "pass@1": {key: value for key, value in enumerate(pass1_all)},
                "pass@10": {key: value for key, value in enumerate(pass10_all)}
            }
    }
    return res


if __name__ == "__main__":

    result_dir = "./eval_results/dsc-6.7b-base/base/"
    # result_dir = "./eval_results/dsc-6.7b-base/4000_dsc_all-checkpoint-1000/"
    output_file = os.path.join(result_dir, "eval.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    num_correct = [0] * 50

    for num_sample in range(20):
        result_file = result_dir + f"sample_{num_sample}.jsonl"
        with open(result_file, "r") as file:
            for i, line in enumerate(file):
                obj = json.loads(line)
                num_correct[i] += obj["passed"]
    
    res = compute_pass_at_k(num_correct)         

    with open(output_file, "w") as file:
        json.dump(res, file, indent=4)
