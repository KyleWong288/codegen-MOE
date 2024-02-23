import argparse
import json
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--output_path", default='./eval_results/gpt_test_2.json', type=str)
args = parser.parse_args()


if __name__ == "__main__":

    # Load the generations:
    results = []
    with open(args.output_path, 'r') as file:
        json_data = json.load(file)
        results = json_data["results"]

    # Compute metrics:
    # Acc: |pred intersection gt| / |pred union gt|
    # Precision: |pred intersection gt| / |pred|
    # Recall: |pred intersection gt | / |gt|
    acc = 0
    precision = 0
    recall = 0

    for result in tqdm(results):
        gt_labels = result["labels"]
        pred_labels = result["answers"]
        intersection = [label for label in pred_labels if label in gt_labels]
        union = list(set(pred_labels).union(set(gt_labels)))
        
        acc += len(intersection) / len(union)
        precision += len(intersection) / len(pred_labels) if len(pred_labels) > 0 else 0
        recall += len(intersection) / len(gt_labels)
        
    acc = acc / len(results)
    precision = precision / len(results)
    recall = recall / len(results)
    f1_score = 2 * precision * recall / (precision + recall)
    
    print("ACC:", acc)
    print("PRECISION:", precision)
    print("RECALL:", recall)
    print("F1 SCORE:", f1_score)