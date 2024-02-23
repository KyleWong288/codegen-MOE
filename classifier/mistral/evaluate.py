import argparse
import json
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--output_path", default='./output/roberta_test-sample.json', type=str)
parser.add_argument("--eval_type", default=1, type=int)
args = parser.parse_args()


# extracts the K highest output logits
# K is set by eval_type
def get_labels(probs, topK):
    res = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:topK]
    return res


if __name__ == "__main__":

    # Load the generations:
    results = []
    with open(args.output_path, 'r') as file:
        json_data = json.load(file)
        results = json_data["results"]

    # Compute metrics:
    acc = 0
    for result in tqdm(results):
        gt_labels = result["gt"]
        pred_labels = get_labels(result["softmax"][0], args.eval_type)
        print(pred_labels)
        print(gt_labels)
        intersection = [label for label in pred_labels if label in gt_labels]
        acc += (len(intersection) > 0)

    acc = acc / len(results)
    print("ACCURACY:", acc)