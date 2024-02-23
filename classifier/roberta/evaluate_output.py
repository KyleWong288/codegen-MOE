import argparse
import json
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--output_path", default='./output/roberta_test.json', type=str)
parser.add_argument("--eval_type", default=1, type=int)
args = parser.parse_args()


# extracts the K highest output logits
# K is set by eval_type
def get_labels(probs, topK):
    res = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:topK]
    return res


# extracts any softmax prob higher than threshold
def get_labels_thresh(probs, threshold):
    res = [idx for idx, prob in enumerate(probs) if prob >= threshold]
    return res


if __name__ == "__main__":

    # Load the generations:
    results = []
    with open(args.output_path, 'r') as file:
        json_data = json.load(file)
        results = json_data["results"]
    THRESHOLD = 0.1

    # Compute metrics:
    # Regular acc: |pred intersection gt| / |pred union gt|
    # Simple acc: +1 if any pred label exists in gt label
    # Threshold acc: if softmax is > threshold, count it as a pred label
    # Precision: |pred intersection gt| / |pred|
    # Recall: |pred intersection gt | / |gt|
    acc = 0
    precision = 0
    recall = 0
    acc_simple = 0
    acc_thresh = 0
    precision_thresh = 0
    recall_thresh = 0

    for result in tqdm(results):

        gt_labels = result["gt"]
        pred_labels = get_labels(result["softmax"][0], args.eval_type)
        pred_labels_thr = get_labels_thresh(result["softmax"][0], THRESHOLD)
        intersection = [label for label in pred_labels if label in gt_labels]
        union = list(set(pred_labels).union(set(gt_labels)))
        intersection_thr = [label for label in pred_labels_thr if label in gt_labels]
        union_thr = list(set(pred_labels_thr).union(set(gt_labels)))

        acc += len(intersection) / len(union)
        precision += len(intersection) / len(pred_labels) if len(pred_labels) > 0 else 0
        recall += len(intersection) / len(gt_labels)
        acc_simple += (len(intersection) > 0)
        acc_thresh += len(intersection_thr) / len(union_thr)
        precision_thresh += len(intersection_thr) / len(pred_labels_thr) if len(pred_labels_thr) > 0 else 0
        recall_thresh += len(intersection_thr) / len(gt_labels)
        

    acc_simple = acc_simple / len(results)
    acc = acc / len(results)
    precision = precision / len(results)
    recall = recall / len(results)
    f1_score = 2 * precision * recall / (precision + recall)
    acc_thresh = acc_thresh / len(results)
    precision_thresh = precision_thresh / len(results)
    recall_thresh = recall_thresh / len(results)
    f1_score_thresh = 2 * precision_thresh * recall_thresh / (precision_thresh + recall_thresh)

    print("ACC:", acc)
    print("PRECISION:", precision)
    print("RECALL:", recall)
    print("F1 SCORE:", f1_score)
    print("SIMPLE ACC:", acc_simple)
    print("THRESHOLD ACC:", acc_thresh)
    print("PRECISION_THRESH:", precision_thresh)
    print("RECALL_THRESH:", recall_thresh)
    print("F1 SCORE THRESH:", f1_score_thresh)
    