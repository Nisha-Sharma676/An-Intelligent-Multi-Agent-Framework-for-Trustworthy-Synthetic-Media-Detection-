import pickle
import torch
from utils import iou_1d


PREDICTION_FILE = "lavdf_inferenceresults.pkl"
THRESHOLDS = [0.5, 0.75, 0.9, 0.95]


def calculate_ap(values, total_labels):
    if total_labels == 0 or len(values) == 0:
        return 0.0

    values = values[values[:, 0].argsort(descending=True)]
    tp = values[:, 1].float()
    fp = 1.0 - tp

    tp_cum = torch.cumsum(tp, dim=0)
    fp_cum = torch.cumsum(fp, dim=0)

    recall = tp_cum / total_labels
    precision = tp_cum / (tp_cum + fp_cum)

    recall = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
    precision = torch.cat([torch.tensor([0.0]), precision, torch.tensor([0.0])])

    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    changes = torch.where(recall[1:] != recall[:-1])[0]
    ap = torch.sum(
        (recall[changes + 1] - recall[changes]) * precision[changes + 1]
    )
    return float(ap)


def evaluate(data, threshold):
    all_scores = []
    total_labels = 0

    for sample in data.values():
        proposals = sample["proposals"].detach().cpu().float()
        labels = torch.as_tensor(sample["labels"], dtype=torch.float32).reshape(-1, 2)
        total_labels += len(labels)

        if len(proposals) == 0:
            continue

        if len(labels) == 0:
            all_scores.append(
                torch.stack([proposals[:, 0], torch.zeros(len(proposals))], dim=1)
            )
            continue

        ious = iou_1d(proposals[:, 1:3], labels)
        order = proposals[:, 0].argsort(descending=True)

        used_labels = set()
        is_tp = torch.zeros(len(proposals))

        for idx in order.tolist():
            best_iou, best_label = ious[idx].max(dim=0)
            label_idx = int(best_label)

            if best_iou.item() >= threshold and label_idx not in used_labels:
                is_tp[idx] = 1.0
                used_labels.add(label_idx)

        all_scores.append(torch.stack([proposals[:, 0], is_tp], dim=1))

    if not all_scores:
        return 0.0

    return calculate_ap(torch.cat(all_scores), total_labels)


with open(PREDICTION_FILE, "rb") as f:
    predictions = pickle.load(f)

print("Samples:", len(predictions))
print("Evaluating saved predictions; training will not run.")

results = {}
for threshold in THRESHOLDS:
    results[threshold] = evaluate(predictions, threshold)
    print(f"Corrected AP@{threshold}: {results[threshold]:.6f}")

print("Corrected mAP:", f"{sum(results.values()) / len(results):.6f}")
print("Note: Compare these results with the original implementation.")