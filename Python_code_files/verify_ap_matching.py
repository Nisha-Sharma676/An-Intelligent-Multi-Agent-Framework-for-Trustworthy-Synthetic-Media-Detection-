import pickle
from pathlib import Path

import torch
from utils import iou_1d


RESULTS_FILE = Path("lavdf_inferenceresults.pkl")
THRESHOLDS = [0.5, 0.75, 0.9, 0.95]

if not RESULTS_FILE.exists():
    raise FileNotFoundError(
        f"Not found: {RESULTS_FILE.resolve()}"
    )

with RESULTS_FILE.open("rb") as f:
    results = pickle.load(f)


def temporal_iou(proposals, labels):
    """Standard temporal IoU on CPU."""
    proposals = torch.as_tensor(proposals).detach().cpu().float()
    labels = torch.as_tensor(labels).detach().cpu().float()

    if labels.numel() == 0:
        return torch.zeros((len(proposals), 0))

    p_start = proposals[:, 0:1]
    p_end = proposals[:, 1:2]
    l_start = labels[:, 0].unsqueeze(0)
    l_end = labels[:, 1].unsqueeze(0)

    intersection = (
        torch.minimum(p_end, l_end)
        - torch.maximum(p_start, l_start)
    ).clamp(min=0)

    union = (p_end - p_start) + (l_end - l_start) - intersection
    return intersection / union.clamp(min=1e-8)


def evaluate(threshold):
    detections = []
    total_gt = 0

    for video_id, item in results.items():
        proposals = torch.as_tensor(
            item["proposals"]
        ).detach().cpu().float()

        labels = torch.as_tensor(
            item["labels"]
        ).detach().cpu().float().reshape(-1, 2)

        total_gt += len(labels)

        if len(proposals) == 0:
            continue

        # Sort proposals by confidence, highest first.
        order = torch.argsort(proposals[:, 0], descending=True)
        proposals = proposals[order]

        ious = temporal_iou(proposals[:, 1:3], labels)
        matched = set()

        for proposal in proposals:
            score = float(proposal[0])
            best_iou = -1.0
            best_label = None

            for label_idx in range(len(labels)):
                if label_idx in matched:
                    continue

                # IoU for this proposal and this ground-truth interval.
                iou = float(
                    temporal_iou(
                        proposal[1:3].reshape(1, 2),
                        labels[label_idx:label_idx + 1]
                    )[0, 0]
                )

                if iou > best_iou:
                    best_iou = iou
                    best_label = label_idx

            is_tp = (
                best_label is not None
                and best_iou >= threshold
            )

            if is_tp:
                matched.add(best_label)

            detections.append((score, int(is_tp)))

    if total_gt == 0:
        return 0.0

    detections.sort(key=lambda x: x[0], reverse=True)

    tp = 0
    fp = 0
    precisions = []
    recalls = []

    for _, is_tp in detections:
        if is_tp:
            tp += 1
        else:
            fp += 1

        precisions.append(tp / max(tp + fp, 1))
        recalls.append(tp / total_gt)

    # Precision-envelope AP calculation.
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]

    return ap


print(f"Test videos: {len(results)}")
print("\nConfidence-ranked AP diagnostic:")

scores = {}
for threshold in THRESHOLDS:
    scores[threshold] = evaluate(threshold)
    print(f"AP@{threshold}: {scores[threshold]:.6f}")

print(f"\nMean AP: {sum(scores.values()) / len(scores):.6f}")
print("\nDiagnostic complete. Original files were not modified.")
print("This is a comparison implementation, not yet a confirmed reproduction of the paper's protocol.")