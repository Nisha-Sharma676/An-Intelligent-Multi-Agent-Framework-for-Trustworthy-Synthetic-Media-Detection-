import os
import re
import json
import pickle
import numpy as np

BASELINE_FILE = (
    r"D:\ics-av-deepfake_team_share\baseline_final"
    r"\lavdf_test_results.pkl"
)
PROPOSED_FILE = (
    r"D:\proposed_model_package\proposed_model\outputs"
    r"\temporal_localizer_test_predictions.json"
)
OUTPUT_FILE = (
    r"D:\ics-av-deepfake_team_share"
    r"\common_temporal_ap_comparison.json"
)


def as_list(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def normalize_labels(value):
    result = []
    for label in as_list(value or []):
        if len(label) >= 2:
            start, end = float(label[0]), float(label[1])
            if end > start:
                result.append([start, end])
    return result


def normalize_predictions(value):
    result = []
    for proposal in value:
        if len(proposal) >= 3:
            score, start, end = map(float, proposal[:3])
            if end > start:
                result.append([score, start, end])
    return result


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def evaluate_ap(predictions, labels_by_video, threshold):
    total_gt = sum(len(x) for x in labels_by_video.values())
    ranked = []

    for video, proposals in predictions.items():
        for score, start, end in proposals:
            ranked.append((score, video, start, end))

    ranked.sort(key=lambda x: x[0], reverse=True)

    matched = {video: set() for video in labels_by_video}
    tp, fp = [], []

    for score, video, start, end in ranked:
        labels = labels_by_video[video]
        candidates = [
            (iou([start, end], gt), idx)
            for idx, gt in enumerate(labels)
            if idx not in matched[video]
        ]

        best_iou, best_idx = max(candidates, default=(0.0, -1))

        if best_idx >= 0 and best_iou >= threshold:
            matched[video].add(best_idx)
            tp.append(1.0)
            fp.append(0.0)
        else:
            tp.append(0.0)
            fp.append(1.0)

    if total_gt == 0 or not tp:
        return 0.0

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recall = tp / total_gt
    precision = tp / np.maximum(tp + fp, 1e-12)

    mrec = np.r_[0.0, recall, 1.0]
    mpre = np.r_[0.0, precision, 0.0]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(
        np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    )


with open(BASELINE_FILE, "rb") as f:
    baseline_raw = pickle.load(f)

with open(PROPOSED_FILE, "r", encoding="utf-8") as f:
    proposed_raw = json.load(f)

baseline = {}
baseline_labels = {}

for path, record in baseline_raw.items():
    match = re.search(r"test_(\d+)\.pkl$", os.path.basename(path))
    if not match:
        continue

    video = int(match.group(1))
    baseline[video] = normalize_predictions(
        as_list(record["proposals"])
    )
    baseline_labels[video] = normalize_labels(record.get("labels", []))

proposed = {}
proposed_labels = {}

for path, record in proposed_raw.items():
    match = re.search(r"test/(\d+)\.mp4$", path.replace("\\", "/"))
    if not match:
        continue

    video = int(match.group(1))
    proposed[video] = normalize_predictions(record.get("proposals", []))
    proposed_labels[video] = normalize_labels(record.get("labels", []))

common = sorted(
    set(baseline) & set(proposed)
    & set(baseline_labels) & set(proposed_labels)
)

if not common:
    raise RuntimeError("No matching test video IDs found.")

for video in common:
    if baseline_labels[video] != proposed_labels[video]:
        raise RuntimeError(f"Ground-truth mismatch for video ID {video}")

labels = {video: baseline_labels[video] for video in common}
b_preds = {video: baseline[video] for video in common}
p_preds = {video: proposed[video] for video in common}

results = {
    "videos_compared": len(common),
    "ground_truth_segments": sum(len(x) for x in labels.values()),
    "evaluator": (
        "Global confidence ranking; one-to-one greedy matching "
        "to highest-IoU unmatched ground truth; IoU >= threshold; "
        "precision-recall envelope integration"
    ),
    "baseline_proposals_per_video": {
        "min": min(map(len, b_preds.values())),
        "max": max(map(len, b_preds.values())),
    },
    "proposed_proposals_per_video": {
        "min": min(map(len, p_preds.values())),
        "max": max(map(len, p_preds.values())),
    },
    "AP": {},
}

for threshold in (0.5, 0.75, 0.95):
    b_ap = evaluate_ap(b_preds, labels, threshold)
    p_ap = evaluate_ap(p_preds, labels, threshold)

    results["AP"][str(threshold)] = {
        "BMN_baseline": b_ap,
        "proposed_localizer": p_ap,
        "difference": p_ap - b_ap,
    }

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
print("\nSaved:", OUTPUT_FILE)