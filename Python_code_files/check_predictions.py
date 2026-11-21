import pickle
import torch
from utils import iou_1d

with open("lavdf_inferenceresults.pkl", "rb") as f:
    predictions = pickle.load(f)

thresholds = [0.5, 0.75]
results = {t: 0 for t in thresholds}
total_with_gt = 0
total_gt = 0

for item in predictions.values():
    proposals = item["proposals"].detach().cpu()
    labels = torch.as_tensor(item["labels"]).float()
    total_gt += len(labels)

    if len(labels) == 0:
        continue

    total_with_gt += 1

    # Highest-confidence proposal
    top_idx = proposals[:, 0].argmax()
    top_proposal = proposals[top_idx:top_idx + 1, 1:]
    ious = iou_1d(top_proposal, labels)

    best_iou = ious.max().item()

    for t in thresholds:
        if best_iou >= t:
            results[t] += 1

print("Total test samples:", len(predictions))
print("Samples with ground truth:", total_with_gt)
print("Total ground-truth intervals:", total_gt)

for t in thresholds:
    print(
        f"Top-confidence proposal IoU >= {t}: "
        f"{results[t]}/{total_with_gt}"
    )