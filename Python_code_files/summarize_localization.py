import pickle
import torch
from utils import iou_1d

FILE = "lavdf_inferenceresults.pkl"

with open(FILE, "rb") as f:
    data = pickle.load(f)

total = len(data)
with_fake = 0
empty = 0
top1_ious = []
top1_scores = []
best_ious = []

for sample in data.values():
    proposals = sample["proposals"].detach().cpu().float()
    labels = torch.as_tensor(
        sample["labels"], dtype=torch.float32
    ).reshape(-1, 2)

    if len(labels) == 0:
        empty += 1
        continue

    with_fake += 1

    # Top proposal = highest-confidence proposal
    top_idx = proposals[:, 0].argmax()
    top_proposal = proposals[top_idx:top_idx + 1]

    top1_scores.append(float(top_proposal[0, 0]))

    ious = iou_1d(top_proposal[:, 1:3], labels)
    top1_ious.append(float(ious.max()))

    # Best overlap among all proposals for this sample
    all_ious = iou_1d(proposals[:, 1:3], labels)
    best_ious.append(float(all_ious.max()))

print("Total test samples:", total)
print("Samples with fake intervals:", with_fake)
print("Samples without fake intervals:", empty)
print("Samples with fake intervals checked:", len(top1_ious))

if top1_ious:
    print("Mean top-1 confidence:", sum(top1_scores) / len(top1_scores))
    print("Mean top-1 IoU:", sum(top1_ious) / len(top1_ious))
    print("Mean best-proposal IoU:", sum(best_ious) / len(best_ious))

    for threshold in [0.5, 0.75, 0.9]:
        top1_count = sum(x >= threshold for x in top1_ious)
        best_count = sum(x >= threshold for x in best_ious)
        print(
            f"IoU >= {threshold}: "
            f"top-1={top1_count}/{len(top1_ious)}, "
            f"best proposal={best_count}/{len(best_ious)}"
        )