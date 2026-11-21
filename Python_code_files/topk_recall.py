import pickle
import torch
from utils import iou_1d

with open("lavdf_inferenceresults.pkl", "rb") as f:
    data = pickle.load(f)

counts = {5: 0, 10: 0, 20: 0}
total = 0

for sample in data.values():
    proposals = sample["proposals"].detach().cpu().float()
    labels = torch.as_tensor(
        sample["labels"], dtype=torch.float32
    ).reshape(-1, 2)

    if len(labels) == 0 or len(proposals) == 0:
        continue

    total += 1
    order = proposals[:, 0].argsort(descending=True)

    for k in counts:
        selected = proposals[order[:k]]
        overlaps = iou_1d(selected[:, 1:3], labels)

        if (overlaps >= 0.5).any():
            counts[k] += 1

print("Fake samples evaluated:", total)
for k, count in counts.items():
    print(f"Top {k} proposals with IoU >= 0.5: {count}/{total}")