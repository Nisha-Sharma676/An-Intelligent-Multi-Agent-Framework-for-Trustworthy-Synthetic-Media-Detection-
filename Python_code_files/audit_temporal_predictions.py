import pickle
import numpy as np
import torch

FILE_PATH = "lavdf_inferenceresults.pkl"

with open(FILE_PATH, "rb") as file:
    results = pickle.load(file)

print("=" * 65)
print("BMN TEMPORAL PREDICTION AUDIT")
print("=" * 65)

total_videos = len(results)
total_proposals = 0
total_gt = 0
empty_gt_videos = 0
invalid_proposals = 0
invalid_gt = 0

proposal_starts = []
proposal_ends = []
proposal_scores = []
gt_starts = []
gt_ends = []

for video_id, data in results.items():

    proposals = torch.as_tensor(
        data["proposals"]
    ).detach().cpu().numpy().reshape(-1, 3)

    labels = torch.as_tensor(
        data["labels"]
    ).detach().cpu().numpy().reshape(-1, 2)

    total_proposals += len(proposals)
    total_gt += len(labels)

    if len(labels) == 0:
        empty_gt_videos += 1

    # Check prediction validity
    if len(proposals):
        invalid_proposals += int(
            np.sum(
                (~np.isfinite(proposals).all(axis=1))
                | (proposals[:, 2] <= proposals[:, 1])
            )
        )

        valid = (
            np.isfinite(proposals).all(axis=1)
            & (proposals[:, 2] > proposals[:, 1])
        )

        proposal_starts.extend(proposals[valid, 1])
        proposal_ends.extend(proposals[valid, 2])
        proposal_scores.extend(proposals[valid, 0])

    # Check ground-truth validity
    if len(labels):
        invalid_gt += int(
            np.sum(
                (~np.isfinite(labels).all(axis=1))
                | (labels[:, 1] <= labels[:, 0])
            )
        )

        valid_labels = (
            np.isfinite(labels).all(axis=1)
            & (labels[:, 1] > labels[:, 0])
        )

        gt_starts.extend(labels[valid_labels, 0])
        gt_ends.extend(labels[valid_labels, 1])

print("\n1. DATASET SUMMARY")
print("-" * 45)
print("Videos:", total_videos)
print("Proposals:", total_proposals)
print("Ground-truth intervals:", total_gt)
print("Videos with empty GT:", empty_gt_videos)

print("\n2. INVALID INTERVAL CHECK")
print("-" * 45)
print("Invalid proposals:", invalid_proposals)
print("Invalid GT intervals:", invalid_gt)

print("\n3. PREDICTION TIMESTAMP RANGE")
print("-" * 45)

if proposal_starts:
    print("Minimum start:", round(float(np.min(proposal_starts)), 4))
    print("Maximum start:", round(float(np.max(proposal_starts)), 4))
    print("Minimum end:", round(float(np.min(proposal_ends)), 4))
    print("Maximum end:", round(float(np.max(proposal_ends)), 4))
    print("Minimum score:", round(float(np.min(proposal_scores)), 6))
    print("Maximum score:", round(float(np.max(proposal_scores)), 6))
else:
    print("No valid proposals found.")

print("\n4. GROUND-TRUTH TIMESTAMP RANGE")
print("-" * 45)

if gt_starts:
    print("Minimum start:", round(float(np.min(gt_starts)), 4))
    print("Maximum start:", round(float(np.max(gt_starts)), 4))
    print("Minimum end:", round(float(np.min(gt_ends)), 4))
    print("Maximum end:", round(float(np.max(gt_ends)), 4))
else:
    print("No valid ground-truth intervals found.")

print("\n5. FIRST THREE VIDEO SAMPLES")
print("-" * 45)

for video_id, data in list(results.items())[:3]:

    proposals = torch.as_tensor(
        data["proposals"]
    ).detach().cpu().numpy().reshape(-1, 3)

    labels = torch.as_tensor(
        data["labels"]
    ).detach().cpu().numpy().reshape(-1, 2)

    print("\nVideo ID:", video_id)
    print("First 3 proposals [score, start, end]:")
    print(proposals[:3])
    print("First 3 GT intervals [start, end]:")
    print(labels[:3])

print("\n" + "=" * 65)
print("AUDIT COMPLETE — original predictions unchanged.")
print("=" * 65)