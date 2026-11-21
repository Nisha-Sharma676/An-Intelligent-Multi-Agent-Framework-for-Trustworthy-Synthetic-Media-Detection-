import json
import pickle
from pathlib import Path

import numpy as np
import torch

PREDICTIONS_FILE = Path("lavdf_inferenceresults.pkl")
METADATA_FILE = Path(r"D:\LAV-DF\LAV-DF\metadata.json")


def to_numpy(value):
    return torch.as_tensor(value).detach().cpu().numpy()


def interval_iou(start, end, gt_start, gt_end):
    intersection = max(0.0, min(end, gt_end) - max(start, gt_start))
    union = max(end, gt_end) - min(start, gt_start)

    return intersection / union if union > 0 else 0.0


def main():
    with open(PREDICTIONS_FILE, "rb") as file:
        predictions = pickle.load(file)

    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    metadata_index = {
        record["file"].replace("\\", "/").lower(): record
        for record in metadata
        if record.get("file")
    }

    duration_violations = 0
    invalid_proposals = 0
    videos_with_gt = 0
    total_gt = 0
    top1_ious = []
    best_ious = []
    best_proposal_recalls = {0.5: 0, 0.75: 0, 0.95: 0}
    high_conf_scores = []

    for video_id, data in predictions.items():
        name = Path(video_id).stem
        split, index = name.split("_", 1)
        key = f"{split}/{index}.mp4".lower()

        record = metadata_index.get(key)
        if record is None:
            continue

        duration = float(record["duration"])
        proposals = to_numpy(data["proposals"]).reshape(-1, 3)
        labels = to_numpy(data["labels"]).reshape(-1, 2)

        valid = (
            np.isfinite(proposals).all(axis=1)
            & (proposals[:, 2] > proposals[:, 1])
        )
        invalid_proposals += int((~valid).sum())
        proposals = proposals[valid]

        if len(proposals):
            duration_violations += int(
                np.sum(
                    (proposals[:, 1] < 0)
                    | (proposals[:, 2] > duration + 1e-3)
                )
            )

            order = np.argsort(-proposals[:, 0])
            proposals = proposals[order]
            high_conf_scores.append(float(proposals[0, 0]))

        if len(labels) == 0:
            continue

        videos_with_gt += 1
        total_gt += len(labels)

        # Best IoU across proposals for each ground-truth interval
        for gt_start, gt_end in labels:
            ious = [
                interval_iou(start, end, gt_start, gt_end)
                for score, start, end in proposals
            ]

            best_iou = max(ious, default=0.0)
            best_ious.append(best_iou)

            for threshold in best_proposal_recalls:
                if best_iou >= threshold:
                    best_proposal_recalls[threshold] += 1

            top1_iou = (
                interval_iou(
                    proposals[0, 1], proposals[0, 2],
                    gt_start, gt_end
                )
                if len(proposals) else 0.0
            )
            top1_ious.append(top1_iou)

    print("=" * 65)
    print("BMN PROPOSAL QUALITY AUDIT")
    print("=" * 65)
    print("Prediction videos:", len(predictions))
    print("Videos with GT:", videos_with_gt)
    print("GT intervals:", total_gt)
    print("Invalid proposals:", invalid_proposals)
    print("Proposals outside video duration:", duration_violations)

    if total_gt:
        print("\nBest proposal recall by IoU threshold:")
        for threshold, count in best_proposal_recalls.items():
            print(
                f"IoU >= {threshold:.2f}: "
                f"{count}/{total_gt} ({100 * count / total_gt:.2f}%)"
            )

        print(
            "\nMean best-proposal IoU:",
            round(float(np.mean(best_ious)), 4)
        )
        print(
            "Mean top-1 IoU:",
            round(float(np.mean(top1_ious)), 4)
        )

    if high_conf_scores:
        print(
            "Mean top-confidence score:",
            round(float(np.mean(high_conf_scores)), 6)
        )

    print("\nAudit complete. Predictions were not modified.")


if __name__ == "__main__":
    main()