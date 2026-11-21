import pickle
import numpy as np
from pathlib import Path

# Existing BMN predictions file
PREDICTION_FILE = Path("lavdf_inferenceresults.pkl")

IOU_THRESHOLDS = [0.50, 0.75]
TOP_K_VALUES = [1, 5, 10, 20]


def calculate_iou(proposal, ground_truth):
    start = max(float(proposal[0]), float(ground_truth[0]))
    end = min(float(proposal[1]), float(ground_truth[1]))

    intersection = max(0.0, end - start)

    proposal_length = max(
        0.0, float(proposal[1]) - float(proposal[0])
    )
    gt_length = max(
        0.0, float(ground_truth[1]) - float(ground_truth[0])
    )

    union = proposal_length + gt_length - intersection

    return intersection / union if union > 0 else 0.0


def main():
    if not PREDICTION_FILE.exists():
        print("ERROR: Prediction file not found!")
        print("Expected:", PREDICTION_FILE.resolve())
        return

    with open(PREDICTION_FILE, "rb") as file:
        results = pickle.load(file)

    total_gt = 0
    hits = {
        threshold: {k: 0 for k in TOP_K_VALUES}
        for threshold in IOU_THRESHOLDS
    }

    videos_with_gt = 0

    for video_name, item in results.items():
        proposals = item["proposals"]
        labels = item["labels"]

        if hasattr(proposals, "detach"):
            proposals = proposals.detach().cpu().numpy()
        else:
            proposals = np.asarray(proposals)

        if hasattr(labels, "detach"):
            labels = labels.detach().cpu().numpy()
        else:
            labels = np.asarray(labels)

        proposals = np.asarray(proposals).reshape(-1, 3)
        labels = np.asarray(labels).reshape(-1, 2)

        # Sort proposals by confidence, highest first.
        if len(proposals) > 0:
            proposals = proposals[
                np.argsort(-proposals[:, 0])
            ]

        # Proposal format: [confidence, start, end]
        # Ground-truth format: [start, end]
        if len(labels) == 0:
            continue

        videos_with_gt += 1

        for gt in labels:
            total_gt += 1

            for threshold in IOU_THRESHOLDS:
                for k in TOP_K_VALUES:
                    top_proposals = proposals[:k, 1:3]

                    if len(top_proposals) == 0:
                        continue

                    best_iou = max(
                        calculate_iou(proposal, gt)
                        for proposal in top_proposals
                    )

                    if best_iou >= threshold:
                        hits[threshold][k] += 1

    print("\n========== BMN RANKING AUDIT ==========")
    print("Prediction file:", PREDICTION_FILE.resolve())
    print("Videos with ground truth:", videos_with_gt)
    print("Total ground-truth intervals:", total_gt)

    if total_gt == 0:
        print("ERROR: No ground-truth intervals found.")
        return

    for threshold in IOU_THRESHOLDS:
        print(f"\nRecall at IoU >= {threshold:.2f}")

        for k in TOP_K_VALUES:
            count = hits[threshold][k]
            recall = 100.0 * count / total_gt

            print(
                f"Top-{k:2d}: {count}/{total_gt} "
                f"({recall:.2f}%)"
            )

    print("\nAudit complete. No model weights were changed.")


if __name__ == "__main__":
    main()