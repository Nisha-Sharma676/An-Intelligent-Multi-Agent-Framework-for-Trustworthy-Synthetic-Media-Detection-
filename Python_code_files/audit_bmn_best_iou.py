import pickle
import numpy as np
from pathlib import Path

PREDICTION_FILE = Path("lavdf_inferenceresults.pkl")
IOU_THRESHOLDS = [0.50, 0.75, 0.90, 0.95]


def calculate_iou(proposal, gt):
    intersection = max(
        0.0,
        min(float(proposal[1]), float(gt[1]))
        - max(float(proposal[0]), float(gt[0]))
    )

    proposal_length = max(
        0.0, float(proposal[1]) - float(proposal[0])
    )
    gt_length = max(0.0, float(gt[1]) - float(gt[0]))

    union = proposal_length + gt_length - intersection

    return intersection / union if union > 0 else 0.0


def main():
    if not PREDICTION_FILE.exists():
        print("ERROR: File not found:", PREDICTION_FILE.resolve())
        return

    with open(PREDICTION_FILE, "rb") as f:
        results = pickle.load(f)

    best_ious = []
    best_proposal_scores = []

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

        # Proposal columns: confidence, start, end
        # GT columns: start, end
        for gt in labels:
            if len(proposals) == 0:
                best_ious.append(0.0)
                best_proposal_scores.append(0.0)
                continue

            ious = [
                calculate_iou(proposal[1:3], gt)
                for proposal in proposals
            ]

            best_index = int(np.argmax(ious))
            best_ious.append(ious[best_index])
            best_proposal_scores.append(
                float(proposals[best_index, 0])
            )

    best_ious = np.asarray(best_ious)
    best_proposal_scores = np.asarray(best_proposal_scores)

    print("\n========== BMN BEST-IOU AUDIT ==========")
    print("Prediction file:", PREDICTION_FILE.resolve())
    print("Total GT intervals:", len(best_ious))

    if len(best_ious) == 0:
        print("ERROR: No ground-truth intervals found.")
        return

    print(f"Mean best IoU: {best_ious.mean():.4f}")
    print(f"Median best IoU: {np.median(best_ious):.4f}")

    for threshold in IOU_THRESHOLDS:
        count = int(np.sum(best_ious >= threshold))
        recall = 100.0 * count / len(best_ious)

        print(
            f"Best possible IoU >= {threshold:.2f}: "
            f"{count}/{len(best_ious)} ({recall:.2f}%)"
        )

    print(
        "\nMean confidence of best-IoU proposals:",
        f"{best_proposal_scores.mean():.6f}"
    )

    print("\nAudit complete. No model weights were changed.")


if __name__ == "__main__":
    main()