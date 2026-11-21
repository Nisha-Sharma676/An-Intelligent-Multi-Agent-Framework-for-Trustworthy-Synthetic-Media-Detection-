import pickle
import numpy as np
import torch

from metrics import AP


# --------------------------------------------------
# 1. Load saved BMN predictions
# --------------------------------------------------

FILE_PATH = "lavdf_inferenceresults.pkl"

with open(FILE_PATH, "rb") as file:
    results = pickle.load(file)

if not isinstance(results, dict):
    raise TypeError("Prediction file must contain a dictionary.")

print("BMN prediction file loaded successfully!")


# --------------------------------------------------
# 2. Convert predictions and labels to CPU tensors
# --------------------------------------------------

def prepare_cpu_results(raw_results):
    """
    Move proposals and ground-truth labels to CPU.
    This prevents CPU/GPU device mismatch errors.
    """

    cpu_results = {}

    for video_id, data in raw_results.items():

        proposals = torch.as_tensor(
            data["proposals"]
        ).detach().cpu().float()

        labels = torch.as_tensor(
            data["labels"]
        ).detach().cpu().float()

        # Expected proposal format: [confidence, start, end]
        if proposals.numel() == 0:
            proposals = torch.empty((0, 3), dtype=torch.float32)
        elif proposals.numel() % 3 != 0:
            raise ValueError(
                f"Invalid proposal shape for video {video_id}: "
                f"{tuple(proposals.shape)}"
            )
        else:
            proposals = proposals.reshape(-1, 3)

        # Expected label format: [start, end]
        if labels.numel() == 0:
            labels = torch.empty((0, 2), dtype=torch.float32)
        elif labels.numel() % 2 != 0:
            raise ValueError(
                f"Invalid ground-truth label shape for video {video_id}: "
                f"{tuple(labels.shape)}"
            )
        else:
            labels = labels.reshape(-1, 2)

        # Validate interval coordinates
        if not torch.isfinite(proposals).all():
            raise ValueError(
                f"Non-finite proposal values found for video {video_id}."
            )

        if not torch.isfinite(labels).all():
            raise ValueError(
                f"Non-finite ground-truth values found for video {video_id}."
            )

        cpu_results[video_id] = {
            "proposals": proposals,
            "labels": labels
        }

    return cpu_results


cpu_results = prepare_cpu_results(results)


# --------------------------------------------------
# 3. Calculate temporal Intersection over Union (IoU)
# --------------------------------------------------

def calculate_iou(segment1, segment2):
    """Calculate temporal IoU between two time intervals."""

    start = max(float(segment1[0]), float(segment2[0]))
    end = min(float(segment1[1]), float(segment2[1]))

    intersection = max(0.0, end - start)

    length1 = max(0.0, float(segment1[1]) - float(segment1[0]))
    length2 = max(0.0, float(segment2[1]) - float(segment2[0]))

    union = length1 + length2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# --------------------------------------------------
# 4. Calculate confidence-ranked diagnostic AP
# --------------------------------------------------

def calculate_ap(prediction_results, threshold):
    """
    Calculate diagnostic temporal AP.

    Predictions are ranked globally by confidence.
    Each ground-truth interval can be matched only once.
    """

    predictions = []
    ground_truth = {}

    for video_id, data in prediction_results.items():

        labels = data["labels"].numpy()
        proposals = data["proposals"].numpy()

        ground_truth[video_id] = labels

        for score, start, end in proposals:

            # Ignore invalid or zero-duration intervals
            if end <= start:
                continue

            predictions.append(
                (float(score), video_id, float(start), float(end))
            )

    # Highest-confidence predictions are evaluated first
    predictions.sort(key=lambda item: item[0], reverse=True)

    total_gt = sum(len(labels) for labels in ground_truth.values())

    if total_gt == 0:
        return 0.0

    matched = {
        video_id: set()
        for video_id in ground_truth
    }

    true_positives = []
    false_positives = []

    for score, video_id, start, end in predictions:

        labels = ground_truth[video_id]

        best_iou = 0.0
        best_index = -1

        # Find the highest-IoU unmatched ground-truth interval
        for index, label in enumerate(labels):

            if index in matched[video_id]:
                continue

            iou = calculate_iou(
                (start, end),
                label
            )

            if iou > best_iou:
                best_iou = iou
                best_index = index

        if best_index >= 0 and best_iou >= threshold:

            matched[video_id].add(best_index)
            true_positives.append(1)
            false_positives.append(0)

        else:
            true_positives.append(0)
            false_positives.append(1)

    if not true_positives:
        return 0.0

    tp = np.cumsum(true_positives)
    fp = np.cumsum(false_positives)

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / total_gt

    # Add boundary points for interpolated AP calculation
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    # Construct the precision envelope
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(
            precision[index],
            precision[index + 1]
        )

    indices = np.where(recall[1:] != recall[:-1])[0]

    ap = np.sum(
        (recall[indices + 1] - recall[indices])
        * precision[indices + 1]
    )

    return float(ap)


# --------------------------------------------------
# 5. Evaluate with both implementations
# --------------------------------------------------

def main():

    thresholds = [0.5, 0.75, 0.9, 0.95]

    total_videos = len(cpu_results)

    total_gt = sum(
        len(data["labels"])
        for data in cpu_results.values()
    )

    empty_gt_videos = sum(
        len(data["labels"]) == 0
        for data in cpu_results.values()
    )

    total_proposals = sum(
        len(data["proposals"])
        for data in cpu_results.values()
    )

    print("\n" + "=" * 65)
    print("BMN TEMPORAL LOCALIZATION EVALUATION")
    print("=" * 65)

    print(f"Total videos             : {total_videos}")
    print(f"Total proposals          : {total_proposals}")
    print(f"Total GT intervals       : {total_gt}")
    print(f"Videos with empty labels : {empty_gt_videos}")
    print("Evaluation device        : CPU")

    if total_gt == 0:
        print("\nERROR: No ground-truth intervals found.")
        return

    print("\nAP COMPARISON")
    print("-" * 65)
    print(
        f"{'Threshold':<15}"
        f"{'Existing AP':<22}"
        f"{'Diagnostic AP':<22}"
    )
    print("-" * 65)

    for threshold in thresholds:

        # Existing evaluator from metrics.py
        existing_ap = AP(
            iou_thresholds=[threshold],
            device="cpu"
        )(cpu_results)[threshold]

        existing_ap = float(
            torch.as_tensor(existing_ap).detach().cpu()
        )

        # Independent confidence-ranked diagnostic evaluator
        diagnostic_ap = calculate_ap(
            cpu_results,
            threshold
        )

        print(
            f"AP@{threshold:.2f}       "
            f"{existing_ap * 100:>8.4f}%             "
            f"{diagnostic_ap * 100:>8.4f}%"
        )

    print("-" * 65)
    print("Evaluation completed.")
    print("Original prediction file was not modified.")
    print("=" * 65)


if __name__ == "__main__":
    main()