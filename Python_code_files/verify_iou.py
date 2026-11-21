import pickle
from pathlib import Path

import torch
import utils
from metrics import AP


RESULTS = Path("lavdf_inferenceresults.pkl")

if not RESULTS.exists():
    raise FileNotFoundError(
        f"Prediction file not found: {RESULTS.resolve()}"
    )

with open(RESULTS, "rb") as f:
    results = pickle.load(f)

original_iou = utils.iou_1d


def standard_temporal_iou(proposal, target):
    """Standard temporal IoU: intersection / union."""

    p = torch.as_tensor(proposal).detach().cpu()
    t = torch.as_tensor(target).detach().cpu()

    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError(f"Unexpected proposal shape: {p.shape}")

    if t.numel() == 0:
        return torch.zeros((len(p), 0), dtype=p.dtype)

    t = t.reshape(-1, 2)

    p_start = p[:, 0:1]
    p_end = p[:, 1:2]
    t_start = t[:, 0].unsqueeze(0)
    t_end = t[:, 1].unsqueeze(0)

    intersection = (
        torch.minimum(p_end, t_end)
        - torch.maximum(p_start, t_start)
    ).clamp(min=0)

    p_length = (p_end - p_start).clamp(min=0)
    t_length = (t_end - t_start).clamp(min=0)

    union = p_length + t_length - intersection

    return intersection / union.clamp(min=1e-8)


def evaluate(iou_function):
    # Ensure both IoU inputs are always on CPU.
    def cpu_iou(proposal, target):
        proposal = torch.as_tensor(proposal).detach().cpu()
        target = torch.as_tensor(target).detach().cpu()
        return iou_function(proposal, target)

    utils.iou_1d = cpu_iou

    # Make a CPU-only copy; preserve original predictions.
    cpu_results = {}

    for key, item in results.items():
        cpu_results[key] = {
            "proposals": torch.as_tensor(
                item["proposals"]
            ).detach().cpu(),
            "labels": torch.as_tensor(
                item["labels"]
            ).detach().cpu(),
        }

    evaluator = AP(
        iou_thresholds=[0.5, 0.75, 0.9, 0.95],
        device="cpu"
    )

    scores = evaluator(cpu_results)

    return {
        float(threshold): float(score)
        for threshold, score in scores.items()
    }


try:
    print("Evaluating with original IoU...")
    old_scores = evaluate(original_iou)

    print("\nEvaluating with standard temporal IoU...")
    new_scores = evaluate(standard_temporal_iou)

finally:
    utils.iou_1d = original_iou


print("\n===== ORIGINAL IoU RESULTS =====")
for threshold, score in old_scores.items():
    print(f"AP@{threshold}: {score:.6f}")

print("\n===== STANDARD TEMPORAL IoU RESULTS =====")
for threshold, score in new_scores.items():
    print(f"AP@{threshold}: {score:.6f}")

print("\nComparison finished.")
print("Original predictions and utils.py were not modified.")
print("Note: AP matching logic remains unchanged.")