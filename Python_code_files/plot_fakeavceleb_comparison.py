import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss
)

# Load base and proposed predictions
base = pd.read_csv("results/base_model/base_predictions.csv")

with open(
    "proposed_model/outputs/fakeavceleb_proposed_494_results.json",
    encoding="utf-8"
) as f:
    proposed = json.load(f)["predictions"]

# Match predictions using exact feature filenames
base["match_key"] = base["Filepath"].apply(
    lambda x: Path(str(x).replace("\\", "/")).name.lower()
)

proposed_df = pd.DataFrame(proposed)
proposed_df["match_key"] = proposed_df["feature_file"].apply(
    lambda x: Path(str(x).replace("\\", "/")).name.lower()
)

if base["match_key"].duplicated().any():
    raise ValueError("Duplicate filenames in base predictions.")

if proposed_df["match_key"].duplicated().any():
    raise ValueError("Duplicate filenames in proposed predictions.")

merged = base.merge(
    proposed_df,
    on="match_key",
    how="inner",
    suffixes=("_base", "_proposed"),
    validate="one_to_one"
)

print("Matched samples:", len(merged))

if len(merged) != 494:
    raise ValueError(
        f"Expected 494 matched samples, got {len(merged)}. "
        "Check filenames before comparing."
    )

# Verify actual labels match
y_base = (
    merged["Actual"].astype(str).str.strip().str.lower() == "fake"
).astype(int).to_numpy()

y_proposed = (
    merged["actual"].astype(str).str.strip().str.lower() == "fake"
).astype(int).to_numpy()

if not np.array_equal(y_base, y_proposed):
    raise ValueError("Actual labels differ between the two models.")

# Probabilities of the Fake class
p_base = 1 - merged["Prediction_Score"].to_numpy(dtype=float)
p_proposed = merged["fake_probability"].to_numpy(dtype=float)


def calculate_metrics(y, probabilities):
    p = np.clip(probabilities, 1e-7, 1 - 1e-7)
    predictions = (p >= 0.5).astype(int)

    # Expected Calibration Error using 10 confidence bins
    confidence = np.maximum(p, 1 - p)
    correct = (predictions == y).astype(float)
    bins = np.minimum((confidence * 10).astype(int), 9)
    ece = 0.0

    for b in range(10):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(
                correct[mask].mean() - confidence[mask].mean()
            )

    return [
        accuracy_score(y, predictions),
        precision_score(y, predictions, zero_division=0),
        recall_score(y, predictions, zero_division=0),
        f1_score(y, predictions, zero_division=0),
        roc_auc_score(y, p),
        average_precision_score(y, p),
        log_loss(y, p),
        brier_score_loss(y, p),
        ece
    ]


metrics = [
    "Accuracy",
    "Fake Precision",
    "Fake Recall",
    "Fake F1-score",
    "ROC-AUC",
    "Average Precision (AP)",
    "NLL",
    "Brier Score",
    "ECE"
]

base_scores = calculate_metrics(y_base, p_base)
proposed_scores = calculate_metrics(y_proposed, p_proposed)

print("\nFakeAVCeleb: Base vs Proposed Judge Fusion")
print("-" * 55)

for name, base_value, proposed_value in zip(
    metrics, base_scores, proposed_scores
):
    print(
        f"{name:<25} Base: {base_value:.4f} | "
        f"Proposed: {proposed_value:.4f}"
    )

# Plot settings: light cyan to deep blue
colors = ["#B8F2F2", "#073B8C"]
x = np.arange(len(metrics))
output = Path("project_graphs")
output.mkdir(exist_ok=True)

# 1. Line plot
plt.figure(figsize=(12, 6))

plt.plot(
    x, base_scores,
    marker="o", linewidth=2.5,
    color=colors[0], label="Base Model"
)

plt.plot(
    x, proposed_scores,
    marker="o", linewidth=2.5,
    color=colors[1], label="Proposed Judge Fusion"
)

plt.xticks(x, metrics, rotation=35, ha="right")
plt.ylim(0, 1)
plt.ylabel("Metric Value")
plt.title("FakeAVCeleb: Base vs Proposed Model")
plt.grid(axis="y", alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(output / "fakeavceleb_line_plot.png", dpi=300)
plt.show()
plt.close()

# 2. Grouped bar chart
width = 0.36

plt.figure(figsize=(12, 6))

plt.bar(
    x - width / 2, base_scores, width,
    color=colors[0], label="Base Model"
)

plt.bar(
    x + width / 2, proposed_scores, width,
    color=colors[1], label="Proposed Judge Fusion"
)

plt.xticks(x, metrics, rotation=35, ha="right")
plt.ylim(0, 1)
plt.ylabel("Metric Value")
plt.title("FakeAVCeleb: Metric Comparison")
plt.grid(axis="y", alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(output / "fakeavceleb_bar_chart.png", dpi=300)
plt.show()
plt.close()

print("\nPlots saved in project_graphs/")