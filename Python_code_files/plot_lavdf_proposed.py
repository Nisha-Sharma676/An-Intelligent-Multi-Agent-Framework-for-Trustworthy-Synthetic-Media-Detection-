import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    log_loss, brier_score_loss
)

# Load proposed predictions and LAV-DF ground-truth labels
with open(
    r"proposed_model\outputs\proposed_test_results.json",
    encoding="utf-8"
) as f:
    predictions = json.load(f)["predictions"]

with open(r"D:\LAV-DF\LAV-DF\metadata.json", encoding="utf-8") as f:
    metadata = json.load(f)

labels = {
    row["file"]: int(row["n_fakes"] > 0)
    for row in metadata
    if row.get("split") == "test"
}

df = pd.DataFrame(predictions)
df["actual"] = df["file"].map(labels)

if df["actual"].isna().any():
    raise ValueError("Some predictions have no matching ground-truth label.")

y = df["actual"].to_numpy(dtype=int)
p = np.clip(df["fake_probability"].to_numpy(dtype=float), 1e-7, 1 - 1e-7)
pred = (p >= 0.5).astype(int)

# Calculate all 9 classification metrics
accuracy = accuracy_score(y, pred)
precision = precision_score(y, pred, zero_division=0)
recall = recall_score(y, pred, zero_division=0)
f1 = f1_score(y, pred, zero_division=0)
auc = roc_auc_score(y, p)
ap = average_precision_score(y, p)
nll = log_loss(y, p, labels=[0, 1])
brier = brier_score_loss(y, p)

# 10-bin Expected Calibration Error
confidence = np.maximum(p, 1 - p)
correct = (pred == y).astype(float)
bin_ids = np.minimum((confidence * 10).astype(int), 9)
ece = 0.0

for b in range(10):
    mask = bin_ids == b
    if mask.any():
        ece += mask.mean() * abs(
            correct[mask].mean() - confidence[mask].mean()
        )

metrics = [
    "Accuracy", "Fake Precision", "Fake Recall", "Fake F1-score",
    "ROC-AUC", "Average Precision (AP)", "NLL", "Brier Score", "ECE"
]
scores = [accuracy, precision, recall, f1, auc, ap, nll, brier, ece]

print("\nLAV-DF Proposed Model | Test Samples:", len(df))
for name, value in zip(metrics, scores):
    print(f"{name:<25}: {value:.4f}")

# Plot styling
output = Path("project_graphs")
output.mkdir(exist_ok=True)
x = np.arange(len(metrics))
light_blue = "#B8F2F2"
deep_blue = "#073B8C"

# 1. Line plot
plt.figure(figsize=(12, 6))
plt.plot(x, scores, marker="o", linewidth=2.5, color=deep_blue)
plt.xticks(x, metrics, rotation=35, ha="right")
plt.ylim(0, 1)
plt.ylabel("Metric Value")
plt.title("LAV-DF Proposed Model - Classification Metrics")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(output / "lavdf_proposed_line_plot.png", dpi=300)
plt.show()
plt.close()

# 2. Bar chart
plt.figure(figsize=(12, 6))
plt.bar(x, scores, color=light_blue, edgecolor=deep_blue)
plt.xticks(x, metrics, rotation=35, ha="right")
plt.ylim(0, 1)
plt.ylabel("Metric Value")
plt.title("LAV-DF Proposed Model - Classification Metrics")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(output / "lavdf_proposed_bar_chart.png", dpi=300)
plt.show()
plt.close()

print("\nSaved both plots in project_graphs/")