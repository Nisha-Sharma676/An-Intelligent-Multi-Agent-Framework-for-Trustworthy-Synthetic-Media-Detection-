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
    balanced_accuracy_score
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent

JSON_PATH = (
    BASE_DIR
    / "proposed_model"
    / "outputs"
    / "fakeavceleb_proposed_494_results.json"
)

GRAPH_DIR = BASE_DIR / "project_graphs"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# Load predictions
with open(JSON_PATH, "r", encoding="utf-8") as file:
    data = json.load(file)

# Handle either a list or a dictionary containing results
if isinstance(data, dict):
    for key in ["results", "predictions", "samples", "data"]:
        if isinstance(data.get(key), list):
            data = data[key]
            break

if not isinstance(data, list) or len(data) == 0:
    raise ValueError("No prediction records found in the JSON file.")

df = pd.DataFrame(data)

required_columns = {"actual", "fake_probability"}
missing = required_columns - set(df.columns)

if missing:
    raise ValueError(
        f"Missing columns: {missing}. "
        f"Available columns: {list(df.columns)}"
    )

# Convert ground-truth labels to binary values:
# Fake = 1, Real = 0
def convert_label(value):
    if isinstance(value, (int, float, np.integer, np.floating)):
        if value in (0, 1):
            return int(value)

    label = str(value).strip().lower()

    if label in {"fake", "1", "true", "yes"}:
        return 1
    if label in {"real", "0", "false", "no"}:
        return 0

    raise ValueError(f"Unrecognized actual label: {value}")

y_true = np.array([convert_label(x) for x in df["actual"]])
y_score = pd.to_numeric(
    df["fake_probability"], errors="raise"
).to_numpy()

if not np.isfinite(y_score).all():
    raise ValueError("fake_probability contains non-finite values.")

if ((y_score < 0) | (y_score > 1)).any():
    raise ValueError("fake_probability must be between 0 and 1.")

if len(np.unique(y_true)) < 2:
    raise ValueError("Both Real and Fake labels are needed.")

# Evaluate each decision threshold
thresholds = np.round(np.arange(0.10, 0.91, 0.05), 2)
results = []

for threshold in thresholds:
    # Probability >= threshold means predicted Fake
    y_pred = (y_score >= threshold).astype(int)

    results.append({
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1_score": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        )
    })

results_df = pd.DataFrame(results)

# Save metric values for research and reporting
csv_path = GRAPH_DIR / "decision_threshold_metrics.csv"
results_df.to_csv(csv_path, index=False)

# Plot all five metrics
fig, ax = plt.subplots(figsize=(11, 6.5))

metric_styles = [
    ("accuracy", "Accuracy", "#145DA0", "o"),
    ("precision", "Precision", "#2E86C1", "s"),
    ("recall", "Recall", "#73BFE2", "^"),
    ("f1_score", "F1-score", "#1B998B", "D"),
    ("balanced_accuracy", "Balanced Accuracy", "#6C5CE7", "v")
]

for column, label, color, marker in metric_styles:
    ax.plot(
        results_df["threshold"],
        results_df[column] * 100,
        label=label,
        color=color,
        marker=marker,
        linewidth=2,
        markersize=5
    )

ax.set_title(
    "Decision Threshold Analysis - FakeAVCeleb",
    fontsize=14,
    fontweight="bold",
    pad=14
)
ax.set_xlabel("Fake Probability Threshold", fontsize=11)
ax.set_ylabel("Metric Score (%)", fontsize=11)

ax.set_xticks(thresholds)
ax.set_xticklabels(
    [f"{t:.2f}" for t in thresholds],
    rotation=45
)

ax.set_ylim(0, 105)
ax.grid(axis="y", linestyle=":", alpha=0.35)
ax.set_axisbelow(True)

# Keep legend outside the plotting area
ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
    frameon=True
)

plt.tight_layout(rect=[0, 0, 0.79, 1])

output_path = GRAPH_DIR / "decision_threshold_analysis.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nDecision Threshold Analysis completed!")
print(f"Samples evaluated: {len(y_true)}")
print(f"Real samples: {int((y_true == 0).sum())}")
print(f"Fake samples: {int((y_true == 1).sum())}")
print(f"Graph saved at: {output_path}")
print(f"Metrics CSV saved at: {csv_path}")

print("\nMetrics by threshold:")
print(results_df.round(4).to_string(index=False))