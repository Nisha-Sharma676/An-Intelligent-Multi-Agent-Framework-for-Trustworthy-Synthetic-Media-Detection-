import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Project paths
BASE_DIR = Path(__file__).resolve().parent
JSON_PATH = (
    BASE_DIR
    / "proposed_model"
    / "outputs"
    / "fakeavceleb_agent_comparison.json"
)
GRAPH_DIR = BASE_DIR / "project_graphs"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# Load FakeAVCeleb comparison results
with open(JSON_PATH, "r", encoding="utf-8") as file:
    data = json.load(file)

models = data["models"]

# Metrics to compare
metric_keys = [
    "accuracy_percent",
    "f1_fake_percent",
    "roc_auc_percent",
    "average_precision_percent",
]

metric_labels = [
    "Accuracy",
    "F1-score",
    "ROC-AUC",
    "Average Precision",
]

model_names = list(models.keys())

# Extract metric values
values = {
    metric: [models[name][metric] for name in model_names]
    for metric in metric_keys
}

# Blue palette
colors = ["#D6EEF5", "#91CDE8", "#4A90C4", "#145DA0"]

x = np.arange(len(metric_labels))
width = 0.19

fig, ax = plt.subplots(figsize=(11, 6.5))

# Draw grouped bars
for i, model_name in enumerate(model_names):
    model_values = [values[key][i] for key in metric_keys]

    positions = x + (i - (len(model_names) - 1) / 2) * width

    bars = ax.bar(
        positions,
        model_values,
        width=width,
        color=colors[i],
        edgecolor="white",
        linewidth=0.7,
        label=model_name,
    )

    # Percentage labels
    ax.bar_label(
        bars,
        fmt="%.1f",
        padding=2,
        fontsize=7,
        rotation=90,
    )

# Formatting
ax.set_title(
    f"FakeAVCeleb Agent Comparison ({data['samples']} Samples)",
    fontsize=14,
    fontweight="bold",
    pad=15,
)

ax.set_xlabel("Evaluation Metrics", fontsize=11)
ax.set_ylabel("Score (%)", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(metric_labels)
ax.set_ylim(0, 115)

ax.grid(axis="y", linestyle=":", alpha=0.35)
ax.set_axisbelow(True)

# Legend outside the plot
ax.legend(
    title="Models",
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
    frameon=True,
)

plt.tight_layout(rect=[0, 0, 0.80, 1])

output_path = GRAPH_DIR / "fakeavceleb_agent_comparison.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("FakeAVCeleb Agent Comparison graph created!")
print(f"Samples: {data['samples']}")
print(f"Saved at: {output_path}")

print("\nMetric Summary:")
for name in model_names:
    result = models[name]
    print(
        f"{name:15s} | "
        f"Accuracy: {result['accuracy_percent']:.2f}% | "
        f"F1: {result['f1_fake_percent']:.2f}% | "
        f"ROC-AUC: {result['roc_auc_percent']:.2f}% | "
        f"AP: {result['average_precision_percent']:.2f}%"
    )