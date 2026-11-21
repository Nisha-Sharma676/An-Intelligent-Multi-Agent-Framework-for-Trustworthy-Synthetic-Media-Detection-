import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Project paths
BASE_DIR = Path(__file__).resolve().parent

JSON_PATH = (
    BASE_DIR
    / "proposed_model"
    / "outputs"
    / "calibration_audit_dev.json"
)

GRAPH_DIR = BASE_DIR / "project_graphs"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# Load calibration data
with open(JSON_PATH, "r", encoding="utf-8") as file:
    data = json.load(file)

agent_data = data["agent_calibration"]

# Map short JSON names to readable agent names
agent_names = {
    "vv": "V-V",
    "aa": "A-A",
    "av": "A-V"
}

rows = []

for key, name in agent_names.items():
    agent = agent_data[key]
    reliability = agent["reliability"]

    # Calculate mean predicted reliability weighted by bin counts
    bins = reliability.get("bins", [])
    total_count = sum(item["count"] for item in bins)

    if total_count > 0:
        mean_reliability = sum(
            item["mean_predicted_reliability"] * item["count"]
            for item in bins
        ) / total_count
    else:
        mean_reliability = None

    rows.append({
        "Agent": name,
        "Agent Accuracy": agent["agent_accuracy"],
        "Mean Reliability": mean_reliability,
        "Brier Score": reliability["brier_score"],
        "ECE": reliability["ece_10_bins"]
    })

df = pd.DataFrame(rows)

# Save the metric table
csv_path = GRAPH_DIR / "agent_reliability_metrics.csv"
df.to_csv(csv_path, index=False)

# Draw three panels because the metrics have different meanings/scales
fig, axes = plt.subplots(1, 3, figsize=(13, 5.5))

colors = ["#91CDE8", "#145DA0", "#2E86C1"]

plots = [
    ("Mean Reliability", "Mean Predicted Reliability", True),
    ("Brier Score", "Brier Score (lower is better)", False),
    ("ECE", "ECE (lower is better)", False)
]

for ax, (column, title, percentage) in zip(axes, plots):
    values = df[column].astype(float).to_numpy()

    if percentage:
        values = values * 100

    bars = ax.bar(
        df["Agent"],
        values,
        color=colors,
        edgecolor="white",
        width=0.65
    )

    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.set_axisbelow(True)

    highest = max(values) if len(values) else 1
    ax.set_ylim(0, highest * 1.25 if highest > 0 else 1)

    for bar, value in zip(bars, values):
        label = f"{value:.2f}%" if percentage else f"{value:.4f}"

        ax.annotate(
            label,
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=9
        )

fig.suptitle(
    f"Agent Reliability Comparison ({data['split'].upper()} Set, "
    f"{data['samples']} Samples)",
    fontsize=14,
    fontweight="bold",
    y=1.04
)

plt.tight_layout()

output_path = GRAPH_DIR / "agent_reliability_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nAgent reliability comparison graph created!")
print(f"Split: {data['split']}")
print(f"Samples: {data['samples']}")
print(f"Graph saved at: {output_path}")
print(f"CSV saved at: {csv_path}")
print("\nMetrics:")
print(df.round(4).to_string(index=False))