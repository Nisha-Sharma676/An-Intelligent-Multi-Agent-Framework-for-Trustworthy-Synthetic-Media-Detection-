import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Project paths
BASE_DIR = Path(__file__).resolve().parent
JSON_PATH = BASE_DIR / "common_temporal_ap_comparison.json"
GRAPH_DIR = BASE_DIR / "project_graphs"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# Load actual comparison data
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

ap_data = data["AP"]
thresholds = list(ap_data.keys())

baseline = [
    ap_data[t]["BMN_baseline"] * 100
    for t in thresholds
]

proposed = [
    ap_data[t]["proposed_localizer"] * 100
    for t in thresholds
]

x = np.arange(len(thresholds))
width = 0.34

# Blue palette
fig, ax = plt.subplots(figsize=(10, 6.5))

bars1 = ax.bar(
    x - width / 2,
    baseline,
    width,
    label="BMN Baseline",
    color="#91CDE8",
    edgecolor="white"
)

bars2 = ax.bar(
    x + width / 2,
    proposed,
    width,
    label="Proposed Localizer",
    color="#145DA0",
    edgecolor="white"
)

# Label each bar with its actual percentage.
# Extra decimal places preserve very small AP values.
def format_ap(value):
    if value < 0.01:
        return f"{value:.6f}%"
    return f"{value:.2f}%"

for bars, values in [(bars1, baseline), (bars2, proposed)]:
    for bar, value in zip(bars, values):
        ax.annotate(
            format_ap(value),
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0
        )

ax.set_title(
    "Temporal Localization Performance on LAV-DF Test Set",
    fontsize=14,
    fontweight="bold",
    pad=15
)

ax.set_xlabel("IoU Threshold", fontsize=11)
ax.set_ylabel("Average Precision (%)", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels([f"AP@{t}" for t in thresholds])

# Leave room for labels above the largest bars
ax.set_ylim(0, max(baseline + proposed) * 1.30)

ax.grid(axis="y", linestyle=":", alpha=0.35)
ax.set_axisbelow(True)

# Legend outside the plot
ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
    frameon=True
)

plt.tight_layout(rect=[0, 0, 0.80, 1])

output_path = GRAPH_DIR / "temporal_localization_ap.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Temporal localization graph created!")
print(f"Videos compared: {data['videos_compared']}")
print(f"Ground-truth segments: {data['ground_truth_segments']}")
print(f"Saved at: {output_path}")

print("\nAP Results:")
for threshold in thresholds:
    baseline_ap = ap_data[threshold]["BMN_baseline"] * 100
    proposed_ap = ap_data[threshold]["proposed_localizer"] * 100

    print(
        f"IoU {threshold} | "
        f"BMN: {baseline_ap:.8f}% | "
        f"Proposed: {proposed_ap:.8f}%"
    )