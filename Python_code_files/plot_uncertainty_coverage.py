import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Project paths
BASE_DIR = Path(__file__).resolve().parent
JSON_PATH = BASE_DIR / "proposed_model" / "outputs" / "calibration_audit_dev.json"
GRAPH_DIR = BASE_DIR / "project_graphs"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# Load existing development-set results
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

threshold_data = data["decision_threshold_candidates"]

thresholds = []
coverage = []
uncertainty = []
accuracy = []

for name, values in threshold_data.items():
    thresholds.append(name.replace("_", " / "))
    coverage.append(values["decided_coverage"] * 100)
    uncertainty.append(values["uncertain_coverage"] * 100)
    accuracy.append(values["decided_only_accuracy"] * 100)

x = np.arange(len(thresholds))
width = 0.24

# Blue palette
colors = {
    "coverage": "#B9E3F5",
    "uncertainty": "#4A90C4",
    "accuracy": "#145DA0",
}

fig, ax = plt.subplots(figsize=(10, 6.5))

ax.bar(
    x - width, coverage, width,
    label="Decided Coverage",
    color=colors["coverage"],
    edgecolor="#FFFFFF"
)

ax.bar(
    x, uncertainty, width,
    label="Uncertain Coverage",
    color=colors["uncertainty"],
    edgecolor="#FFFFFF"
)

ax.bar(
    x + width, accuracy, width,
    label="Accuracy on Decided Samples",
    color=colors["accuracy"],
    edgecolor="#FFFFFF"
)

# Display percentage labels
for i in range(len(x)):
    ax.text(
        x[i] - width, coverage[i] + 1,
        f"{coverage[i]:.1f}%",
        ha="center", fontsize=8
    )
    ax.text(
        x[i], uncertainty[i] + 1,
        f"{uncertainty[i]:.1f}%",
        ha="center", fontsize=8
    )
    ax.text(
        x[i] + width, accuracy[i] + 1,
        f"{accuracy[i]:.1f}%",
        ha="center", fontsize=8
    )

ax.set_title(
    "Uncertainty vs Coverage and Accuracy",
    fontsize=14,
    fontweight="bold",
    pad=14
)

ax.set_xlabel("Decision Threshold (REAL / FAKE)")
ax.set_ylabel("Percentage (%)")
ax.set_xticks(x)
ax.set_xticklabels(thresholds)
ax.set_ylim(0, 110)

ax.grid(axis="y", linestyle=":", alpha=0.35)
ax.set_axisbelow(True)

# Legend outside the plot
ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
    frameon=True,
    fontsize=9
)

plt.tight_layout(rect=[0, 0, 0.76, 1])

output_path = GRAPH_DIR / "uncertainty_coverage_tradeoff.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print("Uncertainty vs Coverage plot created successfully!")
print(f"Saved at: {output_path}")

print("\nDevelopment-set results:")
for i, threshold in enumerate(thresholds):
    print(
        f"{threshold} | "
        f"Coverage: {coverage[i]:.2f}% | "
        f"Uncertain: {uncertainty[i]:.2f}% | "
        f"Decided-only accuracy: {accuracy[i]:.2f}%"
    )