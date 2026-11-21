import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Project folders
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "proposed_model" / "outputs"
GRAPH_DIR = BASE_DIR / "project_graphs"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# Load calibration audit results
json_path = OUTPUT_DIR / "calibration_audit_dev.json"

with open(json_path, "r", encoding="utf-8") as file:
    data = json.load(file)

agents = data["agent_calibration"]

# Blue color palette
colors = {
    "vv": "#B9E3F5",
    "aa": "#4A90C4",
    "av": "#145DA0",
}

agent_names = {
    "vv": "V-V Agent",
    "aa": "A-A Agent",
    "av": "A-V Agent",
}

# Create figure
fig, ax = plt.subplots(figsize=(10, 7))

# Perfect calibration reference line
ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="#555555",
    linewidth=1.8,
    label="Perfect Calibration",
)

# Plot each agent's available calibration bins
for agent_key, agent_data in agents.items():

    bins = agent_data["reliability"]["bins"]

    predicted = [
        item["mean_predicted_reliability"]
        for item in bins
    ]

    observed = [
        item["observed_accuracy"]
        for item in bins
    ]

    counts = [
        item["count"]
        for item in bins
    ]

    # Marker sizes represent number of samples in each bin
    sizes = [
        45 + 0.25 * count
        for count in counts
    ]

    ax.scatter(
        predicted,
        observed,
        s=sizes,
        color=colors[agent_key],
        edgecolors="#17365D",
        linewidths=0.8,
        alpha=0.9,
        label=agent_names[agent_key],
        zorder=3,
    )

    # Connect points belonging to the same agent
    order = np.argsort(predicted)

    ax.plot(
        np.array(predicted)[order],
        np.array(observed)[order],
        color=colors[agent_key],
        linewidth=1.8,
        alpha=0.85,
    )

# Formatting
ax.set_title(
    "Reliability Calibration of Multi-Agent Deepfake Detection",
    fontsize=14,
    fontweight="bold",
    pad=15,
)

ax.set_xlabel("Mean Predicted Reliability", fontsize=11)
ax.set_ylabel("Observed Agent Accuracy", fontsize=11)

ax.set_xlim(0.45, 1.0)
ax.set_ylim(0.25, 1.0)

ax.grid(True, linestyle=":", alpha=0.35)

# Legend outside the plot
ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
    frameon=True,
    fontsize=9,
)

# Add ECE scores in a separate text panel
ece_text = "ECE (lower is better)\n"

for agent_key, agent_data in agents.items():
    ece = agent_data["reliability"]["ece_10_bins"]

    ece_text += (
        f"{agent_names[agent_key]}: {ece:.4f}\n"
    )

fig.text(
    0.79,
    0.43,
    ece_text,
    fontsize=10,
    va="top",
    bbox=dict(
        boxstyle="round,pad=0.5",
        facecolor="#EAF4FB",
        edgecolor="#4A90C4",
    ),
)

plt.tight_layout(rect=[0, 0, 0.77, 1])

# Save graph
output_file = GRAPH_DIR / "reliability_calibration_plot.png"

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("Reliability calibration graph created successfully!")
print(f"Saved at: {output_file}")

print("\nAgent Calibration Summary:")
for agent_key, agent_data in agents.items():
    reliability = agent_data["reliability"]

    print(
        f"{agent_names[agent_key]} | "
        f"Accuracy: {agent_data['agent_accuracy']:.4f} | "
        f"ECE: {reliability['ece_10_bins']:.4f} | "
        f"Brier Score: {reliability['brier_score']:.4f}"
    )