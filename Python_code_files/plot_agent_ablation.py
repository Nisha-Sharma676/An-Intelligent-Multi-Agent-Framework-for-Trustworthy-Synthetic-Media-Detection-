import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 1. Project paths
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "proposed_model" / "outputs" / "fusion_comparison.json"
OUTPUT_DIR = BASE_DIR / "project_graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Read JSON file
with open(INPUT_FILE, "r", encoding="utf-8") as file:
    data = json.load(file)

# Agent metrics are inside the "results" key
agent_results = data["results"]

# 3. Metrics to compare
metrics = {
    "Accuracy": "accuracy",
    "F1-score": "f1",
    "ROC-AUC": "roc_auc",
    "Average Precision": "average_precision",
}

# 4. Agent names
agents = [
    "V-V Agent",
    "A-A Agent",
    "A-V Agent",
    "Equal Average",
    "Judge Fusion",
]

# 5. Blue colour palette: light to dark
blue_palette = [
    "#D6EEF5",  # Light blue
    "#79C8E8",  # Sky blue
    "#4A90C4",  # Medium blue
    "#1769AA",  # Dark blue
]

# 6. Extract metric values
results = {}

for agent in agents:
    if agent not in agent_results:
        raise KeyError(
            f"Agent '{agent}' not found. "
            f"Available agents: {list(agent_results.keys())}"
        )

    results[agent] = {}

    for metric_name, metric_key in metrics.items():
        results[agent][metric_name] = agent_results[agent][metric_key]

metric_names = list(metrics.keys())

# ==========================================
# 7. GROUPED BAR CHART
# ==========================================
fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(len(agents))
bar_width = 0.18

for i, metric in enumerate(metric_names):
    values = [results[agent][metric] for agent in agents]
    positions = x + (i - 1.5) * bar_width

    ax.bar(
        positions,
        values,
        width=bar_width,
        label=metric,
        color=blue_palette[i],
        edgecolor="white",
        linewidth=0.5
    )

ax.set_title(
    "LAV-DF Test: Agent Ablation Comparison",
    fontsize=15,
    pad=15
)
ax.set_xlabel("Agent / Fusion Method", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(agents, rotation=15, ha="right")
ax.set_ylim(0, 1.05)
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.set_axisbelow(True)

# Legend outside the graph on the right
ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.03, 1),
    borderaxespad=0,
    frameon=True
)

# Reserve space for the outside legend
fig.subplots_adjust(right=0.76, bottom=0.20)

bar_path = OUTPUT_DIR / "agent_ablation_bar_chart.png"
fig.savefig(bar_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)

# ==========================================
# 8. LINE CHART
# ==========================================
fig, ax = plt.subplots(figsize=(14, 7))

for i, metric in enumerate(metric_names):
    values = [results[agent][metric] for agent in agents]

    ax.plot(
        agents,
        values,
        marker="o",
        linewidth=2.5,
        markersize=6,
        label=metric,
        color=blue_palette[i]
    )

ax.set_title(
    "LAV-DF Test: Agent Ablation Comparison",
    fontsize=15,
    pad=15
)
ax.set_xlabel("Agent / Fusion Method", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_ylim(0, 1.05)
ax.grid(True, linestyle="--", alpha=0.3)
ax.set_axisbelow(True)

# Legend outside the graph on the right
ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.03, 1),
    borderaxespad=0,
    frameon=True
)

fig.subplots_adjust(right=0.76, bottom=0.20)

line_path = OUTPUT_DIR / "agent_ablation_line_plot.png"
fig.savefig(line_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)

print("\nBoth graphs generated successfully!")
print(f"Bar chart saved at: {bar_path}")
print(f"Line chart saved at: {line_path}")