from pathlib import Path
import json
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "proposed_model" / "outputs"
GRAPH_DIR = ROOT / "project_graphs"
GRAPH_DIR.mkdir(exist_ok=True)

# Clean, research-paper-style formatting
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 120,
    "savefig.dpi": 300,
})


def load_json(filename):
    with (OUT / filename).open("r", encoding="utf-8") as f:
        return json.load(f)


def finish_plot(filename):
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / filename, bbox_inches="tight")
    plt.show()
    plt.close()
    print("Saved:", GRAPH_DIR / filename)


# ==================================================
# GRAPH 1: FakeAVCeleb Accuracy
# ==================================================
fake = load_json("fakeavceleb_proposed_494_results.json")

models = ["Base Model", "Proposed Judge Fusion"]
accuracy = [
    80.16,
    fake["metrics"]["accuracy_percent"]
]

fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(
    models, accuracy,
    color="black",
    marker="o",
    markersize=7,
    linewidth=1.8,
    markerfacecolor="white",
    markeredgewidth=1.8
)

for i, value in enumerate(accuracy):
    ax.annotate(
        f"{value:.2f}%",
        (i, value),
        xytext=(0, 9),
        textcoords="offset points",
        ha="center"
    )

ax.set_ylabel("Accuracy (%)")
ax.set_title("FakeAVCeleb Classification Accuracy")
ax.set_ylim(0, 100)
ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

finish_plot("01_fakeavceleb_accuracy_line.png")


# ==================================================
# GRAPH 2: LAV-DF Classification Metrics
# ==================================================
original = load_json("original_same_metrics.json")
proposed = load_json("proposed_reliability_test_metrics.json")

metric_names = ["Accuracy", "Precision", "Recall", "F1-score"]

original_values = [
    original["classification"]["accuracy_threshold_0_5"] * 100,
    original["classification"]["precision_fake"] * 100,
    original["classification"]["recall_fake"] * 100,
    original["classification"]["f1_fake"] * 100
]

proposed_values = [
    proposed["classification"]["accuracy_threshold_0_5"] * 100,
    proposed["classification"]["precision_fake"] * 100,
    proposed["classification"]["recall_fake"] * 100,
    proposed["classification"]["f1_fake"] * 100
]

fig, ax = plt.subplots(figsize=(8, 4.8))

ax.plot(
    metric_names, original_values,
    color="black",
    marker="o",
    markersize=6,
    linewidth=1.8,
    label="Original Fusion"
)

ax.plot(
    metric_names, proposed_values,
    color="dimgray",
    linestyle="--",
    marker="s",
    markerfacecolor="white",
    markersize=6,
    linewidth=1.8,
    label="Reliability-aware Fusion"
)

ax.set_ylabel("Score (%)")
ax.set_title("LAV-DF Classification Performance")
ax.set_ylim(0, 100)
ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
ax.legend(frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

finish_plot("02_lavdf_classification_line.png")

print("\nBoth research-style line graphs generated.")