from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "proposed_model" / "outputs"
GRAPH_DIR = ROOT / "project_graphs"
GRAPH_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 12, "axes.labelsize": 10,
    "legend.fontsize": 9, "figure.dpi": 120, "savefig.dpi": 300,
})

def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save(fig, name):
    fig.tight_layout()
    fig.savefig(GRAPH_DIR / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {GRAPH_DIR / name}")

def style(ax):
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def annotate_bars(ax, bars, fmt="{:.2f}"):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(fmt.format(h), (bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

def main():
    fake_proposed = load_json(OUTPUTS / "fakeavceleb_proposed_494_results.json")
    fake_agents = load_json(OUTPUTS / "fakeavceleb_agent_comparison.json")
    lav_original = load_json(OUTPUTS / "original_same_metrics.json")
    lav_reliability = load_json(OUTPUTS / "proposed_reliability_test_metrics.json")
    temporal_path = ROOT / "common_temporal_ap_comparison.json"

    # 1. FakeAVCeleb accuracy: other base metrics were not confirmed.
    vals = [80.16, float(fake_proposed["metrics"]["accuracy_percent"])]
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    bars = ax.bar([0, 1], vals, width=0.58, color=["#A8A8A8", "#555555"],
                  edgecolor="black", linewidth=0.7)
    ax.set_xticks([0, 1], ["Base model", "Proposed Judge Fusion"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("FakeAVCeleb: Classification Accuracy")
    annotate_bars(ax, bars, "{:.2f}%")
    style(ax)
    ax.text(0.01, -0.22,
            "Confirm identical test samples/protocol before claiming direct superiority.",
            transform=ax.transAxes, fontsize=8, va="top", wrap=True)
    save(fig, "01_fakeavceleb_accuracy.png")

    # 2. Available FakeAVCeleb agent/fusion metrics, not base-vs-proposed.
    names = ["V-V Agent", "A-A Agent", "A-V Agent", "Judge Fusion"]
    metrics = [
        ("accuracy_percent", "Accuracy"),
        ("precision_fake_percent", "Fake precision"),
        ("recall_fake_percent", "Fake recall"),
        ("f1_fake_percent", "Fake F1"),
        ("roc_auc_percent", "ROC-AUC"),
        ("average_precision_percent", "Average precision"),
    ]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(metrics))
    width = 0.18
    for j, model in enumerate(names):
        values = [fake_agents["models"][model][key] for key, _ in metrics]
        ax.bar(x + (j - 1.5) * width, values, width, label=model)
    ax.set_xticks(x, [label for _, label in metrics], rotation=15, ha="right")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 110)
    ax.set_title("FakeAVCeleb: Proposed Framework Agent/Fusion Metrics")
    ax.legend(ncol=2, frameon=False)
    style(ax)
    ax.text(0.01, -0.23,
            "Agent/fusion results within the proposed framework; not a base-vs-proposed comparison.",
            transform=ax.transAxes, fontsize=8, va="top", wrap=True)
    save(fig, "02_fakeavceleb_proposed_agent_metrics.png")

    # 3. LAV-DF original fusion vs reliability-aware fusion.
    orig = lav_original["classification"]
    prop = lav_reliability["classification"]
    metric_defs = [
        ("accuracy_threshold_0_5", "Accuracy"),
        ("precision_fake", "Fake precision"),
        ("recall_fake", "Fake recall"),
        ("f1_fake", "Fake F1"),
    ]
    labels = [label for _, label in metric_defs]
    ov = [orig[k] * 100 for k, _ in metric_defs]
    pv = [prop[k] * 100 for k, _ in metric_defs]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x = np.arange(len(labels))
    width = 0.34
    b1 = ax.bar(x-width/2, ov, width, label="Original fusion",
                color="#B7B7B7", edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x+width/2, pv, width, label="Reliability-aware",
                color="#4D4D4D", edgecolor="black", linewidth=0.6)
    ax.set_xticks(x, labels, rotation=10, ha="right")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    ax.set_title("LAV-DF: Classification Performance")
    ax.legend(frameon=False)
    annotate_bars(ax, b1)
    annotate_bars(ax, b2)
    style(ax)
    ax.text(0.01, -0.22,
            "Original fusion is not assumed to be the published base-paper model.",
            transform=ax.transAxes, fontsize=8, va="top", wrap=True)
    save(fig, "03_lavdf_classification.png")

    # 4. Brier score for predicted agent correctness.
    rel_orig = lav_original["agent_reliability"]
    rel_prop = lav_reliability["agent_reliability"]
    agent_keys = ["vv", "aa", "av"]
    agent_labels = ["V-V", "A-A", "A-V"]
    ov = [rel_orig[a]["reliability_brier_score_for_correctness"] for a in agent_keys]
    pv = [rel_prop[a]["reliability_brier_score_for_correctness"] for a in agent_keys]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    x = np.arange(3)
    width = 0.34
    b1 = ax.bar(x-width/2, ov, width, label="Original fusion",
                color="#B7B7B7", edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x+width/2, pv, width, label="Reliability-aware",
                color="#4D4D4D", edgecolor="black", linewidth=0.6)
    ax.set_xticks(x, agent_labels)
    ax.set_ylabel("Brier score (lower is better)")
    ax.set_title("LAV-DF: Agent Reliability Brier Score")
    ax.legend(frameon=False)
    annotate_bars(ax, b1, "{:.3f}")
    annotate_bars(ax, b2, "{:.3f}")
    style(ax)
    ax.text(0.01, -0.20,
            "Scores evaluate predicted agent correctness, not final fake-probability calibration.",
            transform=ax.transAxes, fontsize=8, va="top", wrap=True)
    save(fig, "04_lavdf_agent_reliability_brier.png")

    # 5. Reliability when each agent is correct vs wrong.
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(3)
    width = 0.34
    correct = [rel_prop[a]["mean_reliability_when_correct"] for a in agent_keys]
    wrong = [rel_prop[a]["mean_reliability_when_wrong"] for a in agent_keys]
    b1 = ax.bar(x-width/2, correct, width, label="When correct",
                color="#555555", edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x+width/2, wrong, width, label="When wrong",
                color="#B7B7B7", edgecolor="black", linewidth=0.6)
    ax.set_xticks(x, agent_labels)
    ax.set_ylabel("Mean reliability score")
    ax.set_ylim(0, 1.08)
    ax.set_title("LAV-DF: Reliability by Agent Correctness")
    ax.legend(frameon=False)
    annotate_bars(ax, b1, "{:.3f}")
    annotate_bars(ax, b2, "{:.3f}")
    style(ax)
    save(fig, "05_lavdf_reliability_correct_vs_wrong.png")

    # 6. Uncertainty, coverage, decided-only accuracy.
    od = lav_original["decisions"]
    pd = lav_reliability["decisions"]
    model_labels = ["Original fusion", "Reliability-aware"]
    uncertainty = [od["uncertain_coverage"]*100, pd["uncertain_coverage"]*100]
    coverage = [od["decided_coverage"]*100, pd["decided_coverage"]*100]
    selective = [od["decided_only_accuracy"]*100, pd["decided_only_accuracy"]*100]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    x = np.arange(2)
    width = 0.34
    b1 = axes[0].bar(x-width/2, coverage, width, label="Decided coverage",
                     color="#555555", edgecolor="black", linewidth=0.6)
    b2 = axes[0].bar(x+width/2, uncertainty, width, label="Uncertainty rate",
                     color="#B7B7B7", edgecolor="black", linewidth=0.6)
    axes[0].set_xticks(x, model_labels, rotation=10, ha="right")
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Percentage (%)")
    axes[0].set_title("Coverage and Uncertainty")
    axes[0].legend(frameon=False, fontsize=8)
    annotate_bars(axes[0], b1)
    annotate_bars(axes[0], b2)
    style(axes[0])
    bars = axes[1].bar(x, selective, width=0.55,
                       color=["#B7B7B7", "#555555"],
                       edgecolor="black", linewidth=0.6)
    axes[1].set_xticks(x, model_labels, rotation=10, ha="right")
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Decided-only Accuracy")
    annotate_bars(axes[1], bars)
    style(axes[1])
    fig.suptitle("LAV-DF: Uncertainty-aware Decision Quality", y=1.02)
    save(fig, "06_lavdf_uncertainty_coverage.png")

    # 7. Temporal AP with explicit limitation.
    if temporal_path.exists():
        temporal = load_json(temporal_path)
        thresholds = ["0.5", "0.75", "0.95"]
        bmn = [temporal["AP"][t]["BMN_baseline"]*100 for t in thresholds]
        proposed = [temporal["AP"][t]["proposed_localizer"]*100 for t in thresholds]
        fig, ax = plt.subplots(figsize=(7.8, 4.6))
        x = np.arange(3)
        width = 0.34
        b1 = ax.bar(x-width/2, bmn, width, label="BMN baseline",
                    color="#B7B7B7", edgecolor="black", linewidth=0.6)
        b2 = ax.bar(x+width/2, proposed, width, label="Proposed localizer",
                    color="#4D4D4D", edgecolor="black", linewidth=0.6)
        ax.set_xticks(x, [f"AP@{t}" for t in thresholds])
        ax.set_ylabel("Average Precision (%)")
        ax.set_title("LAV-DF: Temporal Localization")
        ax.legend(frameon=False)
        annotate_bars(ax, b1, "{:.4f}")
        annotate_bars(ax, b2, "{:.4f}")
        style(ax)
        ax.text(0.01, -0.23,
                "Proposal counts differ; BMN AP@0.5/AP@0.75 mismatch remains unresolved.",
                transform=ax.transAxes, fontsize=8, va="top", wrap=True)
        save(fig, "07_lavdf_temporal_ap.png")

    print("\nGraphs saved in:", GRAPH_DIR)
    print("ECE, final-probability Brier and NLL are not fabricated; they need per-sample probabilities and labels.")

if __name__ == "__main__":
    main()
