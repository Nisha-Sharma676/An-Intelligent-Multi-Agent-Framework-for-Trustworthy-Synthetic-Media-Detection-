import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss,
    brier_score_loss
)

# ---------- Base model: FakeAVCeleb ----------
base = pd.read_csv("results/base_model/base_predictions.csv")

y_base = (
    base["Actual"].astype(str).str.strip().str.lower() == "fake"
).astype(int).to_numpy()

p_base = np.clip(
    1 - base["Prediction_Score"].to_numpy(dtype=float),
    1e-7, 1 - 1e-7
)
pred_base = (p_base >= 0.5).astype(int)

# ---------- Proposed model: LAV-DF ----------
with open(
    "proposed_model/outputs/proposed_test_results.json",
    encoding="utf-8"
) as f:
    proposed = json.load(f)["predictions"]

with open(r"D:\LAV-DF\LAV-DF\metadata.json", encoding="utf-8") as f:
    metadata = json.load(f)

labels = {
    r["file"]: int(r["n_fakes"] > 0)
    for r in metadata if r.get("split") == "test"
}

df = pd.DataFrame(proposed)
df["actual"] = df["file"].map(labels)

if df["actual"].isna().any():
    raise ValueError("Missing LAV-DF ground-truth labels.")

y_prop = df["actual"].to_numpy(dtype=int)
p_prop = np.clip(
    df["fake_probability"].to_numpy(dtype=float),
    1e-7, 1 - 1e-7
)
pred_prop = (p_prop >= 0.5).astype(int)


def calculate_metrics(y, p):
    pred = (p >= 0.5).astype(int)
    values = [
        accuracy_score(y, pred),
        precision_score(y, pred, zero_division=0),
        recall_score(y, pred, zero_division=0),
        f1_score(y, pred, zero_division=0),
        roc_auc_score(y, p),
        average_precision_score(y, p),
        log_loss(y, p, labels=[0, 1]),
        brier_score_loss(y, p)
    ]

    # 10-bin ECE based on confidence and correctness
    confidence = np.maximum(p, 1 - p)
    correct = (pred == y).astype(float)
    bins = np.minimum((confidence * 10).astype(int), 9)
    ece = 0.0

    for b in range(10):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(
                correct[mask].mean() - confidence[mask].mean()
            )

    return values + [ece]


metrics = [
    "Accuracy", "Fake Precision", "Fake Recall", "Fake F1-score",
    "ROC-AUC", "Average Precision (AP)", "NLL", "Brier Score", "ECE"
]

base_scores = calculate_metrics(y_base, p_base)
proposed_scores = calculate_metrics(y_prop, p_prop)

print("\nBase: FakeAVCeleb | Proposed: LAV-DF")
print("Base samples:", len(y_base), "| Proposed samples:", len(y_prop))

for name, b, p in zip(metrics, base_scores, proposed_scores):
    print(f"{name:<25} Base: {b:.4f} | Proposed: {p:.4f}")

# ---------- Plots ----------
out = Path("project_graphs")
out.mkdir(exist_ok=True)

x = np.arange(len(metrics))
base_color = "#B8F2F2"
proposed_color = "#073B8C"

# Line plot
plt.figure(figsize=(12, 6))
plt.plot(x, base_scores, marker="o", linewidth=2.5,
         color=base_color, label="Base - FakeAVCeleb")
plt.plot(x, proposed_scores, marker="o", linewidth=2.5,
         color=proposed_color, label="Proposed - LAV-DF")
plt.xticks(x, metrics, rotation=35, ha="right")
plt.ylim(0, 1)
plt.ylabel("Metric Value")
plt.title("Cross-Dataset Classification Metrics")
plt.grid(axis="y", alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(out / "cross_dataset_line_plot.png", dpi=300)
plt.show()
plt.close()

# Grouped bar chart
width = 0.36
plt.figure(figsize=(12, 6))
plt.bar(x - width / 2, base_scores, width,
        color=base_color, label="Base - FakeAVCeleb")
plt.bar(x + width / 2, proposed_scores, width,
        color=proposed_color, label="Proposed - LAV-DF")
plt.xticks(x, metrics, rotation=35, ha="right")
plt.ylim(0, 1)
plt.ylabel("Metric Value")
plt.title("Cross-Dataset Classification Metrics")
plt.grid(axis="y", alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(out / "cross_dataset_bar_chart.png", dpi=300)
plt.show()
plt.close()

print("\nSaved in project_graphs/")