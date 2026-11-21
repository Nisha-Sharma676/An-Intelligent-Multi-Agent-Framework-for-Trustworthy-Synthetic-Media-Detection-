import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score
)

# Base model: FakeAVCeleb
base = pd.read_csv("results/base_model/base_predictions.csv")

y_base = (
    base["Actual"].astype(str).str.strip().str.lower() == "fake"
).astype(int).to_numpy()

p_base = np.clip(
    1 - base["Prediction_Score"].to_numpy(dtype=float),
    1e-7, 1 - 1e-7
)
pred_base = (p_base >= 0.5).astype(int)

# Proposed model: LAV-DF
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
    raise ValueError("Some LAV-DF labels are missing.")

y_prop = df["actual"].to_numpy(dtype=int)
p_prop = np.clip(
    df["fake_probability"].to_numpy(dtype=float),
    1e-7, 1 - 1e-7
)
pred_prop = (p_prop >= 0.5).astype(int)

# Four metrics
metrics = ["Accuracy", "ROC-AUC", "Fake Recall", "Fake Precision"]

base_scores = [
    accuracy_score(y_base, pred_base),
    roc_auc_score(y_base, p_base),
    recall_score(y_base, pred_base, zero_division=0),
    precision_score(y_base, pred_base, zero_division=0)
]

proposed_scores = [
    accuracy_score(y_prop, pred_prop),
    roc_auc_score(y_prop, p_prop),
    recall_score(y_prop, pred_prop, zero_division=0),
    precision_score(y_prop, pred_prop, zero_division=0)
]

print("\nBase: FakeAVCeleb | Proposed: LAV-DF")
for name, b, p in zip(metrics, base_scores, proposed_scores):
    print(f"{name}: Base={b:.4f}, Proposed={p:.4f}")

# One combined figure with four separate panels
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
x = [0, 1]
labels_x = ["Base\nFakeAVCeleb", "Proposed\nLAV-DF"]
colors = ["#B8F2F2", "#073B8C"]

for ax, metric, b, p in zip(
    axes.flat, metrics, base_scores, proposed_scores
):
    ax.plot(x, [b, p], marker="o", linewidth=2.5,
            markersize=8, color=colors[1])

    ax.scatter(x, [b, p], s=90, color=colors)
    ax.set_xticks(x, labels_x)
    ax.set_ylim(0, 1)
    ax.set_title(metric)
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.25)

    for xi, value in zip(x, [b, p]):
        ax.annotate(
            f"{value:.3f}", (xi, value),
            xytext=(0, 9), textcoords="offset points",
            ha="center"
        )

fig.suptitle("Base vs Proposed Model: Classification Metrics")
fig.tight_layout()

out = Path("project_graphs")
out.mkdir(exist_ok=True)
fig.savefig(out / "four_metrics_combined.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)

print("Saved: project_graphs/four_metrics_combined.png")