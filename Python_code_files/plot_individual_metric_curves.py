import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score
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

# Calculate requested metrics
metric_values = {
    "Accuracy": (
        accuracy_score(y_base, pred_base),
        accuracy_score(y_prop, pred_prop)
    ),
    "ROC-AUC": (
        roc_auc_score(y_base, p_base),
        roc_auc_score(y_prop, p_prop)
    ),
    "Fake Recall": (
        recall_score(y_base, pred_base, zero_division=0),
        recall_score(y_prop, pred_prop, zero_division=0)
    ),
    "Fake Precision": (
        precision_score(y_base, pred_base, zero_division=0),
        precision_score(y_prop, pred_prop, zero_division=0)
    )
}

for name, (b, p) in metric_values.items():
    print(f"{name}: Base={b:.4f}, Proposed={p:.4f}")

# Four separate curve plots
out = Path("project_graphs")
out.mkdir(exist_ok=True)

colors = ["#B8F2F2", "#073B8C"]

for metric, (base_value, proposed_value) in metric_values.items():
    plt.figure(figsize=(7, 5))

    x = ["Base - FakeAVCeleb", "Proposed - LAV-DF"]
    values = [base_value, proposed_value]

    plt.plot(
        x, values,
        marker="o", linewidth=2.5,
        markersize=9, color=colors[1]
    )

    for i, value in enumerate(values):
        plt.annotate(
            f"{value:.3f}",
            (i, value),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center"
        )

    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title(f"{metric}: Base vs Proposed")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    filename = metric.lower().replace(" ", "_").replace("-", "")
    plt.savefig(out / f"{filename}_curve.png", dpi=300)
    plt.show()
    plt.close()

print("All four plots saved in project_graphs/")