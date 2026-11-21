import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score
)

# ==========================================
# 1. BASE MODEL: FakeAVCeleb
# ==========================================
base = pd.read_csv("results/base_model/base_predictions.csv")

y_base = (
    base["Actual"].astype(str).str.strip().str.lower() == "fake"
).astype(int).to_numpy()

p_base = np.clip(
    1 - base["Prediction_Score"].to_numpy(dtype=float),
    1e-7, 1 - 1e-7
)

pred_base = (p_base >= 0.5).astype(int)

# ==========================================
# 2. PROPOSED MODEL: LAV-DF
# ==========================================
with open(
    "proposed_model/outputs/proposed_test_results.json",
    encoding="utf-8"
) as f:
    proposed = json.load(f)["predictions"]

with open(
    r"D:\LAV-DF\LAV-DF\metadata.json",
    encoding="utf-8"
) as f:
    metadata = json.load(f)

labels = {
    row["file"]: int(row["n_fakes"] > 0)
    for row in metadata
    if row.get("split") == "test"
}

df = pd.DataFrame(proposed)
df["actual"] = df["file"].map(labels)

if df["actual"].isna().any():
    raise ValueError("Some LAV-DF ground-truth labels are missing.")

y_prop = df["actual"].to_numpy(dtype=int)

p_prop = np.clip(
    df["fake_probability"].to_numpy(dtype=float),
    1e-7, 1 - 1e-7
)

pred_prop = (p_prop >= 0.5).astype(int)

# ==========================================
# 3. CALCULATE FIVE METRICS
# ==========================================
metrics = [
    "Accuracy",
    "ROC-AUC",
    "Fake Recall",
    "Fake Precision",
    "Average Precision (AP)"
]

base_scores = [
    accuracy_score(y_base, pred_base),
    roc_auc_score(y_base, p_base),
    recall_score(y_base, pred_base, zero_division=0),
    precision_score(y_base, pred_base, zero_division=0),
    average_precision_score(y_base, p_base)
]

proposed_scores = [
    accuracy_score(y_prop, pred_prop),
    roc_auc_score(y_prop, p_prop),
    recall_score(y_prop, pred_prop, zero_division=0),
    precision_score(y_prop, pred_prop, zero_division=0),
    average_precision_score(y_prop, p_prop)
]

print("\nBase: FakeAVCeleb | Proposed: LAV-DF")
print("Base samples:", len(y_base))
print("Proposed samples:", len(y_prop))
print("-" * 55)

for name, b, p in zip(metrics, base_scores, proposed_scores):
    print(f"{name:<25} Base: {b:.4f} | Proposed: {p:.4f}")

# ==========================================
# 4. SAVE PLOTS
# ==========================================
output = Path("project_graphs")
output.mkdir(exist_ok=True)

x = np.arange(len(metrics))
light_cyan = "#B8F2F2"
deep_blue = "#073B8C"

# LINE PLOT: all five metrics in one graph
plt.figure(figsize=(10, 6))

plt.plot(
    x, base_scores,
    marker="o", linewidth=2.5, markersize=8,
    color=light_cyan, label="Base - FakeAVCeleb"
)

plt.plot(
    x, proposed_scores,
    marker="o", linewidth=2.5, markersize=8,
    color=deep_blue, label="Proposed - LAV-DF"
)

plt.xticks(x, metrics, rotation=20, ha="right")
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Five Classification Metrics: Base vs Proposed")
plt.grid(axis="y", alpha=0.25)
plt.legend()
plt.tight_layout()

plt.savefig(
    output / "five_metrics_line_comparison.png",
    dpi=300, bbox_inches="tight"
)
plt.show()
plt.close()

# BAR CHART: all five metrics in one graph
width = 0.36

plt.figure(figsize=(10, 6))

plt.bar(
    x - width / 2, base_scores, width,
    color=light_cyan, label="Base - FakeAVCeleb"
)

plt.bar(
    x + width / 2, proposed_scores, width,
    color=deep_blue, label="Proposed - LAV-DF"
)

plt.xticks(x, metrics, rotation=20, ha="right")
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Five Classification Metrics: Base vs Proposed")
plt.grid(axis="y", alpha=0.25)
plt.legend()
plt.tight_layout()

plt.savefig(
    output / "five_metrics_bar_comparison.png",
    dpi=300, bbox_inches="tight"
)
plt.show()
plt.close()

print("\nBoth plots saved in project_graphs/")