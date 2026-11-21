import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# ==========================================
# 1. FILE PATHS
# ==========================================
BASE_DIR = Path(__file__).resolve().parent

PREDICTIONS_FILE = (
    BASE_DIR
    / "proposed_model"
    / "outputs"
    / "proposed_test_results.json"
)

METADATA_FILE = Path(r"D:\LAV-DF\LAV-DF\metadata.json")

OUTPUT_DIR = BASE_DIR / "project_graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "judge_fusion_roc_comparison.png"

# ==========================================
# 2. LOAD DATA
# ==========================================
with open(PREDICTIONS_FILE, "r", encoding="utf-8") as f:
    prediction_data = json.load(f)

predictions = prediction_data["predictions"]

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata_data = json.load(f)

if isinstance(metadata_data, dict):
    if "file" in metadata_data:
        metadata_records = [metadata_data]
    else:
        metadata_records = [
            value for value in metadata_data.values()
            if isinstance(value, dict) and "file" in value
        ]
else:
    metadata_records = metadata_data

# ==========================================
# 3. MATCH TEST LABELS
# ==========================================
def normalize_path(value):
    return str(value).replace("\\", "/").strip().lower()

def filename_key(value):
    return normalize_path(value).split("/")[-1]

metadata_by_path = {}
metadata_by_filename = {}

for item in metadata_records:
    if str(item.get("split", "")).lower() != "test":
        continue

    file_name = item.get("file")
    if file_name is None or "n_fakes" not in item:
        continue

    metadata_by_path[normalize_path(file_name)] = item
    metadata_by_filename[filename_key(file_name)] = item

y_true = []
scores = {
    "V-V Agent": [],
    "A-A Agent": [],
    "A-V Agent": [],
    "Judge Fusion": [],
}

unmatched = []

for pred in predictions:
    file_name = pred.get("file")
    if not file_name:
        continue

    item = metadata_by_path.get(normalize_path(file_name))

    if item is None:
        item = metadata_by_filename.get(filename_key(file_name))

    if item is None:
        unmatched.append(file_name)
        continue

    # Ground truth: 0 = Real, 1 = Fake
    y_true.append(1 if float(item["n_fakes"]) > 0 else 0)

    agent_probs = pred.get("agent_fake_probability", {})

    scores["V-V Agent"].append(float(agent_probs["vv"]))
    scores["A-A Agent"].append(float(agent_probs["aa"]))
    scores["A-V Agent"].append(float(agent_probs["av"]))
    scores["Judge Fusion"].append(float(pred["fake_probability"]))

if not y_true:
    raise ValueError("No predictions matched metadata.")

# ==========================================
# 4. PLOT ROC CURVES
# ==========================================
colors = {
    "V-V Agent": "#A8D8EA",
    "A-A Agent": "#5DADE2",
    "A-V Agent": "#2874A6",
    "Judge Fusion": "#123B63",
}

fig, ax = plt.subplots(figsize=(10, 8))

for model_name, model_scores in scores.items():
    auc_value = roc_auc_score(y_true, model_scores)

    fpr, tpr, _ = roc_curve(y_true, model_scores)

    ax.plot(
        fpr,
        tpr,
        linewidth=2.5,
        color=colors[model_name],
        label=f"{model_name} (AUC = {auc_value:.3f})"
    )

# Random classifier reference
ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    linewidth=1.5,
    label="Random Classifier (AUC = 0.500)"
)

ax.set_title(
    "ROC Curve Comparison - Proposed Model (LAV-DF Test)",
    fontsize=14,
    pad=15
)

ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.grid(True, linestyle="--", alpha=0.3)

# Legend outside the plot
ax.legend(
    loc="lower right",
    frameon=True,
    fontsize=10
)

fig.tight_layout()

fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)

print("\n========== ROC COMPARISON ==========")
print(f"Matched test samples: {len(y_true)}")
print(f"Unmatched predictions: {len(unmatched)}")

for model_name, model_scores in scores.items():
    auc_value = roc_auc_score(y_true, model_scores)
    print(f"{model_name}: ROC-AUC = {auc_value:.4f}")

print(f"\nROC graph saved to:\n{OUTPUT_FILE}")