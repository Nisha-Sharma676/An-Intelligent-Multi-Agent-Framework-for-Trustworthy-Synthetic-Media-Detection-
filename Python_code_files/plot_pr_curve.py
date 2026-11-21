import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

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

OUTPUT_FILE = OUTPUT_DIR / "judge_fusion_pr_curve.png"

# ==========================================
# 2. LOAD PREDICTIONS AND METADATA
# ==========================================
with open(PREDICTIONS_FILE, "r", encoding="utf-8") as f:
    prediction_data = json.load(f)

predictions = prediction_data["predictions"]

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata_data = json.load(f)

if isinstance(metadata_data, list):
    metadata_records = metadata_data
elif isinstance(metadata_data, dict):
    if "file" in metadata_data:
        metadata_records = [metadata_data]
    else:
        metadata_records = [
            value for value in metadata_data.values()
            if isinstance(value, dict) and "file" in value
        ]
else:
    raise ValueError("Unexpected metadata JSON format.")

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

    agent_probs = pred.get("agent_fake_probability", {})

    # Skip incomplete prediction records
    if not all(k in agent_probs for k in ("vv", "aa", "av")):
        continue

    if "fake_probability" not in pred:
        continue

    # Ground truth: 0 = Real, 1 = Fake
    y_true.append(1 if float(item["n_fakes"]) > 0 else 0)

    scores["V-V Agent"].append(float(agent_probs["vv"]))
    scores["A-A Agent"].append(float(agent_probs["aa"]))
    scores["A-V Agent"].append(float(agent_probs["av"]))
    scores["Judge Fusion"].append(float(pred["fake_probability"]))

if not y_true:
    raise ValueError(
        "No predictions matched the test metadata. "
        "Check the file paths and JSON structure."
    )

# ==========================================
# 4. PLOT PRECISION-RECALL CURVES
# ==========================================
colors = {
    "V-V Agent": "#A8D8EA",
    "A-A Agent": "#5DADE2",
    "A-V Agent": "#2874A6",
    "Judge Fusion": "#123B63",
}

fig, ax = plt.subplots(figsize=(10, 8))

for model_name, model_scores in scores.items():
    precision, recall, _ = precision_recall_curve(
        y_true, model_scores
    )

    ap = average_precision_score(y_true, model_scores)

    ax.plot(
        recall,
        precision,
        linewidth=2.5,
        color=colors[model_name],
        label=f"{model_name} (AP = {ap:.3f})"
    )

# Baseline equals the proportion of fake samples
fake_rate = sum(y_true) / len(y_true)

ax.axhline(
    y=fake_rate,
    linestyle="--",
    color="gray",
    linewidth=1.5,
    label=f"Fake prevalence = {fake_rate:.3f}"
)

ax.set_title(
    "Precision-Recall Curve - Proposed Model (LAV-DF Test)",
    fontsize=14,
    pad=15
)

ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.grid(True, linestyle="--", alpha=0.3)

ax.legend(
    loc="lower left",
    frameon=True,
    fontsize=10
)

fig.tight_layout()
fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)

# ==========================================
# 5. PRINT RESULTS
# ==========================================
print("\n===== PRECISION-RECALL RESULTS =====")
print(f"Matched test samples: {len(y_true)}")
print(f"Fake samples: {sum(y_true)}")
print(f"Real samples: {len(y_true) - sum(y_true)}")
print(f"Unmatched metadata files: {len(unmatched)}")

for model_name, model_scores in scores.items():
    ap = average_precision_score(y_true, model_scores)
    print(f"{model_name}: Average Precision = {ap:.4f}")

print(f"\nGraph saved at:\n{OUTPUT_FILE}")