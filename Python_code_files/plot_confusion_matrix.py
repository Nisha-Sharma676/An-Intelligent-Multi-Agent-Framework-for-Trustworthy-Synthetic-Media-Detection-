import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

# =========================================
# 1. PROJECT PATHS
# =========================================
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

OUTPUT_FILE = OUTPUT_DIR / "judge_fusion_confusion_matrix.png"

THRESHOLD = 0.5

# =========================================
# 2. LOAD PREDICTIONS
# =========================================
with open(PREDICTIONS_FILE, "r", encoding="utf-8") as f:
    prediction_data = json.load(f)

if isinstance(prediction_data, list):
    predictions = prediction_data
elif isinstance(prediction_data, dict):
    predictions = prediction_data.get(
        "predictions",
        prediction_data.get("results", [])
    )
else:
    raise ValueError("Unexpected prediction JSON format.")

if not predictions:
    raise ValueError(
        "Predictions list is empty. Check the JSON structure."
    )

# =========================================
# 3. LOAD METADATA
# =========================================
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

# Keep only test-set metadata
test_metadata = [
    item for item in metadata_records
    if str(item.get("split", "")).lower() == "test"
]

def normalize_path(value):
    return str(value).replace("\\", "/").strip().lower()

def filename_key(value):
    return normalize_path(value).split("/")[-1]

metadata_by_path = {}
metadata_by_filename = {}

for item in test_metadata:
    file_name = item.get("file")

    if file_name is None or "n_fakes" not in item:
        continue

    metadata_by_path[normalize_path(file_name)] = item
    metadata_by_filename[filename_key(file_name)] = item

# =========================================
# 4. MATCH PREDICTIONS WITH TRUE LABELS
# =========================================
y_true = []
y_pred = []
unmatched_files = []

for prediction in predictions:
    file_name = prediction.get("file")

    if not file_name or "fake_probability" not in prediction:
        continue

    item = metadata_by_path.get(normalize_path(file_name))

    if item is None:
        item = metadata_by_filename.get(filename_key(file_name))

    if item is None:
        unmatched_files.append(file_name)
        continue

    # Ground truth: n_fakes > 0 means Fake
    actual_label = 1 if float(item["n_fakes"]) > 0 else 0

    # Predicted label: 0 = Real, 1 = Fake
    fake_probability = float(prediction["fake_probability"])
    predicted_label = 1 if fake_probability >= THRESHOLD else 0

    y_true.append(actual_label)
    y_pred.append(predicted_label)

if not y_true:
    raise ValueError(
        "No predictions matched metadata. "
        "Check filenames and metadata paths."
    )

# =========================================
# 5. CALCULATE CONFUSION MATRIX
# =========================================
# Order: Real (0), Fake (1)
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

tn, fp, fn, tp = cm.ravel()
accuracy = accuracy_score(y_true, y_pred)

print("\n========== JUDGE FUSION ==========")
print("Dataset: LAV-DF")
print("Split: Test")
print(f"Matched samples: {len(y_true)}")
print(f"Unmatched predictions: {len(unmatched_files)}")
print(f"Threshold: {THRESHOLD}")

print("\nConfusion Matrix:")
print(cm)

print(f"\nTrue Negative (TN): {tn}")
print(f"False Positive (FP): {fp}")
print(f"False Negative (FN): {fn}")
print(f"True Positive (TP): {tp}")
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

# =========================================
# 6. PLOT BLUE CONFUSION MATRIX
# =========================================
fig, ax = plt.subplots(figsize=(8, 6))

image = ax.imshow(cm, cmap="Blues")
fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

class_names = ["Real", "Fake"]

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(class_names, fontsize=12)
ax.set_yticklabels(class_names, fontsize=12)

ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
ax.set_ylabel("Actual Label", fontsize=12, labelpad=10)

ax.set_title(
    "Judge Fusion Confusion Matrix - LAV-DF Test",
    fontsize=14,
    pad=15
)

cell_names = [
    ["TN", "FP"],
    ["FN", "TP"]
]

for row in range(2):
    for col in range(2):
        value = int(cm[row, col])

        text_color = (
            "white" if value > cm.max() * 0.55 else "black"
        )

        ax.text(
            col,
            row,
            f"{cell_names[row][col]}\n{value}",
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            color=text_color
        )

fig.tight_layout()
fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)

print("\nConfusion matrix saved successfully!")
print(f"Location: {OUTPUT_FILE}")