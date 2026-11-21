import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "project_graphs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FAKEAV_PATH = BASE_DIR / "proposed_model" / "outputs" / "fakeavceleb_proposed_494_results.json"
LAVDF_PATH = BASE_DIR / "proposed_model" / "outputs" / "proposed_test_results.json"
LAVDF_METADATA_PATH = Path(r"D:\LAV-DF\LAV-DF\metadata.json")

THRESHOLD = 0.50


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def as_records(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ["results", "predictions", "samples", "data"]:
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError("Could not find prediction records list.")


def label_to_binary(value):
    if isinstance(value, (int, float, np.integer, np.floating)):
        if value in (0, 1):
            return int(value)
    value = str(value).strip().lower()
    if value in {"fake", "1", "true", "yes"}:
        return 1
    if value in {"real", "0", "false", "no"}:
        return 0
    return None


def file_key(value):
    return Path(str(value).replace("\\", "/")).stem.lower()


def load_lavdf_metadata(path):
    metadata = load_json(path)
    records = []
    collect_metadata(metadata, records)

    lookup = {}
    for record in records:
        identifier = get_identifier(record)
        if identifier is not None:
            lookup[file_key(identifier)] = record
    return lookup


def collect_metadata(obj, output):
    if isinstance(obj, dict):
        if "n_fakes" in obj:
            output.append(obj)
        for value in obj.values():
            if isinstance(value, (dict, list)):
                collect_metadata(value, output)
    elif isinstance(obj, list):
        for value in obj:
            if isinstance(value, (dict, list)):
                collect_metadata(value, output)


def get_identifier(record):
    for key in [
        "file", "filepath", "file_path", "path", "filename",
        "file_name", "video", "video_path", "video_name", "name", "id"
    ]:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def analyze_dataset(name, records, metadata_lookup=None):
    y_true = []
    y_pred = []
    skipped = 0

    for row in records:
        actual = None

        # Prefer the actual label stored with the prediction, if present
        for key in ["actual", "actual_label", "ground_truth", "label", "type"]:
            if key in row:
                actual = label_to_binary(row[key])
                if actual is not None:
                    break

        # For LAV-DF, derive label from metadata n_fakes > 0
        if actual is None and metadata_lookup is not None:
            identifier = get_identifier(row)
            if identifier is None:
                for key in ["file", "filepath", "file_path", "path", "filename"]:
                    if isinstance(row.get(key), str):
                        identifier = row[key]
                        break

            meta = metadata_lookup.get(file_key(identifier)) if identifier else None
            if meta is not None:
                try:
                    actual = int(float(meta["n_fakes"]) > 0)
                except (TypeError, ValueError):
                    actual = None

        # Use fake_probability consistently for both datasets
        probability = row.get("fake_probability")
        try:
            probability = float(probability)
        except (TypeError, ValueError):
            probability = None

        if actual is None or probability is None or not np.isfinite(probability):
            skipped += 1
            continue

        if not 0 <= probability <= 1:
            skipped += 1
            continue

        prediction = int(probability >= THRESHOLD)
        y_true.append(actual)
        y_pred.append(prediction)

    if not y_true:
        raise ValueError(
            f"No matched labelled predictions for {name}. "
            "Check file identifiers and metadata structure."
        )

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fake_recall = tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "dataset": name,
        "samples_evaluated": len(y_true),
        "skipped": skipped,
        "real_count": int(sum(v == 0 for v in y_true)),
        "fake_count": int(sum(v == 1 for v in y_true)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "FP_rate_among_real_pct": 100 * fp / (tn + fp) if (tn + fp) else 0.0,
        "FN_rate_among_fake_pct": 100 * fn / (tp + fn) if (tp + fn) else 0.0,
        "balanced_accuracy_pct": 100 * balanced_accuracy_score(y_true, y_pred),
        "specificity_pct": 100 * specificity,
        "fake_recall_pct": 100 * fake_recall,
    }


# Load FakeAVCeleb predictions
fakeav_records = as_records(load_json(FAKEAV_PATH))
fakeav = analyze_dataset("FakeAVCeleb", fakeav_records)

# Load LAV-DF predictions and metadata
lavdf_records = as_records(load_json(LAVDF_PATH))
metadata_lookup = load_lavdf_metadata(LAVDF_METADATA_PATH)
lavdf = analyze_dataset("LAV-DF", lavdf_records, metadata_lookup)

results = pd.DataFrame([lavdf, fakeav])

csv_path = OUT_DIR / "dataset_error_analysis.csv"
results.to_csv(csv_path, index=False)

# Plot FP and FN counts, plus balanced accuracy
fig, axes = plt.subplots(1, 3, figsize=(13, 5.5))
datasets = results["dataset"].tolist()
colors = ["#91CDE8", "#145DA0"]

plots = [
    ("FP", "False Positives (count)"),
    ("FN", "False Negatives (count)"),
    ("balanced_accuracy_pct", "Balanced Accuracy (%)"),
]

for ax, (column, title) in zip(axes, plots):
    values = results[column].astype(float).to_numpy()
    bars = ax.bar(datasets, values, color=colors, edgecolor="white")

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.set_axisbelow(True)

    if column == "balanced_accuracy_pct":
        ax.set_ylim(0, 105)

    for bar, value in zip(bars, values):
        label = f"{value:.2f}%" if column == "balanced_accuracy_pct" else f"{int(value)}"
        ax.annotate(
            label,
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=9
        )

fig.suptitle(
    f"Dataset-wise Error Analysis (Fake threshold = {THRESHOLD:.2f})",
    fontsize=14,
    fontweight="bold"
)
plt.tight_layout()

png_path = OUT_DIR / "dataset_error_analysis.png"
plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nDataset-wise Error Analysis completed!")
print(f"Threshold: {THRESHOLD}")
print(f"Graph saved: {png_path}")
print(f"CSV saved: {csv_path}")
print("\nResults:")
print(results.round(3).to_string(index=False))