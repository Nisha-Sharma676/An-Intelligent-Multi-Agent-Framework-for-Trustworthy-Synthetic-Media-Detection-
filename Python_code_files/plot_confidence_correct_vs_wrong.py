import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent

JSON_PATH = (
    BASE_DIR
    / "proposed_model"
    / "outputs"
    / "fakeavceleb_proposed_494_results.json"
)

OUT_DIR = BASE_DIR / "project_graphs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load predictions
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

if isinstance(data, dict):
    for key in ["results", "predictions", "samples", "data"]:
        if isinstance(data.get(key), list):
            data = data[key]
            break

df = pd.DataFrame(data)

required = {"actual", "fake_probability", "predicted"}
missing = required - set(df.columns)

if missing:
    raise ValueError(
        f"Missing columns: {missing}. "
        f"Available columns: {list(df.columns)}"
    )

# Convert labels to binary: Fake = 1, Real = 0
def convert_label(value):
    if isinstance(value, (int, float, np.integer, np.floating)):
        if value in (0, 1):
            return int(value)

    label = str(value).strip().lower()

    if label in {"fake", "1", "true", "yes"}:
        return 1
    if label in {"real", "0", "false", "no"}:
        return 0

    raise ValueError(f"Unknown label: {value}")

y_true = df["actual"].map(convert_label)
y_pred = df["predicted"].map(convert_label)
fake_probability = pd.to_numeric(
    df["fake_probability"], errors="raise"
)

if not fake_probability.between(0, 1).all():
    raise ValueError("fake_probability must be between 0 and 1.")

# Confidence in the predicted class:
# if predicted Fake, confidence = P(Fake)
# if predicted Real, confidence = P(Real)
df["confidence"] = np.where(
    y_pred == 1,
    fake_probability,
    1 - fake_probability
)

# Use stored correctness if available; otherwise calculate it
if "correct" in df.columns:
    correct_values = df["correct"].astype(str).str.lower()
    df["is_correct"] = correct_values.isin(
        ["true", "1", "yes"]
    )
else:
    df["is_correct"] = (y_true == y_pred)

correct = df.loc[df["is_correct"], "confidence"].to_numpy()
wrong = df.loc[~df["is_correct"], "confidence"].to_numpy()

if len(correct) == 0 or len(wrong) == 0:
    raise ValueError(
        f"Need both correct and wrong predictions. "
        f"Correct={len(correct)}, Wrong={len(wrong)}"
    )

# Save row-level confidence data
csv_path = OUT_DIR / "confidence_correct_vs_wrong.csv"
df.to_csv(csv_path, index=False)

# Plot histogram and box plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

bins = np.linspace(0.5, 1.0, 21)

axes[0].hist(
    correct,
    bins=bins,
    alpha=0.65,
    label=f"Correct (n={len(correct)})",
    color="#91CDE8",
    edgecolor="white"
)
axes[0].hist(
    wrong,
    bins=bins,
    alpha=0.75,
    label=f"Wrong (n={len(wrong)})",
    color="#145DA0",
    edgecolor="white"
)

axes[0].set_title("Confidence Distribution", fontweight="bold")
axes[0].set_xlabel("Confidence in Predicted Class")
axes[0].set_ylabel("Number of Videos")
axes[0].set_xlim(0.5, 1.0)
axes[0].grid(axis="y", linestyle=":", alpha=0.3)
axes[0].set_axisbelow(True)
axes[0].legend(loc="upper left", bbox_to_anchor=(0, 1.0))

box = axes[1].boxplot(
    [correct, wrong],
    tick_labels=["Correct", "Wrong"],
    patch_artist=True,
    showmeans=True,
    medianprops={"color": "#102A43", "linewidth": 2},
    meanprops={
        "marker": "D",
        "markerfacecolor": "#F4A261",
        "markeredgecolor": "#F4A261"
    }
)

for patch, color in zip(box["boxes"], ["#91CDE8", "#145DA0"]):
    patch.set_facecolor(color)

axes[1].set_title("Correct vs Wrong Confidence", fontweight="bold")
axes[1].set_ylabel("Confidence in Predicted Class")
axes[1].set_ylim(0.45, 1.02)
axes[1].grid(axis="y", linestyle=":", alpha=0.3)
axes[1].set_axisbelow(True)

fig.suptitle(
    "Confidence Analysis on FakeAVCeleb Predictions",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()

png_path = OUT_DIR / "confidence_correct_vs_wrong.png"
plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nConfidence analysis graph created!")
print(f"Total videos: {len(df)}")
print(f"Correct predictions: {len(correct)}")
print(f"Wrong predictions: {len(wrong)}")
print(f"Mean confidence (correct): {correct.mean():.4f}")
print(f"Mean confidence (wrong): {wrong.mean():.4f}")
print(f"Graph saved: {png_path}")
print(f"CSV saved: {csv_path}")