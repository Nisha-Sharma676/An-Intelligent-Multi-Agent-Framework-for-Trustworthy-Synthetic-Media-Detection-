import pickle
import csv
import os

PREDICTION_FILE = r"decoder_voxceleb\class_ckpt_None\prediction.pkl"
OUTPUT_FILE = r"results\base_model\base_predictions.csv"

# Load predictions
with open(PREDICTION_FILE, "rb") as f:
    predictions = pickle.load(f)

rows = []

for filepath, item in predictions.items():
    score = float(item["pred"])
    actual_label = int(item["label"])

    # Label mapping:
    # 0 = Fake
    # 1 = Real
    actual = "Fake" if actual_label == 0 else "Real"

    # Prediction using threshold 0.5
    predicted_label = 1 if score >= 0.5 else 0
    predicted = "Real" if predicted_label == 1 else "Fake"

    correct = "Yes" if predicted_label == actual_label else "No"

    rows.append([
        filepath,
        score,
        actual,
        predicted,
        correct
    ])

# Create output directory if needed
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# Save CSV
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    writer.writerow([
        "Filepath",
        "Prediction_Score",
        "Actual",
        "Predicted",
        "Correct"
    ])

    writer.writerows(rows)

print("Total samples:", len(rows))
print("Prediction file created:", OUTPUT_FILE)
print()
print("First 10 predictions:")

for row in rows[:10]:
    print(
        f"{row[2]:5s} | "
        f"Score: {row[1]:.4f} | "
        f"Predicted: {row[3]:5s} | "
        f"Correct: {row[4]}"
    )