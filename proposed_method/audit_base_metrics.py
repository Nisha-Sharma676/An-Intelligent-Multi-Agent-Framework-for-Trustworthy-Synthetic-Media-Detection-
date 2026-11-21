import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, log_loss, brier_score_loss
)

INPUT_FILE = r"results\base_model\base_predictions.csv"
OUTPUT_FILE = r"results\base_model\base_missing_metrics_audit.json"

df = pd.read_csv(INPUT_FILE)

# Actual labels: Fake = 1, Real = 0
y_true = (df["Actual"].str.strip().str.lower() == "fake").astype(int).to_numpy()

# Score in CSV represents P(Real), so convert it to P(Fake)
p_fake = 1.0 - df["Prediction_Score"].astype(float).to_numpy()
p_fake = np.clip(p_fake, 1e-7, 1 - 1e-7)

# Standard classification threshold
y_pred = (p_fake >= 0.5).astype(int)

# Expected Calibration Error (10 bins)
confidence = np.maximum(p_fake, 1 - p_fake)
correct = (y_pred == y_true).astype(float)
bin_ids = np.minimum((confidence * 10).astype(int), 9)

ece = 0.0
for b in range(10):
    mask = bin_ids == b
    if mask.any():
        ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())

# Abstention policy: REAL below .40, FAKE at/above .60
# Between these thresholds, return UNCERTAIN
decided = (p_fake < 0.40) | (p_fake >= 0.60)
selective_pred = (p_fake[decided] >= 0.60).astype(int)

results = {
    "model": "Base Model",
    "dataset": "FakeAVCeleb",
    "samples": int(len(df)),
    "probability_used": "P(Fake) = 1 - Prediction_Score",
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "precision_fake": float(precision_score(y_true, y_pred, zero_division=0)),
    "recall_fake": float(recall_score(y_true, y_pred, zero_division=0)),
    "f1_fake": float(f1_score(y_true, y_pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_true, p_fake)),
    "average_precision": float(average_precision_score(y_true, p_fake)),
    "NLL": float(log_loss(y_true, p_fake)),
    "Brier_score": float(brier_score_loss(y_true, p_fake)),
    "ECE_10_bins": float(ece),
    "confusion_matrix_TN_FP_FN_TP": confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel().tolist(),
    "coverage": float(decided.mean()),
    "uncertainty_rate": float((~decided).mean()),
    "selective_risk": (
        float((selective_pred != y_true[decided]).mean())
        if decided.any() else None
    ),
    "decided_samples": int(decided.sum()),
    "uncertain_samples": int((~decided).sum()),
    "selective_policy": "REAL if p_fake < 0.40; FAKE if p_fake >= 0.60; otherwise UNCERTAIN"
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
print("\nSaved new audit to:", OUTPUT_FILE)