import json
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score
)

OUT = r"proposed_model\outputs"

with open(OUT + r"\proposed_test_results.json", "r") as f:
    results = json.load(f)

with open(r"D:\LAV-DF\LAV-DF\metadata.json", "r") as f:
    metadata = json.load(f)

# Map video filename to REAL/FAKE label
labels = {}
for item in metadata:
    name = item.get("file") or item.get("filename") or item.get("video_path") or item.get("path")
    if name:
        labels[name.replace("\\", "/").split("/")[-1]] = int(
            item.get("n_fakes", 0) > 0
        )

predictions = results["predictions"]
y_true, vv, aa, av, judge = [], [], [], [], []
missing = []

for p in predictions:
    name = p["file"].replace("\\", "/").split("/")[-1]
    if name not in labels:
        missing.append(name)
        continue

    y_true.append(labels[name])
    agents = p["agent_fake_probability"]
    vv.append(float(agents["vv"]))
    aa.append(float(agents["aa"]))
    av.append(float(agents["av"]))
    judge.append(float(p["fake_probability"]))

y = np.array(y_true)
vv, aa, av, judge = map(np.array, [vv, aa, av, judge])
average = (vv + aa + av) / 3

print(f"\nMatched videos: {len(y)} / {len(predictions)}")
print(f"Unmatched videos: {len(missing)}")

if len(y) == 0 or len(np.unique(y)) < 2:
    print("ERROR: Labels could not be matched correctly.")
    print("Metadata sample:", metadata[0])
    print("Prediction sample:", predictions[0]["file"])
else:
    models = [
        ("V-V Agent", vv),
        ("A-A Agent", aa),
        ("A-V Agent", av),
        ("Equal Average", average),
        ("Judge Fusion", judge),
    ]

    print("\n{:<18} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
        "Model", "Acc", "Prec", "Recall", "F1", "ROC-AUC", "AP"
    ))

    for name, scores in models:
        pred = (scores >= 0.5).astype(int)
        print("{:<18} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f}".format(
            name,
            accuracy_score(y, pred),
            precision_score(y, pred, zero_division=0),
            recall_score(y, pred, zero_division=0),
            f1_score(y, pred, zero_division=0),
            roc_auc_score(y, scores),
            average_precision_score(y, scores),
        ))

    report = {
        name: {
            "accuracy": float(accuracy_score(y, (s >= 0.5).astype(int))),
            "precision": float(precision_score(y, (s >= 0.5).astype(int), zero_division=0)),
            "recall": float(recall_score(y, (s >= 0.5).astype(int), zero_division=0)),
            "f1": float(f1_score(y, (s >= 0.5).astype(int), zero_division=0)),
            "roc_auc": float(roc_auc_score(y, s)),
            "average_precision": float(average_precision_score(y, s)),
        }
        for name, s in models
    }

    with open(OUT + r"\fusion_comparison.json", "w") as f:
        json.dump({
            "matched_videos": len(y),
            "unmatched_videos": len(missing),
            "results": report
        }, f, indent=2)

    print("\nSaved results to:", OUT + r"\fusion_comparison.json")