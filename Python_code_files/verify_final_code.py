from pathlib import Path
import ast
import json
import pickle
import torch

ROOT = Path(r"D:\ics-av-deepfake_team_share")
PROPOSED = Path(r"D:\proposed_model_package\proposed_model")
BASELINE = ROOT / "baseline_final"

scripts = [
    BASELINE / "train_regression_pt_lightning.py",
    BASELINE / "evaluate_regression_bmn_pt_lightning.py",
    PROPOSED / "train.py",
    PROPOSED / "evaluate.py",
    PROPOSED / "train_reliability.py",
    PROPOSED / "evaluate_reliability.py",
    PROPOSED / "train_temporal_localizer_fp32.py",
]

print("\n=== PYTHON SYNTAX CHECK ===")
for path in scripts:
    if not path.exists():
        print("NOT FOUND:", path)
        continue
    try:
        ast.parse(path.read_text(encoding="utf-8"))
        print("PASS:", path.name)
    except Exception as e:
        print("FAIL:", path.name, repr(e))

print("\n=== CHECKPOINT FILE CHECK ===")
checkpoints = [
    BASELINE / "last.ckpt",
    BASELINE / "class_model_lavdf-epoch=8-val_loss=0.593.ckpt",
    PROPOSED / "proposed_best.pt",
    PROPOSED / "proposed_reliability_best.pt",
    PROPOSED / "outputs" / "temporal_localizer_best.pt",
]
for path in checkpoints:
    print(("FOUND: " if path.is_file() else "MISSING: "), path)

print("\n=== CHECKPOINT LOAD CHECK ===")
for path in checkpoints:
    if not path.is_file():
        continue
    try:
        obj = torch.load(path, map_location="cpu")
        print("LOAD PASS:", path.name, "| type:", type(obj).__name__)
        if isinstance(obj, dict):
            print("  keys:", list(obj.keys())[:12])
    except Exception as e:
        print("LOAD FAIL:", path.name, repr(e))

print("\n=== SAVED RESULT FILE CHECK ===")
result_files = [
    BASELINE / "baseline_temporal_metrics.json",
    BASELINE / "lavdf_test_results.pkl",
    PROPOSED / "outputs" / "original_same_metrics.json",
    PROPOSED / "outputs" / "proposed_reliability_test_metrics.json",
    PROPOSED / "outputs" / "proposed_reliability_dev_metrics.json",
    PROPOSED / "outputs" / "temporal_localizer_test_metrics.json",
    ROOT / "common_temporal_ap_comparison.json",
]
for path in result_files:
    if not path.is_file():
        print("MISSING:", path)
        continue
    try:
        if path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix == ".pkl":
            with path.open("rb") as f:
                data = pickle.load(f)
        else:
            continue
        print("READ PASS:", path.name, "| type:", type(data).__name__)
    except Exception as e:
        print("READ FAIL:", path.name, repr(e))

print("\nVerification checks finished.")
print("Note: syntax/file/load checks do not prove full training reproducibility.")