from pathlib import Path
import json
import math

ROOT = Path(r"D:\ics-av-deepfake_team_share")
BASELINE = ROOT / "baseline_final"
OUT = Path(r"D:\proposed_model_package\proposed_model\outputs")

def load(path):
    if not path.is_file():
        print("MISSING:", path)
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print("READ ERROR:", path, e)
        return None

def close(a, b, tol=1e-8):
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)

print("\n=== TEMPORAL RESULT CONSISTENCY ===")
b = load(BASELINE / "baseline_temporal_metrics.json")
p = load(OUT / "temporal_localizer_test_metrics.json")
c = load(ROOT / "common_temporal_ap_comparison.json")

if b and p and c:
    print("Baseline videos:", b.get("videos"))
    print("Proposed videos:", p.get("videos"))
    print("Common-evaluator videos:", c.get("videos_compared"))
    print("Common-evaluator GT segments:", c.get("ground_truth_segments"))

    expected = {
        "0.5": (b.get("temporal_AP", {}).get("0.5"),
                p.get("AP@0.5")),
        "0.75": (b.get("temporal_AP", {}).get("0.75"),
                 p.get("AP@0.75")),
        "0.95": (b.get("temporal_AP", {}).get("0.95"),
                 p.get("AP@0.95")),
    }

    for threshold, (b_saved, p_saved) in expected.items():
        row = c.get("AP", {}).get(threshold, {})
        if b_saved is None or p_saved is None or not row:
            print("CHECK INCOMPLETE at IoU", threshold)
            continue
        ok_b = close(row["BMN_baseline"], b_saved)
        ok_p = close(row["proposed_localizer"], p_saved)
        print(
            f"IoU {threshold}: source metrics vs common evaluator "
            f"BMN={'MATCH' if ok_b else 'DIFFER'}; "
            f"proposed={'MATCH' if ok_p else 'DIFFER'}"
        )
        print("  Common AP:", row)

print("\n=== CLASSIFICATION METRICS FOR REPORT CHECK ===")
for filename in [
    "original_same_metrics.json",
    "proposed_reliability_test_metrics.json",
    "proposed_reliability_dev_metrics.json",
]:
    data = load(OUT / filename)
    if data:
        print("\n", filename)
        print(json.dumps(data, indent=2))

print("\nCompare these printed values with the final report tables.")