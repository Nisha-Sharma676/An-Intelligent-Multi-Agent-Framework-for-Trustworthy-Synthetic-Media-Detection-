import json
from pathlib import Path

PROJECT = Path(r"D:\ics-av-deepfake_team_share")
BASELINE_FILE = PROJECT / "baseline_final" / "baseline_temporal_metrics.json"
PROPOSED_FILE = Path(r"D:\proposed_model_package\proposed_model\outputs\temporal_localizer_test_metrics.json")
OUT_DIR = Path(r"D:\proposed_model_package\proposed_model\outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required results file not found:\n{path}\n"
            "Do not retrain. Check the path and run this script again."
        )
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def metric(data, key):
    # Supports either flat keys such as AP@0.5 or nested dicts if present.
    if key in data:
        return float(data[key])
    for candidate in ("temporal_ap", "metrics", "results"):
        nested = data.get(candidate)
        if isinstance(nested, dict) and key in nested:
            return float(nested[key])
    raise KeyError(f"Could not find {key}. Available keys: {list(data.keys())}")


def main():
    baseline = load_json(BASELINE_FILE)
    proposed = load_json(PROPOSED_FILE)

    keys = ("AP@0.5", "AP@0.75", "AP@0.95")
    rows = []
    for key in keys:
        b = metric(baseline, key)
        p = metric(proposed, key)
        rows.append({
            "metric": key,
            "BMN_baseline": b,
            "new_temporal_localizer": p,
            "absolute_change": p - b,
            "relative_change_percent": ((p - b) / b * 100.0) if b else None,
        })

    report = {
        "baseline_file": str(BASELINE_FILE),
        "proposed_file": str(PROPOSED_FILE),
        "baseline_metrics": baseline,
        "proposed_metrics": proposed,
        "comparison": rows,
        "interpretation_note": (
            "Treat this as a preliminary numeric comparison. The new localizer and BMN "
            "may use different proposal-generation methods, so do not claim a definitive "
            "apples-to-apples superiority unless both are evaluated with the same protocol."
        ),
    }
    output = OUT_DIR / "final_temporal_comparison.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nTEMPORAL RESULTS COMPARISON")
    print("-" * 74)
    print(f"{'Metric':<12}{'BMN baseline':>16}{'New localizer':>18}{'Change':>14}")
    for row in rows:
        print(
            f"{row['metric']:<12}"
            f"{row['BMN_baseline']:>16.8f}"
            f"{row['new_temporal_localizer']:>18.8f}"
            f"{row['absolute_change']:>14.8f}"
        )
    print("\nNote:", report["interpretation_note"])
    print("\nSaved:", output)


if __name__ == "__main__":
    main()
