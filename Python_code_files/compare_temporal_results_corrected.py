import json
from pathlib import Path

PROJECT = Path(r"D:\ics-av-deepfake_team_share")
BASELINE_FILE = PROJECT / "baseline_final" / "baseline_temporal_metrics.json"
PROPOSED_FILE = Path(
    r"D:\proposed_model_package\proposed_model\outputs\temporal_localizer_test_metrics.json"
)
OUT_DIR = Path(r"D:\proposed_model_package\proposed_model\outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required results file not found:\n{path}\n"
            "Check the path. Do not rerun training."
        )
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def find_metric(data, key):
    """Read metrics whether stored as AP@0.5 or inside temporal_AP."""
    aliases = {
        "AP@0.5": ("AP@0.5", "AP@0.50", "0.5", "0.50"),
        "AP@0.75": ("AP@0.75", "0.75"),
        "AP@0.95": ("AP@0.95", "0.95"),
    }
    candidates = aliases[key]

    containers = [data]
    for name in ("temporal_AP", "temporal_ap", "metrics", "results"):
        nested = data.get(name)
        if isinstance(nested, dict):
            containers.append(nested)

    for container in containers:
        for name in candidates:
            if name in container:
                value = container[name]
                # Some formats may store each metric as {"value": ...}.
                if isinstance(value, dict) and "value" in value:
                    value = value["value"]
                return float(value)

    raise KeyError(
        f"Could not find {key}. Top-level keys: {list(data.keys())}. "
        f"Please inspect the JSON metric structure."
    )


def main():
    baseline = load_json(BASELINE_FILE)
    proposed = load_json(PROPOSED_FILE)

    keys = ("AP@0.5", "AP@0.75", "AP@0.95")
    rows = []
    for key in keys:
        baseline_value = find_metric(baseline, key)
        proposed_value = find_metric(proposed, key)
        rows.append({
            "metric": key,
            "BMN_baseline": baseline_value,
            "new_temporal_localizer": proposed_value,
            "absolute_change": proposed_value - baseline_value,
            "relative_change_percent": (
                (proposed_value - baseline_value) / baseline_value * 100.0
                if baseline_value != 0 else None
            ),
        })

    report = {
        "baseline_file": str(BASELINE_FILE),
        "proposed_file": str(PROPOSED_FILE),
        "baseline_metrics": baseline,
        "proposed_metrics": proposed,
        "comparison": rows,
        "interpretation_note": (
            "Preliminary numeric comparison only. A definitive apples-to-apples "
            "claim requires both methods to use the same proposal-generation and "
            "evaluation protocol."
        ),
    }

    output = OUT_DIR / "final_temporal_comparison.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nTEMPORAL RESULTS COMPARISON")
    print("-" * 76)
    print(f"{'Metric':<12}{'BMN baseline':>18}{'New localizer':>20}{'Change':>14}")
    for row in rows:
        print(
            f"{row['metric']:<12}"
            f"{row['BMN_baseline']:>18.8f}"
            f"{row['new_temporal_localizer']:>20.8f}"
            f"{row['absolute_change']:>14.8f}"
        )

    print("\nNote:", report["interpretation_note"])
    print("\nSaved:", output)


if __name__ == "__main__":
    main()
