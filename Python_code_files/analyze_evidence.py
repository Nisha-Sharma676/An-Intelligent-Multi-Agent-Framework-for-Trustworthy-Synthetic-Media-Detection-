import json
import os
from pathlib import Path

PREDICTIONS_PATH = Path(r"proposed_model\outputs\proposed_test_results.json")
METADATA_PATH = Path(r"D:\LAV-DF\LAV-DF\metadata.json")
OUTPUT_PATH = Path(r"proposed_model\outputs\evidence_analysis.json")

def is_fake(row):
    return int(int(row.get("n_fakes", 0) or 0) > 0 or bool(row.get("fake_periods")))

def mean(values):
    return sum(values) / len(values) if values else None

def safe_round(value, digits=4):
    return round(value, digits) if value is not None else None

def main():
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Predictions file not found: {PREDICTIONS_PATH.resolve()}")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")

    results = json.loads(PREDICTIONS_PATH.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    test_meta = {
        os.path.basename(row["file"]).lower(): row
        for row in metadata
        if row.get("split") == "test"
    }

    matched = []
    for pred in results.get("predictions", []):
        row = test_meta.get(os.path.basename(pred.get("file", "")).lower())
        if row is not None:
            matched.append((pred, is_fake(row)))

    missing = len(results.get("predictions", [])) - len(matched)
    if not matched:
        raise RuntimeError("No prediction rows matched test metadata. Check file names and paths.")

    n = len(matched)
    decision_counts = {
        name: sum(pred.get("decision") == name for pred, _ in matched)
        for name in ("REAL", "FAKE", "UNCERTAIN")
    }
    uncertain = [(p, y) for p, y in matched if p.get("decision") == "UNCERTAIN"]
    decided = [(p, y) for p, y in matched if p.get("decision") in ("REAL", "FAKE")]

    def final_prediction_correct(p, y):
        decision = p.get("decision")
        return decision == ("FAKE" if y else "REAL")

    agent_results = {}
    for agent in ("vv", "aa", "av"):
        rows = [
            (p, y) for p, y in matched
            if agent in p.get("agent_fake_probability", {})
            and agent in p.get("reliability", {})
        ]
        correct, wrong = [], []
        for p, y in rows:
            predicted_fake = float(p["agent_fake_probability"][agent]) >= 0.5
            (correct if predicted_fake == bool(y) else wrong).append(p)

        rel_correct = [float(p["reliability"][agent]) for p in correct]
        rel_wrong = [float(p["reliability"][agent]) for p in wrong]
        rel_all = [float(p["reliability"][agent]) for p, _ in rows]
        prob_rows = [
            (float(p["agent_fake_probability"][agent]), y) for p, y in rows
        ]
        brier = mean([(prob - y) ** 2 for prob, y in prob_rows])
        agent_results[agent] = {
            "n": len(rows),
            "accuracy_at_threshold_0_5": safe_round(len(correct) / len(rows) if rows else None),
            "mean_reliability_all": safe_round(mean(rel_all)),
            "mean_reliability_when_correct": safe_round(mean(rel_correct)),
            "mean_reliability_when_wrong": safe_round(mean(rel_wrong)),
            "reliability_correct_minus_wrong": safe_round(
                (mean(rel_correct) - mean(rel_wrong))
                if rel_correct and rel_wrong else None
            ),
            "probability_brier_score": safe_round(brier),
            "note": "Reliability is not proven calibrated by these summaries alone."
        }

    report = {
        "source_predictions": str(PREDICTIONS_PATH),
        "source_metadata": str(METADATA_PATH),
        "matched_test_videos": n,
        "missing_metadata_matches": missing,
        "decision_counts": decision_counts,
        "uncertain": {
            "count": len(uncertain),
            "coverage_percent": safe_round(100 * len(uncertain) / n),
            "correctness_not_applicable": True,
            "note": "UNCERTAIN rows are abstentions; evaluate separately rather than count as REAL/FAKE."
        },
        "decided_only": {
            "count": len(decided),
            "coverage_percent": safe_round(100 * len(decided) / n),
            "accuracy": safe_round(
                sum(final_prediction_correct(p, y) for p, y in decided) / len(decided)
                if decided else None
            ),
        },
        "agent_reliability_analysis": agent_results,
        "limitations": [
            "This script analyzes existing predictions only; it does not train or modify the model.",
            "The model's agent inputs are feature representations, not explicit synchronization scores.",
            "A reliability score being high or low does not by itself establish calibrated trustworthiness.",
            "UNCERTAIN decisions are excluded from decided-only accuracy."
        ]
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Evidence analysis complete. No training was run.")
    print("Matched test videos:", n)
    print("Missing metadata matches:", missing)
    print("Decision counts:", decision_counts)
    print("Uncertain coverage (%):", report["uncertain"]["coverage_percent"])
    print("Decided-only accuracy:", report["decided_only"]["accuracy"])
    print("Agent reliability analysis:")
    for agent, values in agent_results.items():
        print(
            f"  {agent}: accuracy={values['accuracy_at_threshold_0_5']}, "
            f"rel_correct={values['mean_reliability_when_correct']}, "
            f"rel_wrong={values['mean_reliability_when_wrong']}, "
            f"delta={values['reliability_correct_minus_wrong']}"
        )
    print("Saved report:", OUTPUT_PATH.resolve())

if __name__ == "__main__":
    main()
