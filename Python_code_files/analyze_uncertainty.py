import json
from collections import Counter

FILE = r"C:\Users\AMAYA\ics-av-deepfake\checkpoints\temporal_evidence_results.json"

with open(FILE, "r") as f:
    results = json.load(f)

uncertain = [
    r for r in results
    if r["decision"] == "UNCERTAIN"
]

fake_uncertain = sum(
    1 for r in uncertain
    if r["actual_label"] == "FAKE"
)

real_uncertain = sum(
    1 for r in uncertain
    if r["actual_label"] == "REAL"
)

fake_total = sum(
    1 for r in results
    if r["actual_label"] == "FAKE"
)

real_total = sum(
    1 for r in results
    if r["actual_label"] == "REAL"
)

print()
print("==============================")
print("UNCERTAINTY ANALYSIS")
print("==============================")

print("Total samples:", len(results))

print("Uncertain samples:", len(uncertain))

print(
    "Uncertainty percentage:",
    f"{len(uncertain) / len(results) * 100:.2f}%"
)

print()

print("Actual FAKE samples:", fake_total)

print(
    "FAKE samples classified UNCERTAIN:",
    fake_uncertain
)

print(
    "Percentage of FAKE samples uncertain:",
    f"{fake_uncertain / fake_total * 100:.2f}%"
)

print()

print("Actual REAL samples:", real_total)

print(
    "REAL samples classified UNCERTAIN:",
    real_uncertain
)

print(
    "Percentage of REAL samples uncertain:",
    f"{real_uncertain / real_total * 100:.2f}%"
)

print()

print("==============================")
print("UNCERTAIN SAMPLE PROBABILITY")
print("==============================")

if uncertain:

    probabilities = [
        r["fake_probability"]
        for r in uncertain
    ]

    print(
        "Minimum:",
        f"{min(probabilities):.4f}"
    )

    print(
        "Maximum:",
        f"{max(probabilities):.4f}"
    )

    print(
        "Average:",
        f"{sum(probabilities) / len(probabilities):.4f}"
    )

print()
print("==============================")
print("DONE")
print("==============================")