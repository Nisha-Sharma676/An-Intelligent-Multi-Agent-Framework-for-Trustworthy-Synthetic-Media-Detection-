import json
import pickle
from pathlib import Path

import numpy as np
import torch


# Existing prediction file
PREDICTIONS_FILE = Path("lavdf_inferenceresults.pkl")

# LAV-DF metadata file
METADATA_FILE = Path(r"D:\LAV-DF\LAV-DF\metadata.json")


def to_numpy(value):
    return torch.as_tensor(value).detach().cpu().numpy()


def main():
    if not PREDICTIONS_FILE.exists():
        print("ERROR: Prediction file not found:", PREDICTIONS_FILE)
        return

    if not METADATA_FILE.exists():
        print("ERROR: Metadata file not found:", METADATA_FILE)
        print("Please verify the dataset path.")
        return

    with open(PREDICTIONS_FILE, "rb") as file:
        results = pickle.load(file)

    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    print("=" * 65)
    print("TEMPORAL METADATA AUDIT")
    print("=" * 65)

    print("Prediction videos:", len(results))
    print("Metadata type:", type(metadata).__name__)

    # Inspect metadata structure without assuming its schema
    if isinstance(metadata, dict):
        print("Metadata keys:", list(metadata.keys())[:15])

        if metadata:
            first_key = next(iter(metadata))
            print("First metadata key:", first_key)
            print("First metadata value type:",
                  type(metadata[first_key]).__name__)

            if isinstance(metadata[first_key], dict):
                print("First record fields:",
                      list(metadata[first_key].keys())[:30])

    elif isinstance(metadata, list):
        print("Metadata records:", len(metadata))

        if metadata:
            first_record = metadata[0]
            print("First record type:", type(first_record).__name__)

            if isinstance(first_record, dict):
                print("First record fields:",
                      list(first_record.keys())[:30])

    print("\n" + "-" * 65)
    print("SAMPLE PREDICTION RECORDS")
    print("-" * 65)

    for video_id, data in list(results.items())[:5]:
        proposals = to_numpy(data["proposals"]).reshape(-1, 3)
        labels = to_numpy(data["labels"]).reshape(-1, 2)

        print("\nVideo ID:", video_id)
        print("Prediction file exists:",
              Path(video_id).exists())
        print("Proposal time range:",
              (float(proposals[:, 1].min()),
               float(proposals[:, 2].max()))
              if len(proposals) else "No proposals")
        print("GT interval count:", len(labels))
        print("GT examples:", labels[:3].tolist())

    print("\n" + "=" * 65)
    print("Audit complete. No files were modified.")
    print("=" * 65)


if __name__ == "__main__":
    main()