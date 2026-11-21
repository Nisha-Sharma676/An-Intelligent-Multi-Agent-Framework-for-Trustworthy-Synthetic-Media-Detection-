import json
import pickle
from pathlib import Path

import numpy as np
import torch

PREDICTIONS_FILE = Path("lavdf_inferenceresults.pkl")
METADATA_FILE = Path(r"D:\LAV-DF\LAV-DF\metadata.json")


def to_array(value, columns):
    return (
        torch.as_tensor(value)
        .detach()
        .cpu()
        .numpy()
        .reshape(-1, columns)
    )


def normalize_filename(filename):
    """Convert feature filename into LAV-DF metadata filename."""
    name = Path(str(filename)).stem

    # test_000001.pkl -> test/000001.mp4
    if "_" in name:
        split, index = name.split("_", 1)
        if index.isdigit():
            return f"{split}/{index}.mp4".lower()

    return ""


def main():
    with open(PREDICTIONS_FILE, "rb") as file:
        predictions = pickle.load(file)

    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    metadata_index = {
        str(record["file"]).replace("\\", "/").lower(): record
        for record in metadata
        if record.get("file")
    }

    matched = 0
    mismatched_gt = 0
    missing = 0
    empty_gt = 0

    print("=" * 65)
    print("LAV-DF METADATA AND GROUND-TRUTH VERIFICATION")
    print("=" * 65)

    for video_id, data in predictions.items():

        metadata_key = normalize_filename(video_id)
        record = metadata_index.get(metadata_key)

        stored_gt = to_array(data["labels"], 2)

        if record is None:
            missing += 1
            if missing <= 5:
                print("\nMetadata not found:", video_id, "->", metadata_key)
            continue

        matched += 1

        metadata_gt = np.asarray(
            record.get("fake_periods", []),
            dtype=np.float64
        ).reshape(-1, 2)

        if len(stored_gt) == 0:
            empty_gt += 1

        same = (
            stored_gt.shape == metadata_gt.shape
            and np.allclose(stored_gt, metadata_gt, atol=1e-3)
        )

        if not same:
            mismatched_gt += 1
            if mismatched_gt <= 10:
                print("\nGT mismatch:", video_id)
                print("Metadata:", metadata_gt.tolist())
                print("Predictions file labels:", stored_gt.tolist())

        if matched <= 3:
            print("\nVideo:", video_id)
            print("Metadata file:", record["file"])
            print("Duration:", record.get("duration"))
            print("Metadata fake periods:", metadata_gt.tolist())
            print("Stored GT:", stored_gt.tolist())
            print("GT match:", same)

    print("\n" + "=" * 65)
    print("FINAL SUMMARY")
    print("=" * 65)
    print("Prediction records:", len(predictions))
    print("Metadata matches:", matched)
    print("Metadata not found:", missing)
    print("Ground-truth mismatches:", mismatched_gt)
    print("Matched videos with empty GT:", empty_gt)
    print("Original predictions were not modified.")


if __name__ == "__main__":
    main()