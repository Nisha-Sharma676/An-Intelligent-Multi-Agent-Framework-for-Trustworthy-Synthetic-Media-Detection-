import os
import json

# Current project paths
ROOT = r"D:\ics-av-deepfake_team_share\data_files\FakeAVCeleb"
FEATURE_DIR = r"D:\ics-av-deepfake_team_share\decoder_voxceleb_feat"

# Output folder
OUTPUT_ROOT = r"D:\ics-av-deepfake_team_share\clean_eval_data\FakeAVCeleb"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# --------------------------------------------------
# 1. Load original metadata
# --------------------------------------------------

splits = {}

for split in ["train", "val", "test"]:
    input_json = os.path.join(ROOT, f"{split}_data.json")

    with open(input_json, "r") as f:
        splits[split] = json.load(f)

    print(f"{split.upper()} metadata:", len(splits[split]))


# --------------------------------------------------
# 2. Find all feature files that actually exist
# --------------------------------------------------

available = {}

for filename in os.listdir(FEATURE_DIR):
    if filename.lower().endswith(".pkl"):
        available[filename] = os.path.join(FEATURE_DIR, filename)

print("\nAvailable PKL features:", len(available))


# --------------------------------------------------
# 3. Match metadata with existing features
# --------------------------------------------------

matched = {}

for split in ["train", "val", "test"]:
    for item in splits[split]:

        feature_file = item.get("feature_file")

        if not feature_file:
            continue

        feature_name = os.path.splitext(feature_file)[0] + ".pkl"

        if feature_name in available:
            matched.setdefault(feature_name, []).append(
                (split, item)
            )


print("Unique matched features:", len(matched))


# --------------------------------------------------
# 4. Resolve overlaps
#
# Priority:
# TEST > VAL > TRAIN
# --------------------------------------------------

clean_splits = {
    "train": [],
    "val": [],
    "test": []
}

for feature_name, entries in matched.items():

    # Check which splits contain this feature
    split_names = [x[0] for x in entries]

    if "test" in split_names:
        selected_split = "test"

    elif "val" in split_names:
        selected_split = "val"

    else:
        selected_split = "train"

    # Use the metadata item belonging to selected split
    selected_item = next(
        item for split, item in entries
        if split == selected_split
    )

    clean_splits[selected_split].append(selected_item)


# --------------------------------------------------
# 5. Save clean JSON files
# --------------------------------------------------

for split in ["train", "val", "test"]:

    output_json = os.path.join(
        OUTPUT_ROOT,
        f"{split}_data.json"
    )

    with open(output_json, "w") as f:
        json.dump(
            clean_splits[split],
            f,
            indent=2
        )

    real_count = sum(
        1
        for item in clean_splits[split]
        if str(item.get("type", "")).strip()
        == "RealVideo-RealAudio"
    )

    fake_count = len(clean_splits[split]) - real_count

    print(f"\n{split.upper()}")
    print("Total:", len(clean_splits[split]))
    print("Real :", real_count)
    print("Fake :", fake_count)
    print("Saved:", output_json)


# --------------------------------------------------
# 6. Final verification
# --------------------------------------------------

feature_sets = {}

for split in ["train", "val", "test"]:

    feature_sets[split] = set()

    for item in clean_splits[split]:

        feature_file = item.get("feature_file")

        if feature_file:
            feature_name = (
                os.path.splitext(feature_file)[0] + ".pkl"
            )
            feature_sets[split].add(feature_name)


print("\n========== FINAL VERIFICATION ==========")

print(
    "TRAIN ∩ VAL:",
    len(feature_sets["train"] & feature_sets["val"])
)

print(
    "TRAIN ∩ TEST:",
    len(feature_sets["train"] & feature_sets["test"])
)

print(
    "VAL ∩ TEST:",
    len(feature_sets["val"] & feature_sets["test"])
)

print(
    "TOTAL UNIQUE:",
    len(
        feature_sets["train"]
        | feature_sets["val"]
        | feature_sets["test"]
    )
)

print("\nCLEAN SPLIT CREATION COMPLETE!")