import os
import json
import csv

DATASET_ROOT = r"D:\LAV-DF\LAV-DF"
FEATURE_DIR = os.path.join(DATASET_ROOT, "decoder_voxceleb_feat")

OUTPUT_DIR = r"D:\ics-av-deepfake_team_share\results\stage1"

TRAIN_TARGET = 5000
DEV_TARGET = 1000

METADATA_FILE = os.path.join(DATASET_ROOT, "metadata.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def feature_name_from_metadata(item):
    feature_file = "_".join(
        str(item["file"]).strip().split("/")
    )
    return feature_file.replace(".mp4", ".pkl")


print("=" * 70)
print("LAV-DF FEATURE MATCHING / SELECTION")
print("=" * 70)

print("\nLoading metadata...")

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print("Total metadata:", len(metadata))

existing_features = set(
    os.listdir(FEATURE_DIR)
)

print("Total existing feature files:", len(existing_features))


matched = {
    "train": [],
    "dev": [],
    "test": []
}

for item in metadata:

    split = item["split"]

    if split not in matched:
        continue

    feature_name = feature_name_from_metadata(item)

    if feature_name in existing_features:
        matched[split].append({
            "file": item["file"],
            "feature_file": feature_name
        })


print("\nExisting matched features:")
print("Train:", len(matched["train"]))
print("Dev:", len(matched["dev"]))
print("Test:", len(matched["test"]))


# ------------------------------------------------------------
# SELECT TARGET SAMPLES
# ------------------------------------------------------------

selected_train = matched["train"][:TRAIN_TARGET]
selected_dev = matched["dev"][:DEV_TARGET]
selected_test = matched["test"]


# ------------------------------------------------------------
# SAVE MANIFESTS
# ------------------------------------------------------------

def save_csv(data, filename):

    path = os.path.join(OUTPUT_DIR, filename)

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=["file", "feature_file"]
        )

        writer.writeheader()
        writer.writerows(data)

    print("Saved:", path)


save_csv(
    selected_train,
    "lavdf_train_selected.csv"
)

save_csv(
    selected_dev,
    "lavdf_dev_selected.csv"
)

save_csv(
    selected_test,
    "lavdf_test_selected.csv"
)


print("\n" + "=" * 70)
print("SELECTION COMPLETE")
print("=" * 70)

print("Selected Train:", len(selected_train))
print("Selected Dev:", len(selected_dev))
print("Selected Test:", len(selected_test))

print("\nTarget:")
print("Train:", TRAIN_TARGET)
print("Dev:", DEV_TARGET)
print("Test: all existing matched")

print("\nManifest files saved in:")
print(OUTPUT_DIR)