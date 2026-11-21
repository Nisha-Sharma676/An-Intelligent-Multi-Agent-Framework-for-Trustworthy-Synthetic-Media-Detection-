import os
import json

ROOT = r"C:\Users\AMAYA\Downloads\FakeAVCeleb_v1.2\FakeAVCeleb"
FEATURE_DIR = os.path.join(ROOT, "decoder_voxceleb_feat")


def create_subset(split_name):
    input_json = os.path.join(ROOT, f"{split_name}_data.json")
    output_json = os.path.join(ROOT, f"{split_name}_subset_data.json")

    with open(input_json, "r") as f:
        metadata = json.load(f)

    available = []

    for item in metadata:
        feature_name = os.path.splitext(item["feature_file"])[0] + ".pkl"
        feature_path = os.path.join(FEATURE_DIR, feature_name)

        if os.path.isfile(feature_path):
            available.append(item)

    with open(output_json, "w") as f:
        json.dump(available, f, indent=2)

    real_count = sum(
        1 for item in available
        if item["type"] == "RealVideo-RealAudio"
    )

    fake_count = len(available) - real_count

    print(f"\n{split_name.upper()}")
    print("Total:", len(available))
    print("Real :", real_count)
    print("Fake :", fake_count)
    print("Saved:", output_json)


create_subset("train")
create_subset("val")