import json
import pickle
import os

root = r"D:\LAV-DF\LAV-DF"

with open(os.path.join(root, "metadata.json"), encoding="utf-8") as f:
    metadata = json.load(f)

meta_map = {}
for x in metadata:
    name = x["split"] + "_" + os.path.basename(x["file"]).replace(".mp4", ".pkl")
    path = os.path.normcase(os.path.normpath(
        os.path.join(root, "decoder_voxceleb_feat", name)
    ))
    meta_map[path] = x

with open("lavdf_test_results.pkl", "rb") as f:
    predictions = pickle.load(f)

matched = mismatched = missing = 0
examples = []

for path, result in predictions.items():
    key = os.path.normcase(os.path.normpath(path))
    item = meta_map.get(key)

    if item is None:
        missing += 1
    elif result.get("labels", []) == item["fake_periods"]:
        matched += 1
    else:
        mismatched += 1
        if len(examples) < 3:
            examples.append((os.path.basename(path),
                             result.get("labels"),
                             item["fake_periods"]))

print("Total predictions:", len(predictions))
print("Matched:", matched)
print("Mismatched:", mismatched)
print("Metadata missing:", missing)
print("Mismatch examples:", examples)
