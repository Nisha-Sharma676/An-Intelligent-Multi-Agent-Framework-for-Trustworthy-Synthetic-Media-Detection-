import pickle
import glob
import torch
import numpy as np
import csv
import os

from model.class_model import ClassificationModel


# Load classification model
model = ClassificationModel.load_from_checkpoint(
    "checkpoints/class_last.ckpt",
    map_location="cpu"
)

model.eval()

# Feature files
feature_folder = (
    r"C:\Users\AMAYA\Downloads\FakeAVCeleb_v1.2"
    r"\FakeAVCeleb\decoder_voxceleb_feat"
)

files = sorted(glob.glob(os.path.join(feature_folder, "*.pkl")))

print("Feature files found:", len(files))

results = []

for i, f in enumerate(files, 1):

    try:
        with open(f, "rb") as file:
            d = pickle.load(file)

        # Convert features to tensors
        av = torch.tensor(
            np.asarray(d["av_hidden_feat"]),
            dtype=torch.float32
        ).unsqueeze(0)

        v = torch.tensor(
            np.asarray(d["v_feat"]),
            dtype=torch.float32
        ).unsqueeze(0)

        a = torch.tensor(
            np.asarray(d["a_feat"]),
            dtype=torch.float32
        ).unsqueeze(0)

        # Actual number of frames
        n = av.shape[1]

        max_len = 500

        # Pad to 500 frames
        av = torch.nn.functional.pad(
            av, (0, 0, 0, max_len - n)
        )

        v = torch.nn.functional.pad(
            v, (0, 0, 0, max_len - n)
        )

        a = torch.nn.functional.pad(
            a, (0, 0, 0, max_len - n)
        )

        # Padding mask
        mask = torch.ones(
            (1, max_len),
            dtype=torch.bool
        )

        mask[:, :n] = False

        frames = torch.tensor([n])

        # Prediction
        with torch.no_grad():
            logit = model(
                av,
                v,
                a,
                frames,
                mask
            )

            probability = torch.sigmoid(logit).item()

        # Threshold = 0.5
        prediction = "Fake" if probability >= 0.5 else "Real"

        results.append([
            os.path.basename(f),
            n,
            probability,
            prediction
        ])

        print(
            f"[{i}/{len(files)}] "
            f"{os.path.basename(f)} -> "
            f"{probability:.4f} -> {prediction}"
        )

    except Exception as e:
        print(
            f"[{i}/{len(files)}] ERROR: "
            f"{os.path.basename(f)} -> {e}"
        )


# Save CSV
output_file = "predictions.csv"

with open(
    output_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "filename",
        "frames",
        "fake_probability",
        "prediction"
    ])

    writer.writerows(results)


print()
print("================================")
print("Prediction completed!")
print("Results saved to:", output_file)
print("Total predictions:", len(results))
print("================================")