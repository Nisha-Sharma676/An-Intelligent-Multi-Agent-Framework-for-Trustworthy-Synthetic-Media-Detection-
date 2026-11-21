import os
import json
import pickle
import numpy as np

import torch
import torch.nn as nn


ROOT = r"C:\Users\AMAYA\Downloads\FakeAVCeleb_v1.2\FakeAVCeleb"
FEATURE_DIR = os.path.join(ROOT, "decoder_voxceleb_feat")

TEST_JSON = os.path.join(ROOT, "test_subset_data.json")

MODEL_PATH = r"C:\Users\AMAYA\ics-av-deepfake\checkpoints\proposed_best.pt"

DEVICE = torch.device("cpu")


# ============================================================
# MODEL
# ============================================================

class ProposedModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.av_reliability = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.v_reliability = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.a_reliability = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.av_projection = nn.Linear(256, 256)
        self.v_projection = nn.Linear(256, 256)
        self.a_projection = nn.Linear(256, 256)

        self.decision = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, av, v, a):

        av_rel = self.av_reliability(av)
        v_rel = self.v_reliability(v)
        a_rel = self.a_reliability(a)

        reliability = torch.cat(
            [av_rel, v_rel, a_rel],
            dim=1
        )

        weights = torch.softmax(
            reliability,
            dim=1
        )

        av_feat = self.av_projection(av)
        v_feat = self.v_projection(v)
        a_feat = self.a_projection(a)

        fused = (
            weights[:, 0:1] * av_feat
            + weights[:, 1:2] * v_feat
            + weights[:, 2:3] * a_feat
        )

        logit = self.decision(fused)

        return logit, weights


# ============================================================
# LOAD MODEL
# ============================================================

model = ProposedModel().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

print("Model loaded successfully.")


# ============================================================
# LOAD TEST DATA
# ============================================================

with open(TEST_JSON, "r") as f:
    metadata = json.load(f)


# ============================================================
# EXPLAIN FIRST 10 SAMPLES
# ============================================================

shown = 0

with torch.no_grad():

    for item in metadata:

        feature_name = (
            os.path.splitext(item["feature_file"])[0]
            + ".pkl"
        )

        feature_path = os.path.join(
            FEATURE_DIR,
            feature_name
        )

        if not os.path.isfile(feature_path):
            continue

        with open(feature_path, "rb") as f:
            data = pickle.load(f)

        av = np.asarray(
            data["av_hidden_feat"],
            dtype=np.float32
        )

        v = np.asarray(
            data["v_feat"],
            dtype=np.float32
        )

        a = np.asarray(
            data["a_feat"],
            dtype=np.float32
        )

        # Temporal pooling
        av = np.mean(av, axis=0)
        v = np.mean(v, axis=0)
        a = np.mean(a, axis=0)

        av_tensor = torch.tensor(av).unsqueeze(0)
        v_tensor = torch.tensor(v).unsqueeze(0)
        a_tensor = torch.tensor(a).unsqueeze(0)

        logit, weights = model(
            av_tensor,
            v_tensor,
            a_tensor
        )

        probability = torch.sigmoid(logit).item()

        weights = weights.squeeze(0).numpy()

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        if probability >= 0.5:
            prediction = "FAKE"
        else:
            prediction = "REAL"

        # ----------------------------------------------------
        # Uncertainty
        # ----------------------------------------------------

        distance_from_boundary = abs(
            probability - 0.5
        )

        if distance_from_boundary < 0.10:
            certainty = "UNCERTAIN"
        elif probability >= 0.5:
            certainty = "FAKE"
        else:
            certainty = "REAL"

        # ----------------------------------------------------
        # Evidence attribution
        # ----------------------------------------------------

        modality_names = [
            "A-V synchronization",
            "V-V synchronization",
            "A-A synchronization"
        ]

        max_index = np.argmax(weights)

        main_evidence = modality_names[max_index]

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        print("\n========================================")
        print("SAMPLE", shown + 1)
        print("========================================")

        print(
            "File:",
            item["feature_file"]
        )

        print(
            "Predicted probability of Fake:",
            round(probability, 4)
        )

        print(
            "Prediction:",
            prediction
        )

        print(
            "Decision:",
            certainty
        )

        print("\nReliability / Adaptive Weights:")

        print(
            "A-V:",
            round(float(weights[0]), 4)
        )

        print(
            "V-V:",
            round(float(weights[1]), 4)
        )

        print(
            "A-A:",
            round(float(weights[2]), 4)
        )

        print(
            "\nPrimary evidence contribution:",
            main_evidence
        )

        print(
            "Actual dataset label:",
            "FAKE"
            if item["type"] != "RealVideo-RealAudio"
            else "REAL"
        )

        shown += 1

        if shown >= 10:
            break


print("\n========================================")
print("EVIDENCE EXPLANATION COMPLETE")
print("========================================")