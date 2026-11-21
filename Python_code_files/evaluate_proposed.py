import os
import json
import pickle
import numpy as np

import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


ROOT = r"C:\Users\AMAYA\Downloads\FakeAVCeleb_v1.2\FakeAVCeleb"
FEATURE_DIR = os.path.join(ROOT, "decoder_voxceleb_feat")

TEST_JSON = os.path.join(ROOT, "test_subset_data.json")

MODEL_PATH = r"C:\Users\AMAYA\ics-av-deepfake\checkpoints\proposed_best.pt"

DEVICE = torch.device("cpu")


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


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

model = ProposedModel().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

print("Model loaded successfully.")


# ------------------------------------------------------------
# Load test metadata
# ------------------------------------------------------------

with open(TEST_JSON, "r") as f:
    metadata = json.load(f)

labels = []
probabilities = []

weight_values = []


# ------------------------------------------------------------
# Test
# ------------------------------------------------------------

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

        av = np.mean(av, axis=0)
        v = np.mean(v, axis=0)
        a = np.mean(a, axis=0)

        av = torch.tensor(av).unsqueeze(0)
        v = torch.tensor(v).unsqueeze(0)
        a = torch.tensor(a).unsqueeze(0)

        logit, weights = model(
            av,
            v,
            a
        )

        probability = torch.sigmoid(
            logit
        ).item()

        # 1 = Fake
        # 0 = Real
        label = (
            0
            if item["type"] == "RealVideo-RealAudio"
            else 1
        )

        labels.append(label)
        probabilities.append(probability)

        weight_values.append(
            weights.squeeze(0).numpy()
        )


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

labels = np.asarray(labels)
probabilities = np.asarray(probabilities)
weight_values = np.asarray(weight_values)

predictions = (
    probabilities >= 0.5
).astype(int)


accuracy = accuracy_score(
    labels,
    predictions
)

precision = precision_score(
    labels,
    predictions,
    zero_division=0
)

recall = recall_score(
    labels,
    predictions,
    zero_division=0
)

f1 = f1_score(
    labels,
    predictions,
    zero_division=0
)

auc = roc_auc_score(
    labels,
    probabilities
)

ap = average_precision_score(
    labels,
    probabilities
)

cm = confusion_matrix(
    labels,
    predictions
)


# ------------------------------------------------------------
# Average reliability weights
# ------------------------------------------------------------

average_weights = np.mean(
    weight_values,
    axis=0
)


print("\n==============================")
print("PROPOSED METHOD TEST RESULTS")
print("==============================")

print("Test samples:", len(labels))

print("\nAccuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
print("AUC      :", auc)
print("AP       :", ap)

print("\nConfusion Matrix:")
print(cm)

print("\nAverage Adaptive Weights:")
print("A-V:", average_weights[0])
print("V-V:", average_weights[1])
print("A-A:", average_weights[2])