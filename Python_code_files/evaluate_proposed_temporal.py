import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


DATASET_ROOT = r"C:\Users\AMAYA\Downloads\FakeAVCeleb_v1.2\FakeAVCeleb"
TEST_JSON = os.path.join(DATASET_ROOT, "test_subset_data.json")
FEATURE_DIR = os.path.join(DATASET_ROOT, "decoder_voxceleb_feat")

MODEL_PATH = r"C:\Users\AMAYA\ics-av-deepfake\checkpoints\proposed_temporal_best.pt"

BATCH_SIZE = 4
MAX_FRAMES = 250


class ProposedDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        feature_name = item["feature_file"].replace(".json", ".pkl")
        feature_path = os.path.join(FEATURE_DIR, feature_name)

        data = torch.load(feature_path, map_location="cpu")

        av = np.asarray(data["av_hidden_feat"], dtype=np.float32)
        v = np.asarray(data["v_feat"], dtype=np.float32)
        a = np.asarray(data["a_feat"], dtype=np.float32)

        av = torch.tensor(av[:MAX_FRAMES])
        v = torch.tensor(v[:MAX_FRAMES])
        a = torch.tensor(a[:MAX_FRAMES])

        label = 1 if item["type"] == "RealVideo-RealAudio" else 0

        return av, v, a, label


def pad_collate(batch):
    av_list, v_list, a_list, labels = zip(*batch)

    max_len = max(x.shape[0] for x in av_list)

    def pad_sequence(x):
        if x.shape[0] < max_len:
            padding = torch.zeros(
                max_len - x.shape[0],
                x.shape[1]
            )
            x = torch.cat([x, padding], dim=0)

        return x

    av = torch.stack([pad_sequence(x) for x in av_list])
    v = torch.stack([pad_sequence(x) for x in v_list])
    a = torch.stack([pad_sequence(x) for x in a_list])

    labels = torch.tensor(
        labels,
        dtype=torch.float32
    )

    return av, v, a, labels


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=64):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.projection = nn.Linear(
            hidden_dim * 2,
            256
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        output = torch.mean(output, dim=1)
        return self.projection(output)


class ReliabilityModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.network(x)


class ProposedTemporalModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.av_encoder = TemporalEncoder()
        self.v_encoder = TemporalEncoder()
        self.a_encoder = TemporalEncoder()

        self.av_reliability = ReliabilityModule()
        self.v_reliability = ReliabilityModule()
        self.a_reliability = ReliabilityModule()

        self.fusion_projection = nn.Linear(256, 256)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, av, v, a):
        av_feat = self.av_encoder(av)
        v_feat = self.v_encoder(v)
        a_feat = self.a_encoder(a)

        av_score = self.av_reliability(av_feat)
        v_score = self.v_reliability(v_feat)
        a_score = self.a_reliability(a_feat)

        reliability = torch.cat(
            [av_score, v_score, a_score],
            dim=1
        )

        weights = torch.softmax(
            reliability,
            dim=1
        )

        fused = (
            weights[:, 0:1] * av_feat
            + weights[:, 1:2] * v_feat
            + weights[:, 2:3] * a_feat
        )

        fused = self.fusion_projection(fused)
        logits = self.classifier(fused).squeeze(1)

        return logits, weights


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

dataset = ProposedDataset(TEST_JSON)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=pad_collate
)

print("Test samples:", len(dataset))

model = ProposedTemporalModel().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("Model loaded successfully.")

all_labels = []
all_predictions = []
all_probabilities = []
all_weights = []

with torch.no_grad():
    for av, v, a, labels in loader:
        av = av.to(device)
        v = v.to(device)
        a = a.to(device)

        logits, weights = model(av, v, a)

        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).long()

        all_labels.extend(labels.numpy())
        all_predictions.extend(predictions.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())
        all_weights.extend(weights.cpu().numpy())

labels = np.array(all_labels)
predictions = np.array(all_predictions)
probabilities = np.array(all_probabilities)
weights = np.array(all_weights)

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

print("\nPROPOSED TEMPORAL TEST RESULTS")
print("Test samples:", len(labels))
print(f"Accuracy : {accuracy:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall   : {recall:.6f}")
print(f"F1 Score : {f1:.6f}")
print(f"AUC      : {auc:.6f}")
print(f"AP       : {ap:.6f}")

print("\nConfusion Matrix:")
print(cm)

average_weights = weights.mean(axis=0)

print("\nAverage Adaptive Weights:")
print(f"A-V: {average_weights[0]:.6f}")
print(f"V-V: {average_weights[1]:.6f}")
print(f"A-A: {average_weights[2]:.6f}")