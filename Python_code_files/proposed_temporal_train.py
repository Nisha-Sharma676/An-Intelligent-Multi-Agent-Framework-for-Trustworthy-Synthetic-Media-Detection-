import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


DATASET_ROOT = r"C:\Users\AMAYA\Downloads\FakeAVCeleb_v1.2\FakeAVCeleb"
TRAIN_JSON = os.path.join(DATASET_ROOT, "train_subset_data.json")
VAL_JSON = os.path.join(DATASET_ROOT, "val_subset_data.json")
FEATURE_DIR = os.path.join(DATASET_ROOT, "decoder_voxceleb_feat")

MODEL_SAVE_PATH = r"C:\Users\AMAYA\ics-av-deepfake\checkpoints\proposed_temporal_best.pt"

BATCH_SIZE = 4
EPOCHS = 15
LEARNING_RATE = 1e-4
MAX_FRAMES = 250


class ProposedDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            self.data = json.load(f)

        print(os.path.basename(json_path), "samples:", len(self.data))

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
    labels = torch.tensor(labels, dtype=torch.float32)

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

        self.projection = nn.Linear(hidden_dim * 2, 256)

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

        weights = torch.softmax(reliability, dim=1)

        fused = (
            weights[:, 0:1] * av_feat
            + weights[:, 1:2] * v_feat
            + weights[:, 2:3] * a_feat
        )

        fused = self.fusion_projection(fused)
        logits = self.classifier(fused).squeeze(1)

        return logits, weights


def evaluate(model, loader, device):
    model.eval()

    predictions = []
    labels = []
    probabilities = []

    with torch.no_grad():
        for av, v, a, y in loader:
            av = av.to(device)
            v = v.to(device)
            a = a.to(device)
            y = y.to(device)

            logits, _ = model(av, v, a)
            probs = torch.sigmoid(logits)

            predictions.extend((probs >= 0.5).cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
            labels.extend(y.cpu().numpy())

    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    labels = np.array(labels)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    auc = roc_auc_score(labels, probabilities)

    return accuracy, f1, auc


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

train_dataset = ProposedDataset(TRAIN_JSON)
val_dataset = ProposedDataset(VAL_JSON)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=pad_collate
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=pad_collate
)

real_count = sum(
    1 for x in train_dataset.data
    if x["type"] == "RealVideo-RealAudio"
)

fake_count = len(train_dataset.data) - real_count

print("Real samples:", real_count)
print("Fake samples:", fake_count)

pos_weight = torch.tensor(
    real_count / fake_count,
    dtype=torch.float32,
    device=device
)

print("Positive class weight:", pos_weight.item())

model = ProposedTemporalModel().to(device)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

best_auc = 0.0

print("\nSTARTING TEMPORAL TRAINING")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for av, v, a, y in train_loader:
        av = av.to(device)
        v = v.to(device)
        a = a.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits, _ = model(av, v, a)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)

    val_acc, val_f1, val_auc = evaluate(
        model,
        val_loader,
        device
    )

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"Val F1: {val_f1:.4f} | "
        f"Val AUC: {val_auc:.4f}"
    )

    if val_auc > best_auc:
        best_auc = val_auc

        torch.save(
            model.state_dict(),
            MODEL_SAVE_PATH
        )

        print("  Best temporal model saved")

print("\nTEMPORAL TRAINING COMPLETE")
print("Best Validation AUC:", best_auc)
print("Model saved at:", MODEL_SAVE_PATH)