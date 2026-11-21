import os
import json
import pickle
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# ============================================================
# PATHS
# ============================================================

ROOT = r"C:\Users\AMAYA\Downloads\FakeAVCeleb_v1.2\FakeAVCeleb"
FEATURE_DIR = os.path.join(ROOT, "decoder_voxceleb_feat")

TRAIN_JSON = os.path.join(ROOT, "train_subset_data.json")
VAL_JSON = os.path.join(ROOT, "val_subset_data.json")

MODEL_PATH = r"C:\Users\AMAYA\ics-av-deepfake\checkpoints\proposed_best.pt"


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 0.0001

DEVICE = torch.device("cpu")

print("Device:", DEVICE)


# ============================================================
# DATASET
# ============================================================

class FakeAVCelebDataset(Dataset):

    def __init__(self, json_file):

        with open(json_file, "r") as f:
            metadata = json.load(f)

        self.samples = []

        for item in metadata:

            feature_name = os.path.splitext(
                item["feature_file"]
            )[0] + ".pkl"

            feature_path = os.path.join(
                FEATURE_DIR,
                feature_name
            )

            if os.path.isfile(feature_path):

                # 1 = Fake
                # 0 = Real
                label = 0 if item["type"] == "RealVideo-RealAudio" else 1

                self.samples.append(
                    (feature_path, label)
                )

        print(
            os.path.basename(json_file),
            "samples:",
            len(self.samples)
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        feature_path, label = self.samples[index]

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

        # Temporal mean pooling
        av = np.mean(av, axis=0)
        v = np.mean(v, axis=0)
        a = np.mean(a, axis=0)

        return (
            torch.tensor(av),
            torch.tensor(v),
            torch.tensor(a),
            torch.tensor(label, dtype=torch.float32)
        )


# ============================================================
# PROPOSED MODEL
# ============================================================

class ProposedModel(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # Reliability Assessment
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Feature projection
        # ----------------------------------------------------

        self.av_projection = nn.Linear(256, 256)
        self.v_projection = nn.Linear(256, 256)
        self.a_projection = nn.Linear(256, 256)

        # ----------------------------------------------------
        # Decision Module
        # ----------------------------------------------------

        self.decision = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, av, v, a):

        # ----------------------------------------------------
        # Reliability scores
        # ----------------------------------------------------

        av_rel = self.av_reliability(av)
        v_rel = self.v_reliability(v)
        a_rel = self.a_reliability(a)

        reliability = torch.cat(
            [av_rel, v_rel, a_rel],
            dim=1
        )

        # Adaptive weighting
        weights = torch.softmax(
            reliability,
            dim=1
        )

        # ----------------------------------------------------
        # Project features
        # ----------------------------------------------------

        av_feat = self.av_projection(av)
        v_feat = self.v_projection(v)
        a_feat = self.a_projection(a)

        # ----------------------------------------------------
        # Adaptive Fusion
        # ----------------------------------------------------

        fused = (
            weights[:, 0:1] * av_feat
            + weights[:, 1:2] * v_feat
            + weights[:, 2:3] * a_feat
        )

        # ----------------------------------------------------
        # Decision
        # ----------------------------------------------------

        logit = self.decision(fused)

        return logit, weights


# ============================================================
# LOAD DATA
# ============================================================

train_dataset = FakeAVCelebDataset(
    TRAIN_JSON
)

val_dataset = FakeAVCelebDataset(
    VAL_JSON
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# MODEL
# ============================================================

model = ProposedModel().to(DEVICE)


# ============================================================
# CLASS WEIGHT
# ============================================================

fake_count = sum(
    label for _, label in train_dataset.samples
)

real_count = len(train_dataset) - fake_count

pos_weight = torch.tensor(
    [real_count / fake_count],
    dtype=torch.float32
).to(DEVICE)

print("Real samples:", real_count)
print("Fake samples:", fake_count)
print("Positive class weight:", pos_weight.item())


criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def evaluate():

    model.eval()

    all_labels = []
    all_probs = []

    total_loss = 0

    with torch.no_grad():

        for av, v, a, labels in val_loader:

            av = av.to(DEVICE)
            v = v.to(DEVICE)
            a = a.to(DEVICE)
            labels = labels.to(DEVICE)

            logits, weights = model(
                av, v, a
            )

            loss = criterion(
                logits.squeeze(1),
                labels
            )

            total_loss += loss.item()

            probs = torch.sigmoid(
                logits.squeeze(1)
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_probs.extend(
                probs.cpu().numpy()
            )

    all_labels = np.asarray(all_labels)
    all_probs = np.asarray(all_probs)

    predictions = (
        all_probs >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        all_labels,
        predictions
    )

    f1 = f1_score(
        all_labels,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        all_labels,
        all_probs
    )

    return (
        total_loss / len(val_loader),
        accuracy,
        f1,
        auc
    )


# ============================================================
# TRAINING
# ============================================================

best_auc = 0.0

print("\n==============================")
print("STARTING PROPOSED MODEL TRAINING")
print("==============================\n")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for av, v, a, labels in train_loader:

        av = av.to(DEVICE)
        v = v.to(DEVICE)
        a = a.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        logits, weights = model(
            av, v, a
        )

        loss = criterion(
            logits.squeeze(1),
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    val_loss, val_acc, val_f1, val_auc = evaluate()

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {total_loss / len(train_loader):.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"Val F1: {val_f1:.4f} | "
        f"Val AUC: {val_auc:.4f}"
    )

    # Save best model according to validation AUC
    if val_auc > best_auc:

        best_auc = val_auc

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            "  ✓ Best model saved"
        )


print("\n==============================")
print("TRAINING COMPLETE")
print("==============================")

print("Best Validation AUC:", best_auc)
print("Model saved at:")
print(MODEL_PATH)