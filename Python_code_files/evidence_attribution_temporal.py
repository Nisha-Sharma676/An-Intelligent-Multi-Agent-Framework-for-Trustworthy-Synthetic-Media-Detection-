import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


DATASET_ROOT = r"C:\Users\AMAYA\Downloads\FakeAVCeleb_v1.2\FakeAVCeleb"
TEST_JSON = os.path.join(DATASET_ROOT, "test_subset_data.json")
FEATURE_DIR = os.path.join(DATASET_ROOT, "decoder_voxceleb_feat")

MODEL_PATH = r"C:\Users\AMAYA\ics-av-deepfake\checkpoints\proposed_temporal_best.pt"

OUTPUT_PATH = r"C:\Users\AMAYA\ics-av-deepfake\checkpoints\temporal_evidence_results.json"

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

results = []

with torch.no_grad():
    sample_index = 0

    for av, v, a, labels in loader:
        av = av.to(device)
        v = v.to(device)
        a = a.to(device)

        logits, weights = model(av, v, a)

        probabilities = torch.sigmoid(logits)

        for i in range(len(labels)):
            probability = float(
                probabilities[i].cpu().item()
            )

            av_weight = float(
                weights[i, 0].cpu().item()
            )

            vv_weight = float(
                weights[i, 1].cpu().item()
            )

            aa_weight = float(
                weights[i, 2].cpu().item()
            )

            actual_label = int(
                labels[i].item()
            )

            if 0.40 <= probability <= 0.60:
                decision = "UNCERTAIN"
                explanation = (
                    "The model is uncertain because the "
                    "predicted fake probability is close "
                    "to the decision boundary."
                )
            elif probability >= 0.5:
                decision = "FAKE"
                explanation = (
                    "The model predicts a fake sample "
                    "based on the combined evidence."
                )
            else:
                decision = "REAL"
                explanation = (
                    "The model predicts a real sample "
                    "based on the combined evidence."
                )

            evidence_weights = {
                "A-V Cross-modal": av_weight,
                "V-V Intra-modal": vv_weight,
                "A-A Intra-modal": aa_weight
            }

            primary_evidence = max(
                evidence_weights,
                key=evidence_weights.get
            )

            if decision == "UNCERTAIN":
                explanation = (
                    "The model is uncertain because the "
                    "predicted fake probability is close "
                    "to the decision boundary."
                )
            else:
                explanation = (
                    f"{primary_evidence} evidence received "
                    "the highest adaptive weight."
                )

            result = {
                "sample_index": sample_index,
                "fake_probability": probability,
                "decision": decision,
                "A-V_weight": av_weight,
                "V-V_weight": vv_weight,
                "A-A_weight": aa_weight,
                "primary_evidence": primary_evidence,
                "actual_label": "REAL" if actual_label == 1 else "FAKE",
                "explanation": explanation
            }

            results.append(result)

            if sample_index < 10:
                print("\nSample:", sample_index)
                print(f"Fake Probability: {probability:.4f}")
                print("Decision:", decision)
                print(f"A-V Weight: {av_weight:.4f}")
                print(f"V-V Weight: {vv_weight:.4f}")
                print(f"A-A Weight: {aa_weight:.4f}")
                print("Primary Evidence:", primary_evidence)
                print(
                    "Actual Label:",
                    "REAL" if actual_label == 1 else "FAKE"
                )
                print("Explanation:", explanation)

            sample_index += 1


fake_count = sum(
    1 for x in results
    if x["decision"] == "FAKE"
)

real_count = sum(
    1 for x in results
    if x["decision"] == "REAL"
)

uncertain_count = sum(
    1 for x in results
    if x["decision"] == "UNCERTAIN"
)

print("\nSummary:")
print("Total samples:", len(results))
print("Fake predictions:", fake_count)
print("Real predictions:", real_count)
print("Uncertain predictions:", uncertain_count)

with open(OUTPUT_PATH, "w") as f:
    json.dump(
        results,
        f,
        indent=4
    )

print("\nEvidence results saved at:")
print(OUTPUT_PATH)