import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from feature_loader import AVFeatureDataset, collate_features
from agents import ProposedAVModel


ROOT = Path(__file__).resolve().parent.parent
FEATURE_DIR = r"D:\LAV-DF\LAV-DF\decoder_voxceleb_feat"
METADATA = r"D:\LAV-DF\LAV-DF\metadata.json"
CHECKPOINT = ROOT / "proposed_model" / "outputs" / "proposed_reliability_best.pt"
OUTPUT = ROOT / "proposed_model" / "outputs" / "modality_ablation_dev.json"


def calculate_metrics(y, p):
    pred = (p >= 0.5).astype(int)

    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    result = {
        "accuracy": float((pred == y).mean()),
        "precision_fake": precision,
        "recall_fake": recall,
        "f1_fake": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }

    try:
        from sklearn.metrics import roc_auc_score
        result["roc_auc"] = (
            float(roc_auc_score(y, p))
            if len(np.unique(y)) == 2 else None
        )
    except ImportError:
        result["roc_auc"] = None

    return result


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset = AVFeatureDataset(FEATURE_DIR, METADATA, "dev")
    print("Dev samples:", len(dataset))

    if len(dataset) == 0:
        raise RuntimeError("Dev dataset is empty.")

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_features,
    )

    model = ProposedAVModel().to(device)
    checkpoint = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(
        checkpoint.get("model_state_dict", checkpoint),
        strict=True,
    )
    model.eval()

    # Full model plus one-modality-at-a-time and modality-removal tests.
    experiments = {
        "full_av": (),
        "visual_only": ("a", "av"),
        "audio_only": ("v", "av"),
        "audiovisual_only": ("v", "a"),
        "without_visual": ("v",),
        "without_audio": ("a",),
        "without_audiovisual": ("av",),
    }

    collected = {
        name: {"labels": [], "probabilities": []}
        for name in experiments
    }

    with torch.no_grad():
        for batch in loader:
            y = batch["y"].to(device)

            for name, masked_keys in experiments.items():
                x = {
                    key: batch[key].to(device).clone()
                    for key in ("v", "a", "av")
                }

                for key in masked_keys:
                    x[key] = torch.zeros_like(x[key])

                out = model(x)
                probabilities = (
                    torch.sigmoid(out["logit"])
                    .reshape(-1)
                    .cpu()
                    .numpy()
                )

                collected[name]["labels"].extend(
                    y.reshape(-1).cpu().numpy().astype(int).tolist()
                )
                collected[name]["probabilities"].extend(
                    probabilities.tolist()
                )

    results = {
        "split": "dev",
        "checkpoint": str(CHECKPOINT),
        "samples": len(dataset),
        "method": (
            "Inference-time zero masking. This is diagnostic ablation, "
            "not retraining with individual modalities."
        ),
        "experiments": {},
    }

    for name, values in collected.items():
        y = np.asarray(values["labels"], dtype=int)
        p = np.asarray(values["probabilities"], dtype=float)

        results["experiments"][name] = calculate_metrics(y, p)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n========== MODALITY ABLATION: DEV ==========")
    print(f"{'Experiment':<24} {'Accuracy':>10} {'F1':>10} {'ROC-AUC':>10}")

    for name, metrics in results["experiments"].items():
        auc = metrics.get("roc_auc")
        auc_text = f"{auc:.4f}" if auc is not None else "N/A"

        print(
            f"{name:<24} "
            f"{metrics['accuracy']:.4f} "
            f"{metrics['f1_fake']:.4f} "
            f"{auc_text:>10}"
        )

    print("\nSaved results:", OUTPUT)


if __name__ == "__main__":
    main()