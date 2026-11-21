import os
import json
import pickle

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, Callback

from model.pretrain_model import ConsistencyPretrainModel
from dataset.lavdf_inference import LavdfInference, Metadata
from dataset.collater import BatchCollaterInferencePTLightning
from model.sync_model import PositionalEncoding


# ============================================================
# SETTINGS
# ============================================================

DATASET_ROOT = r"D:\LAV-DF\LAV-DF"

CHECKPOINT = (
    r"D:\ics-av-deepfake_team_share"
    r"\checkpoints\last.ckpt"
)

OUTPUT_DIR = os.path.join(
    DATASET_ROOT,
    "decoder_voxceleb_feat"
)

NUM_SAMPLES = 10

BATCH_SIZE = 2

NUM_WORKERS = 0

FRAME_PADDING = 500

FPS = 25

IMG_SIZE = 96


# ============================================================
# CALLBACK
# Saves synchronization features into .pkl files
# ============================================================

class SaveFeatures(Callback):

    def on_predict_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0
    ):

        (
            cross_modal_out,
            v_out,
            a_out,
            av_hidden_feat,
            v_hidden_feat,
            a_hidden_feat,
            v_feat,
            a_feat
        ) = outputs

        batch_size = len(cross_modal_out)

        for i in range(batch_size):

            n_frames = int(batch["video_frames"][i])

            feature_file_path = (
                str(batch["feature_file_path"][i])
            )

            # Make sure .pkl extension is used
            feature_file_path = os.path.splitext(
                feature_file_path
            )[0] + ".pkl"

            os.makedirs(
                os.path.dirname(feature_file_path),
                exist_ok=True
            )

            # ------------------------------------------------
            # Create feature dictionary
            # ------------------------------------------------

            feat_json = {}

            feat_json["filepath"] = feature_file_path

            # Actual synchronization outputs
            feat_json["av_sync"] = (
                cross_modal_out[i][:n_frames]
                .detach()
                .cpu()
                .tolist()
            )

            feat_json["v_sync"] = (
                v_out[i][:n_frames]
                .detach()
                .cpu()
                .tolist()
            )

            feat_json["a_sync"] = (
                a_out[i][:n_frames]
                .detach()
                .cpu()
                .tolist()
            )

            # Hidden synchronization representation
            feat_json["av_hidden_feat"] = (
                av_hidden_feat[i][:n_frames]
                .detach()
                .cpu()
                .tolist()
            )

            # Original visual/audio features
            feat_json["v_feat"] = (
                v_feat[i][:n_frames]
                .detach()
                .cpu()
                .tolist()
            )

            feat_json["a_feat"] = (
                a_feat[i][:n_frames]
                .detach()
                .cpu()
                .tolist()
            )

            feat_json["video_frames"] = n_frames

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            with open(
                feature_file_path,
                "wb"
            ) as f:

                pickle.dump(
                    feat_json,
                    f
                )

            print(
                "[SAVED]",
                feature_file_path,
                "| frames:",
                n_frames
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("LAV-DF DEV FEATURE EXTRACTION")
    print("=" * 70)

    print("Dataset:")
    print(DATASET_ROOT)

    print("\nCheckpoint:")
    print(CHECKPOINT)

    print("\nNumber of samples:")
    print(NUM_SAMPLES)

    # ========================================================
    # 1. Check GPU
    # ========================================================

    print("\n" + "=" * 70)
    print("GPU CHECK")
    print("=" * 70)

    print(
        "CUDA available:",
        torch.cuda.is_available()
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    else:

        print(
            "WARNING: CUDA is not available."
        )

    # ========================================================
    # 2. Load metadata
    # ========================================================

    print("\n" + "=" * 70)
    print("LOADING LAV-DF METADATA")
    print("=" * 70)

    metadata_path = os.path.join(
        DATASET_ROOT,
        "metadata.json"
    )

    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as f:

        metadata = json.load(f)

    print(
        "Total metadata:",
        len(metadata)
    )

    # ========================================================
    # 3. Select DEV split
    # ========================================================

    dev_metadata = [
        item
        for item in metadata
        if item["split"] == "dev"
    ]

    print(
        "Total DEV samples:",
        len(dev_metadata)
    )

    # Take first 350 DEV samples
    selected_metadata = dev_metadata[:NUM_SAMPLES]

    print(
        "Selected DEV samples:",
        len(selected_metadata)
    )

    # ========================================================
    # 4. Check existing features
    # ========================================================

    extraction_metadata = []

    existing_count = 0

    for item in selected_metadata:

        feature_file = "_".join(
            str(item["file"])
            .strip()
            .split("/")
        )

        feature_path = os.path.join(
            OUTPUT_DIR,
            feature_file
        )

        feature_path = feature_path.replace(
            ".mp4",
            ".pkl"
        )

        if os.path.isfile(feature_path):

            existing_count += 1

        else:

            extraction_metadata.append(
                item
            )

    print(
        "Already existing:",
        existing_count
    )

    print(
        "Need extraction:",
        len(extraction_metadata)
    )

    if len(extraction_metadata) == 0:

        print(
            "\nAll 350 selected DEV features already exist."
        )

        return

    # ========================================================
    # 5. Load pretrained synchronization model
    # ========================================================

    print("\n" + "=" * 70)
    print("LOADING PRETRAINED SYNCHRONIZATION MODEL")
    print("=" * 70)

    model = (
        ConsistencyPretrainModel
        .load_from_checkpoint(
            CHECKPOINT,
            max_len=FRAME_PADDING
        )
    )

    print(
        "Checkpoint loaded successfully."
    )

    # ========================================================
    # 6. Configure positional encodings
    # ========================================================

    model.audio_encoder.block.pos_embedding = (
        PositionalEncoding(
            192,
            0,
            FRAME_PADDING
        )
    )

    model.audio_consistency.position_embed = (
        PositionalEncoding(
            256,
            0,
            FRAME_PADDING
        )
    )

    model.video_consistency.position_embed = (
        PositionalEncoding(
            256,
            0,
            FRAME_PADDING
        )
    )

    model.cross_modal_consistency.position_embed = (
        PositionalEncoding(
            256,
            0,
            FRAME_PADDING
        )
    )

    model.eval()

    # ========================================================
    # 7. Create LAV-DF dataset object
    # ========================================================

    print("\n" + "=" * 70)
    print("CREATING DEV DATASET")
    print("=" * 70)

    dataset = LavdfInference(
        subset="dev",
        root=DATASET_ROOT,
        frame_padding=FRAME_PADDING,
        fps=FPS,
        img_size=IMG_SIZE,
        pad_to_max_len=True
    )

    # IMPORTANT:
    # The original LavdfInference class does not filter
    # metadata according to split.
    #
    # Therefore we explicitly assign only our selected
    # DEV metadata here.

    dataset.metadata = [
        Metadata(**item)
        for item in extraction_metadata
    ]

    print(
        "Dataset samples for extraction:",
        len(dataset)
    )

    # ========================================================
    # 8. Create DataLoader
    # ========================================================

    collator = BatchCollaterInferencePTLightning(
        True,
        FPS,
        FRAME_PADDING
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collator
    )

    # ========================================================
    # 9. Create Lightning Trainer
    # ========================================================

    print("\n" + "=" * 70)
    print("CREATING TRAINER")
    print("=" * 70)

    trainer = Trainer(
    logger=False,
    enable_checkpointing=False,
    accelerator=(
        "gpu"
        if torch.cuda.is_available()
        else "cpu"
    ),
    devices=1,
    callbacks=[
        SaveFeatures()
    ]
)

    # ========================================================
    # 10. Start extraction
    # ========================================================

    print("\n" + "=" * 70)
    print("STARTING DEV FEATURE EXTRACTION")
    print("=" * 70)

    trainer.predict(
        model=model,
        dataloaders=dataloader
    )

    # ========================================================
    # 11. Finish
    # ========================================================

    print("\n" + "=" * 70)
    print("DEV FEATURE EXTRACTION COMPLETE")
    print("=" * 70)

    print(
        "Features saved inside:"
    )

    print(
        OUTPUT_DIR
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()