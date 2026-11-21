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

# FINAL TOTAL TARGETS
TRAIN_TARGET = 2500
DEV_TARGET = 500

BATCH_SIZE = 2
NUM_WORKERS = 0

FRAME_PADDING = 500
FPS = 25
IMG_SIZE = 96


# ============================================================
# SAVE FEATURES CALLBACK
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

            n_frames = int(
                batch["video_frames"][i]
            )

            feature_file_path = str(
                batch["feature_file_path"][i]
            )

            feature_file_path = (
                os.path.splitext(
                    feature_file_path
                )[0] + ".pkl"
            )

            os.makedirs(
                os.path.dirname(feature_file_path),
                exist_ok=True
            )

            feat_json = {}

            feat_json["filepath"] = feature_file_path

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

            feat_json["av_hidden_feat"] = (
                av_hidden_feat[i][:n_frames]
                .detach()
                .cpu()
                .tolist()
            )

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
            # SAFETY CHECK
            # Never overwrite an existing feature
            # ------------------------------------------------

            if os.path.isfile(feature_file_path):

                print(
                    "[SKIP EXISTING]",
                    os.path.basename(
                        feature_file_path
                    )
                )

                continue

            # ------------------------------------------------
            # SAVE
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
                os.path.basename(
                    feature_file_path
                ),
                "| frames:",
                n_frames
            )


# ============================================================
# FEATURE PATH
# ============================================================

def get_feature_path(item):

    feature_file = "_".join(
        str(item["file"])
        .strip()
        .split("/")
    )

    feature_path = os.path.join(
        OUTPUT_DIR,
        feature_file
    )

    feature_path = (
        os.path.splitext(feature_path)[0]
        + ".pkl"
    )

    return feature_path


# ============================================================
# SELECT EXISTING + MISSING SAMPLES
# ============================================================

def select_samples(
    metadata,
    split_name,
    target_count
):

    split_metadata = [
        item
        for item in metadata
        if str(
            item["split"]
        ).lower()
        == split_name.lower()
    ]

    print(
        f"\nAvailable {split_name} metadata:",
        len(split_metadata)
    )

    # --------------------------------------------------------
    # Existing features
    # --------------------------------------------------------

    existing_items = []
    missing_items = []

    for item in split_metadata:

        feature_path = get_feature_path(item)

        if os.path.isfile(feature_path):

            existing_items.append(item)

        else:

            missing_items.append(item)

    print(
        f"Existing {split_name} features:",
        len(existing_items)
    )

    # --------------------------------------------------------
    # We keep ALL existing features first.
    # Then add missing samples until target is reached.
    # --------------------------------------------------------

    if len(existing_items) >= target_count:

        selected = existing_items[:target_count]

    else:

        needed = (
            target_count
            - len(existing_items)
        )

        selected = (
            existing_items
            + missing_items[:needed]
        )

    selected_existing = sum(
        os.path.isfile(
            get_feature_path(item)
        )
        for item in selected
    )

    selected_missing = (
        len(selected)
        - selected_existing
    )

    print(
        f"Selected {split_name}:",
        len(selected)
    )

    print(
        f"Already available:",
        selected_existing
    )

    print(
        f"New extraction required:",
        selected_missing
    )

    return selected


# ============================================================
# EXTRACT ONE SPLIT
# ============================================================

def extract_split(
    model,
    metadata,
    split_name,
    target_count
):

    print("\n" + "=" * 70)
    print(
        f"{split_name.upper()} FEATURE EXTRACTION"
    )
    print("=" * 70)

    selected_metadata = select_samples(
        metadata,
        split_name,
        target_count
    )

    # --------------------------------------------------------
    # ONLY missing samples go to extraction
    # --------------------------------------------------------

    extraction_metadata = []

    for item in selected_metadata:

        feature_path = get_feature_path(item)

        if not os.path.isfile(feature_path):

            extraction_metadata.append(item)

    print(
        "\nNeed to extract:",
        len(extraction_metadata)
    )

    # --------------------------------------------------------
    # Nothing new required
    # --------------------------------------------------------

    if len(extraction_metadata) == 0:

        print(
            f"All {split_name} target features already exist."
        )

        return

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = LavdfInference(
        subset="Train",
        root=DATASET_ROOT,
        frame_padding=FRAME_PADDING,
        fps=FPS,
        img_size=IMG_SIZE,
        pad_to_max_len=True
    )

    dataset.metadata = [
        Metadata(**item)
        for item in extraction_metadata
    ]

    print(
        "Samples going to extraction:",
        len(dataset)
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # START EXTRACTION
    # --------------------------------------------------------

    print(
        f"\nSTARTING {split_name.upper()} EXTRACTION..."
    )

    trainer.predict(
        model=model,
        dataloaders=dataloader
    )

    print(
        f"\n{split_name.upper()} EXTRACTION COMPLETE"
    )


# ============================================================
# COUNT CURRENT FEATURES
# ============================================================

def show_counts():

    files = [
        f
        for f in os.listdir(OUTPUT_DIR)
        if f.endswith(".pkl")
    ]

    train_count = sum(
        f.startswith("train_")
        for f in files
    )

    dev_count = sum(
        f.startswith("dev_")
        for f in files
    )

    test_count = sum(
        f.startswith("test_")
        for f in files
    )

    print("\n" + "=" * 70)
    print("CURRENT FEATURE COUNTS")
    print("=" * 70)

    print("Train:", train_count)
    print("Dev:", dev_count)
    print("Test:", test_count)
    print("Total:", len(files))


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("LAV-DF FINAL FEATURE EXTRACTION")
    print("=" * 70)

    print(
        "\nDataset:",
        DATASET_ROOT
    )

    print(
        "Checkpoint:",
        CHECKPOINT
    )

    print(
        "Output:",
        OUTPUT_DIR
    )

    # --------------------------------------------------------
    # CURRENT COUNT
    # --------------------------------------------------------

    show_counts()

    # --------------------------------------------------------
    # LOAD METADATA
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOADING METADATA")
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

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOADING PRETRAINED MODEL")
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

    # --------------------------------------------------------
    # POSITIONAL ENCODINGS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    extract_split(
        model=model,
        metadata=metadata,
        split_name="train",
        target_count=TRAIN_TARGET
    )

    # --------------------------------------------------------
    # DEV
    # --------------------------------------------------------

    extract_split(
        model=model,
        metadata=metadata,
        split_name="dev",
        target_count=DEV_TARGET
    )

    # --------------------------------------------------------
    # FINAL COUNT
    # --------------------------------------------------------

    show_counts()

    print("\n" + "=" * 70)
    print(
        "FINAL FEATURE EXTRACTION COMPLETE"
    )
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()