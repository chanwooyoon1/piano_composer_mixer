import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import json
import pandas as pd

from music_model import ComposerMusicTransformer
from structured_tokenizer import (
    PAD_TOKEN,
    BAR_TOKEN,
    CHORD_START,
    PITCH_START,
    DURATION_START,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
TOKEN_DIR = DATA_DIR / "structured_token_data"
MANIFEST_PATH = DATA_DIR / "structured_token_manifest.csv"
CONFIG_PATH = DATA_DIR / "structured_tokenizer_config.json"

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "structured_piano_transformer"
)

BASE_BEST_PATH = (
    CHECKPOINT_DIR
    / "best_structured_piano_transformer.pt"
)

AUG_BEST_PATH = (
    CHECKPOINT_DIR
    / "best_structured_piano_transformer_augmented.pt"
)

AUG_LAST_PATH = (
    CHECKPOINT_DIR
    / "last_structured_piano_transformer_augmented.pt"
)


if not torch.cuda.is_available():
    raise RuntimeError(
        "Training requires a CUDA-capable GPU."
    )

device = torch.device("cuda")

BATCH_SIZE = 4

with open(CONFIG_PATH, encoding="utf-8") as file:
    tokenizer_config = json.load(file)

manifest = pd.read_csv(MANIFEST_PATH)

train_df = manifest[
    manifest["split"] == "train"
].copy()

val_df = manifest[
    manifest["split"] == "validation"
].copy()



if not BASE_BEST_PATH.exists():
    raise FileNotFoundError(
        f"Stage 1 checkpoint not found: {BASE_BEST_PATH}\n"
        "Run train1.py before running train2.py."
    )

base_checkpoint = torch.load(
    BASE_BEST_PATH,
    map_location=device,
    weights_only=False,
)

VOCAB_SIZE = base_checkpoint["vocab_size"]
NUM_COMPOSERS = base_checkpoint["num_composers"]
CONTEXT_LENGTH = base_checkpoint["context_length"]
D_MODEL = base_checkpoint["d_model"]
NUM_HEADS = base_checkpoint["num_heads"]
NUM_LAYERS = base_checkpoint["num_layers"]
FFN_DIM = base_checkpoint["ffn_dim"]
DROPOUT = base_checkpoint["dropout"]
composer_map = base_checkpoint["composer_map"]

model = ComposerMusicTransformer(
    vocab_size=VOCAB_SIZE,
    num_composers=NUM_COMPOSERS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    ffn_dim=FFN_DIM,
    dropout=DROPOUT,
).to(device)

model.lm_head.weight = model.token_embedding.weight

model.load_state_dict(
    base_checkpoint["model_state_dict"]
)

class TransposedStructuredDataset(Dataset):
    def __init__(self, dataframe, start_probability=0.30, max_transpose=0):
        self.df = dataframe.reset_index(drop=True)
        self.start_probability = start_probability
        self.max_transpose = max_transpose

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        token_path = TOKEN_DIR / Path(row["token_path"]).name
        tokens = np.load(token_path).astype(np.int64)

        max_start = max(0, len(tokens) - CONTEXT_LENGTH - 1)

        if max_start == 0 or np.random.random() < self.start_probability:
            start = 0
        else:
            bar_starts = np.flatnonzero(tokens == BAR_TOKEN)
            valid_starts = bar_starts[bar_starts <= max_start]

            if len(valid_starts) > 0:
                start = int(np.random.choice(valid_starts))
            else:
                start = np.random.randint(0, max_start + 1)

        chunk = tokens[start : start + CONTEXT_LENGTH + 1].copy()
        if self.max_transpose > 0:
            pitch_mask = (
                (chunk >= PITCH_START)
                & (chunk < DURATION_START)
            )

            if np.any(pitch_mask):
                pitch_indices = chunk[pitch_mask] - PITCH_START

                min_shift = max(
                    -self.max_transpose,
                    -int(pitch_indices.min()),
                )
                max_shift = min(
                    self.max_transpose,
                    87 - int(pitch_indices.max()),
                )

                shift = random.randint(min_shift, max_shift)

                chunk[pitch_mask] += shift

                chord_mask = (
                    (chunk >= CHORD_START)
                    & (chunk < CHORD_START + 24)
                )

                chord_ids = chunk[chord_mask] - CHORD_START
                qualities = chord_ids // 12
                roots = chord_ids % 12

                chunk[chord_mask] = (
                    CHORD_START
                    + qualities * 12
                    + (roots + shift) % 12
                )

        if len(chunk) < CONTEXT_LENGTH + 1:
            chunk = np.pad(
                chunk,
                (0, CONTEXT_LENGTH + 1 - len(chunk)),
                constant_values=PAD_TOKEN,
            )

        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])

        return x, y, torch.tensor(int(row["composer_id"]))

train_dataset_aug = TransposedStructuredDataset(
    train_df,
    start_probability=0.30,
    max_transpose=5,
)

val_dataset = TransposedStructuredDataset(
    val_df,
    start_probability=0.30,
    max_transpose=0,
)

train_loader_aug = DataLoader(
    train_dataset_aug,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.95),
    weight_decay=0.01,
)

optimizer.load_state_dict(
    base_checkpoint["optimizer_state_dict"]
)

for state in optimizer.state.values():
    for key, value in state.items():
        if torch.is_tensor(value):
            state[key] = value.to(device)

for group in optimizer.param_groups:
    group["lr"] = 1e-4

scaler = torch.amp.GradScaler("cuda")
scaler.load_state_dict(base_checkpoint["scaler_state_dict"])


best_val_loss = base_checkpoint["best_val_loss"]
START_EPOCH = base_checkpoint["epoch"] + 1 
TARGET_EPOCH = 34 
print(
    f"From epoch {START_EPOCH} to {TARGET_EPOCH}"
)
print(f"Starting best validation loss: {best_val_loss:.4f}")

for epoch in range(START_EPOCH, TARGET_EPOCH + 1):
    model.train()
    train_loss_sum = 0.0

    for batch_x, batch_y, batch_composer_ids in train_loader_aug:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        batch_composer_ids = batch_composer_ids.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):
            logits = model(batch_x, batch_composer_ids)

            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                batch_y.reshape(-1),
                ignore_index=PAD_TOKEN,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        scaler.step(optimizer)
        scaler.update()

        train_loss_sum += loss.item()

    train_loss = train_loss_sum / len(train_loader_aug)

    model.eval()
    val_loss_sum = 0.0

    with torch.inference_mode():
        for batch_x, batch_y, batch_composer_ids in val_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_composer_ids = batch_composer_ids.to(
                device,
                non_blocking=True,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                logits = model(batch_x, batch_composer_ids)

                loss = F.cross_entropy(
                    logits.reshape(-1, VOCAB_SIZE),
                    batch_y.reshape(-1),
                    ignore_index=PAD_TOKEN,
                )

            val_loss_sum += loss.item()

    val_loss = val_loss_sum / len(val_loader)
    is_best = val_loss < best_val_loss

    if is_best:
        best_val_loss = val_loss

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_loss": best_val_loss,
        "vocab_size": VOCAB_SIZE,
        "num_composers": NUM_COMPOSERS,
        "context_length": CONTEXT_LENGTH,
        "d_model": D_MODEL,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "ffn_dim": FFN_DIM,
        "dropout": DROPOUT,
        "tokenizer_format": tokenizer_config["format"],
        "composer_map": composer_map,
        "augmentation": "pitch_transpose_plus_matching_chord_root",
    }

    torch.save(checkpoint, AUG_LAST_PATH)

    if is_best:
        torch.save(checkpoint, AUG_BEST_PATH)

    print(
        f"augmented {epoch:02d}/{TARGET_EPOCH} | "
        f"train: {train_loss:.4f} | "
        f"validation: {val_loss:.4f} | "
        f"best: {best_val_loss:.4f}"
    )
