from pathlib import Path

import torch
import torch.nn.functional as F


LEARNING_RATE = 3e-4
EPOCHS = 20

if not torch.cuda.is_available():
    raise RuntimeError(
        "Training requires a CUDA-capable GPU."
    )

device = torch.device("cuda")
model = model.to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    betas=(0.9, 0.95),
    weight_decay=0.01,
)

scaler = torch.amp.GradScaler("cuda")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "structured_piano_transformer"
)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_PATH = (
    CHECKPOINT_DIR
    / "best_structured_piano_transformer.pt"
)

LAST_PATH = (
    CHECKPOINT_DIR
    / "last_structured_piano_transformer.pt"
)　

composer_map = {
    "Frédéric Chopin": 0,
    "Franz Schubert": 1,
    "Ludwig van Beethoven": 2,
    "Johann Sebastian Bach": 3,
    "Franz Liszt": 4,
    "OTHER": 5,
}

best_val_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss_sum = 0.0

    for batch_x, batch_y, batch_composer_ids in train_loader:
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

    train_loss = train_loss_sum / len(train_loader)

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
    }

    torch.save(checkpoint, LAST_PATH)

    if is_best:
        torch.save(checkpoint, BEST_PATH)

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"train: {train_loss:.4f} | "
        f"validation: {val_loss:.4f} | "
        f"best: {best_val_loss:.4f}"
    )
