"""
scripts/train.py
================
Student ID : 21396816
Python     : 3.10 or higher
Purpose    : Fine-tune a pretrained ResNet18 on the deepfake-face dataset and
             save the best-performing checkpoint to disk.

Training behaviour
------------------
* AdamW optimiser (lr 1e-4, weight_decay 1e-4) — conservative fine-tune LR
  that preserves pretrained features while allowing the head to specialise.
* CrossEntropyLoss — applies log-softmax internally; model must output raw
  logits (no softmax in model.py).
* Best-model checkpointing on validation accuracy.
* Per-epoch CSV log at results/training_log.csv for curve plotting.

Reproducibility note
--------------------
The script is seeded for reproducibility (default seed 42). Minor run-to-run
variation may occur on Apple Silicon (MPS) due to non-deterministic GPU
kernels. CPU runs are fully deterministic.

Usage
-----
.venv/bin/python scripts/train.py
.venv/bin/python scripts/train.py --epochs 10 --batch-size 64 --lr 3e-4
.venv/bin/python scripts/train.py --freeze-backbone    # ablation: fixed backbone
.venv/bin/python scripts/train.py --no-pretrained      # ablation: random init
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, ".")
from src.deepfake_detector.data import get_dataloaders
from src.deepfake_detector.model import build_model, count_parameters


# ---------------------------------------------------------------------------
# 1. CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ResNet18 deepfake detector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root",       type=Path, default=Path("data/working"))
    parser.add_argument("--output-dir",      type=Path, default=Path("models"))
    parser.add_argument("--results-dir",     type=Path, default=Path("results"))
    parser.add_argument("--epochs",          type=int,  default=5)
    parser.add_argument("--batch-size",      type=int,  default=32)
    parser.add_argument("--lr",              type=float,default=1e-4)
    parser.add_argument("--num-workers",     type=int,  default=4)
    parser.add_argument("--seed",            type=int,  default=42)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Initialise backbone with random weights (ablation).",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze all layers except the final head (ablation).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 2. Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, PyTorch CPU, and (if available) the MPS/CUDA backend.

    Full determinism on Apple Silicon MPS is not guaranteed due to
    non-deterministic GPU kernels. CPU-only runs are fully reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# 3. Device selection
# ---------------------------------------------------------------------------

def select_device() -> torch.device:
    """Return the best available device: MPS → CUDA → CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# 4. Training loop — one epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one full pass over *loader* in training mode.

    Returns
    -------
    (epoch_loss, epoch_acc) — mean cross-entropy loss and top-1 accuracy.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Accumulate weighted loss (correct for variable last-batch size).
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


# ---------------------------------------------------------------------------
# 5. Validation loop
# ---------------------------------------------------------------------------

def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate *model* on *loader* without computing gradients.

    Returns
    -------
    (epoch_loss, epoch_acc) — mean cross-entropy loss and top-1 accuracy.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate data loading, model construction, training, and checkpointing."""
    t_script_start = time.perf_counter()
    args = parse_args()

    # ── Seed & device ────────────────────────────────────────────────────────
    seed_everything(args.seed)
    device = select_device()
    print(f"\n[CONFIG] device={device.type}  seed={args.seed}  "
          f"epochs={args.epochs}  batch_size={args.batch_size}  lr={args.lr}")

    # ── Output directories ───────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    print(f"\n[DATA] Loading from {args.data_root} …")
    loaders = get_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    print("\n[MODEL] Building ResNet18 …")
    model = build_model(
        num_classes=2,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
    )
    model = model.to(device)

    total, trainable = count_parameters(model)
    print(f"  Total params:     {total:>12,}")
    print(f"  Trainable params: {trainable:>12,}")
    if args.freeze_backbone:
        print("  Backbone frozen — only the head will be updated.")
    if args.no_pretrained:
        print("  WARNING: random initialisation — no pretrained weights.")

    # ── Loss & optimiser ─────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )

    # ── CSV log ──────────────────────────────────────────────────────────────
    log_path = args.results_dir / "training_log.csv"
    print(f"\n[LOG] Writing training log → {log_path}")

    # Context manager guarantees the file is flushed and closed even if an
    # exception is raised mid-training — no buffered rows lost on crash.
    with log_path.open("w", newline="", encoding="utf-8") as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow(
            ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "epoch_seconds"]
        )

        # ── Training loop ────────────────────────────────────────────────────
        best_val_acc = 0.0
        best_model_path = args.output_dir / "best_model.pt"

        print(f"\n{'─'*64}")
        print(f"  {'Epoch':<8} {'T-Loss':>8} {'T-Acc':>8} {'V-Loss':>8} {'V-Acc':>8} {'Time':>8}")
        print(f"{'─'*64}")

        for epoch in range(1, args.epochs + 1):
            t0 = time.perf_counter()

            train_loss, train_acc = train_one_epoch(
                model, loaders["train"], criterion, optimizer, device
            )
            val_loss, val_acc = validate(
                model, loaders["val"], criterion, device
            )

            epoch_seconds = time.perf_counter() - t0

            print(
                f"  {epoch:<8} {train_loss:>8.4f} {train_acc:>8.4f} "
                f"{val_loss:>8.4f} {val_acc:>8.4f} {epoch_seconds:>7.1f}s"
            )

            log_writer.writerow([
                epoch,
                f"{train_loss:.6f}", f"{train_acc:.6f}",
                f"{val_loss:.6f}",   f"{val_acc:.6f}",
                f"{epoch_seconds:.1f}",
            ])
            log_file.flush()   # persist row immediately so log survives a crash

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
                # Save a config JSON alongside the weights so the checkpoint is
                # self-describing — the inference script reads this instead of
                # hard-coding architecture assumptions in two places.
                config = {
                    "architecture": "resnet18",
                    "num_classes": 2,
                    "image_size": 224,
                    "imagenet_normalised": True,
                    "class_to_idx": {"fake": 0, "real": 1},
                    "best_val_acc": round(best_val_acc, 6),
                }
                (args.output_dir / "best_model_config.json").write_text(
                    json.dumps(config, indent=2)
                )
                print(f"  ✓ New best val acc {val_acc:.4f} — checkpoint saved.")

    # ── Test-set evaluation (held-out; used only once) ───────────────────────
    # Val accuracy is optimistic — model selection was based on it. Test
    # accuracy is the honest headline number for the Report and README.
    print("\n[TEST] Loading best checkpoint for held-out test evaluation …")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_acc = validate(model, loaders["test"], criterion, device)
    print(f"  Test loss : {test_loss:.4f}")
    print(f"  Test acc  : {test_acc:.4f}")

    # ── Final summary ─────────────────────────────────────────────────────────
    total_time = time.perf_counter() - t_script_start

    print(f"\n{'─'*64}")
    print(f"  Training complete.")
    print(f"  Best val accuracy : {best_val_acc:.4f}")
    print(f"  Test accuracy     : {test_acc:.4f}")
    print(f"  Checkpoint path   : {best_model_path}")
    print(f"  Training log      : {log_path}")
    print(f"  Total time        : {total_time:.1f} s")
    print(f"{'─'*64}\n")


if __name__ == "__main__":
    main()