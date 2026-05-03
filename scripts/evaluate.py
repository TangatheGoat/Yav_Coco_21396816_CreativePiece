"""
scripts/evaluate.py
====================
Student ID : 21396816
Python     : 3.10 or higher
Purpose    : Evaluate the best checkpoint on the held-out test set and write
             metrics, plots, and per-sample predictions to results/.

Outputs
-------
results/evaluation_metrics.json   — accuracy, precision, recall, F1, AUC
results/confusion_matrix.png      — heatmap (rows=true, cols=predicted)
results/roc_curve.png             — ROC curve with AUC annotation
results/per_sample_predictions.csv — one row per test image

Usage
-----
.venv/bin/python scripts/evaluate.py
.venv/bin/python scripts/evaluate.py --checkpoint models/best_model.pt \
                                      --data-root  data/working \
                                      --results-dir results
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on headless servers
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

sys.path.insert(0, ".")
from src.deepfake_detector.data import get_dataloaders
from src.deepfake_detector.model import build_model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate best checkpoint on the held-out test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",  type=Path, default=Path("models/best_model.pt"),
    )
    parser.add_argument(
        "--config",      type=Path, default=Path("models/best_model_config.json"),
    )
    parser.add_argument(
        "--data-root",   type=Path, default=Path("data/working"),
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results"),
    )
    parser.add_argument(
        "--batch-size",  type=int,  default=32,
    )
    parser.add_argument(
        "--num-workers", type=int,  default=4,
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def select_device() -> torch.device:
    """Return best available device: MPS → CUDA → CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Collection loop
# ---------------------------------------------------------------------------

def collect_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int], list[float]]:
    """Run the model over *loader* and collect ground-truth and predictions.

    Returns
    -------
    y_true   — integer ground-truth labels (0=fake, 1=real)
    y_pred   — integer predicted labels (argmax of logits)
    y_scores — P(fake) probability for each sample (used for ROC / AUC)
    """
    model.eval()
    y_true:   list[int]   = []
    y_pred:   list[int]   = []
    y_scores: list[float] = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            logits = model(inputs)                         # (B, 2)
            probs  = F.softmax(logits, dim=1).cpu()        # (B, 2)
            preds  = probs.argmax(dim=1)                   # (B,)

            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())
            y_scores.extend(probs[:, 0].tolist())          # P(fake) = class 0

    return y_true, y_pred, y_scores


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    out_path: Path,
) -> None:
    """Save a labelled confusion matrix heatmap to *out_path*."""
    cm     = confusion_matrix(y_true, y_pred)
    labels = ["fake", "real"]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion matrix — test set",
    )

    # Annotate each cell with its count.
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(
    y_true: list[int],
    y_scores: list[float],
    out_path: Path,
) -> float:
    """Save a ROC curve plot to *out_path* and return the AUC score."""
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=0)   # pos_label=0 → fake
    roc_auc     = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="steelblue", lw=2,
            label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--",
            label="Random classifier")
    ax.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.05),
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="ROC curve — test set",
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return roc_auc


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_metrics_json(metrics: dict, out_path: Path) -> None:
    """Serialise *metrics* dict to pretty-printed JSON."""
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def write_per_sample_csv(
    loader: torch.utils.data.DataLoader,
    y_true: list[int],
    y_pred: list[int],
    y_scores: list[float],
    out_path: Path,
) -> None:
    """Write one CSV row per test sample: path, true label, pred label, P(fake)."""
    # ImageFolder stores file paths in .samples as (path, class_idx) tuples.
    sample_paths = [Path(s[0]) for s in loader.dataset.samples]
    class_names  = {v: k for k, v in loader.dataset.class_to_idx.items()}

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "true_label", "predicted_label",
                         "probability_fake", "correct"])
        for path, true, pred, score in zip(sample_paths, y_true, y_pred, y_scores):
            writer.writerow([
                path.name,
                class_names[true],
                class_names[pred],
                f"{score:.6f}",
                int(true == pred),
            ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Load checkpoint, collect predictions, compute metrics, write outputs."""
    args = parse_args()

    # ── Guards ───────────────────────────────────────────────────────────────
    if not args.checkpoint.exists():
        sys.exit(
            f"[ERROR] Checkpoint not found: {args.checkpoint}\n"
            f"        Run scripts/train.py first."
        )
    if not args.config.exists():
        sys.exit(
            f"[ERROR] Config not found: {args.config}\n"
            f"        Run scripts/train.py first."
        )

    args.results_dir.mkdir(parents=True, exist_ok=True)

    # ── Device & model ───────────────────────────────────────────────────────
    device = select_device()
    print(f"[INFO] Device     : {device.type}")
    print(f"[INFO] Checkpoint : {args.checkpoint}")

    model_config = json.loads(args.config.read_text(encoding="utf-8"))
    model = build_model(num_classes=model_config["num_classes"], pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    model.eval()

    # ── Data ─────────────────────────────────────────────────────────────────
    loaders = get_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    test_loader = loaders["test"]
    n_samples   = len(test_loader.dataset)
    print(f"[INFO] Test set   : {n_samples:,} samples")

    # ── Collect predictions ──────────────────────────────────────────────────
    y_true, y_pred, y_scores = collect_predictions(model, test_loader, device)

    # ── Metrics ──────────────────────────────────────────────────────────────
    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1        = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # ── Plots ────────────────────────────────────────────────────────────────
    cm_path  = args.results_dir / "confusion_matrix.png"
    roc_path = args.results_dir / "roc_curve.png"

    plot_confusion_matrix(y_true, y_pred, cm_path)
    roc_auc = plot_roc_curve(y_true, y_scores, roc_path)

    # ── JSON output ──────────────────────────────────────────────────────────
    metrics = {
        "checkpoint":  str(args.checkpoint),
        "test_samples": n_samples,
        "accuracy":    round(accuracy,  4),
        "precision":   round(precision, 4),
        "recall":      round(recall,    4),
        "f1":          round(f1,        4),
        "auc":         round(roc_auc,   4),
        "classification_report": classification_report(
            y_true, y_pred,
            target_names=["fake", "real"],
            output_dict=True,
            zero_division=0,
        ),
    }
    write_metrics_json(metrics, args.results_dir / "evaluation_metrics.json")

    # ── Per-sample CSV ───────────────────────────────────────────────────────
    write_per_sample_csv(
        test_loader, y_true, y_pred, y_scores,
        args.results_dir / "per_sample_predictions.csv",
    )

    # ── Print summary ────────────────────────────────────────────────────────
    cm     = confusion_matrix(y_true, y_pred)
    labels = ["fake", "real"]

    print(f"\n{'='*56}")
    print("Test-set evaluation")
    print(f"{'='*56}")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1       : {f1:.4f}")
    print(f"  AUC      : {roc_auc:.4f}")
    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    header = "  " + " " * 8 + "  ".join(f"{l:>5}" for l in labels)
    print(header)
    for i, row_label in enumerate(labels):
        row = "  " + f"{row_label:>6}  " + "  ".join(f"{cm[i, j]:>5,}" for j in range(len(labels)))
        print(row)
    print(f"\n  Plots written to: {args.results_dir}")
    print(f"{'='*56}\n")


if __name__ == "__main__":
    main()