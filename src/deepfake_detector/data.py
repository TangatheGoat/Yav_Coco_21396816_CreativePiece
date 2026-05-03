"""
src/deepfake_detector/data.py
==============================
Student ID : 21396816
Python     : 3.10 or higher
Purpose    : Build reproducible, augmented DataLoaders for the deepfake-face
             detection task from a pre-split ImageFolder directory tree.

Expected directory layout (produced by scripts/prepare_data.py):
    data/working/
    ├── train/  real/  fake/
    ├── val/    real/  fake/
    └── test/   real/  fake/

Usage:
    from src.deepfake_detector.data import get_dataloaders, describe_dataset
    loaders = get_dataloaders("data/working", batch_size=32, num_workers=4)
    loaders["train"], loaders["val"], loaders["test"]
"""
from __future__ import annotations
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

# ---------------------------------------------------------------------------
# ImageNet normalisation statistics (standard; used for pre-trained backbones)
# ---------------------------------------------------------------------------
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_dataloaders(
    data_root: str | Path,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
) -> dict[str, DataLoader]:
    """Build and return train, val, and test DataLoaders.

    Parameters
    ----------
    data_root:
        Path to the pre-split dataset root (contains train/, val/, test/).
    image_size:
        Square resolution to which every image is resized before being fed to
        the model. Defaults to 224, matching standard ImageNet pre-training.
    batch_size:
        Number of samples per mini-batch for all three loaders.
    num_workers:
        Sub-processes used for data loading. Set to 0 to load on the main
        process (useful for debugging on macOS where multiprocessing can
        cause issues with some environments).

    Returns
    -------
    dict with keys "train", "val", "test", each holding a DataLoader.
    """
    data_root = Path(data_root)
    _validate_root(data_root)

    # ── Transforms ──────────────────────────────────────────────────────────
    # Training: light augmentation helps the model generalise to unseen
    # compression artefacts and minor geometric variation in real-world faces.
    # Only horizontal flip is used; vertical flip and strong colour jitter
    # would be unrealistic for face images and could hurt performance.
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])

    # Evaluation: deterministic pipeline — no random ops so metrics are
    # reproducible across runs.
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])

    # ── Datasets ─────────────────────────────────────────────────────────────
    # ImageFolder infers class labels from sub-folder names (fake=0, real=1
    # — alphabetical order). The mapping is printed by describe_dataset() so
    # it is always visible in experiment logs.
    datasets = {
        "train": ImageFolder(data_root / "train", transform=train_transform),
        "val":   ImageFolder(data_root / "val",   transform=eval_transform),
        "test":  ImageFolder(data_root / "test",  transform=eval_transform),
    }

    # ── DataLoaders ──────────────────────────────────────────────────────────
    # drop_last=True for train: avoids a partial final batch that would
    # distort per-batch loss statistics, especially at the end of an epoch.
    # pin_memory=False: safe default; set True only when using a CUDA GPU
    # and verifying that host RAM is not the bottleneck.
    loaders: dict[str, DataLoader] = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=False,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=False,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=False,
        ),
    }

    return loaders


def describe_dataset(data_root: str | Path, batch_size: int = 32) -> None:
    """Print a human-readable summary of the dataset splits, then pull one
    real training batch to confirm the full loader pipeline is healthy.

    Two views are produced:
      1. Metadata view — split sizes, class count, class→index mapping, and
         approximate batch count. Rebuilt from ImageFolder on disk so it
         works before any DataLoader is constructed.
      2. End-to-end view — one batch from the train loader is iterated to
         verify that transforms produce the expected tensor shape, dtype, and
         normalised pixel range. Catches transform misconfiguration, missing
         normalisation, and label-type bugs that the metadata view cannot see.

    Parameters
    ----------
    data_root:
        Path to the pre-split dataset root (same value passed to get_dataloaders).
    batch_size:
        Used to compute approximate batch counts and to pull the sanity batch.
    """
    if batch_size <= 0:
        raise ValueError(
            f"batch_size must be a positive integer, got {batch_size}."
        )

    data_root = Path(data_root)
    _validate_root(data_root)

    splits = ("train", "val", "test")

    print("\n" + "=" * 56)
    print("Dataset summary")
    print("=" * 56)

    for split in splits:
        ds = ImageFolder(data_root / split)
        n_batches = len(ds) // batch_size   # mirrors drop_last behaviour
        print(
            f"  {split:<6} {len(ds):>6,} images  |  "
            f"{len(ds.classes)} classes  |  "
            f"~{n_batches} batches @ bs={batch_size}"
        )
        if split == "train":
            # Print the class→index mapping once; it is the same across splits
            # and determines which integer label maps to 'fake' vs 'real'.
            print(f"         class map: {ds.class_to_idx}")

    print("=" * 56)

    # ── End-to-end sanity check ──────────────────────────────────────────────
    # Pull one batch through the full train loader (transforms included) and
    # assert the expected shape, dtype, and pixel range. num_workers=0 avoids
    # macOS multiprocessing issues in lightweight check scripts.
    print("\nEnd-to-end batch check (train loader, num_workers=0) …")
    loaders = get_dataloaders(data_root, batch_size=batch_size, num_workers=0)
    images, labels = next(iter(loaders["train"]))

    print(f"  images.shape : {tuple(images.shape)}")           # (B, 3, H, W)
    print(f"  images.dtype : {images.dtype}")                  # torch.float32
    print(f"  labels.shape : {tuple(labels.shape)}")           # (B,)
    print(f"  labels.dtype : {labels.dtype}")                  # torch.int64
    print(f"  pixel range  : {images.min().item():.3f} to {images.max().item():.3f}")
    print(f"  label sample : {labels.tolist()[:8]}")

    # Assertions — raise immediately rather than producing silent wrong results.
    assert images.ndim == 4,             "Expected 4-D image tensor (B, C, H, W)"
    assert images.shape[1] == 3,         "Expected 3 colour channels"
    assert str(images.dtype) == "torch.float32", "Expected float32 images"
    assert str(labels.dtype) == "torch.int64",   "Expected int64 labels"
    assert images.min().item() < 0,      "Pixel min ≥ 0 — normalisation may be missing"
    assert images.max().item() > 1,      "Pixel max ≤ 1 — normalisation may be missing"

    print("  ✓ All assertions passed.\n" + "=" * 56 + "\n")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_root(data_root: Path) -> None:
    """Exit with a clear message if any expected split folder is missing."""
    if not data_root.is_dir():
        raise FileNotFoundError(
            f"[data.py] Dataset root not found: {data_root}\n"
            f"          Run scripts/prepare_data.py first."
        )
    for split in ("train", "val", "test"):
        split_dir = data_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"[data.py] Expected split folder missing: {split_dir}\n"
                f"          Re-run scripts/prepare_data.py to rebuild the dataset."
            )