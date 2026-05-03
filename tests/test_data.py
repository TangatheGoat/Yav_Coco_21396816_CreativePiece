"""
tests/test_data.py
==================
Student ID : 21396816
Tests for src/deepfake_detector/data.py

Prerequisites
-------------
data/working/ must exist (run scripts/prepare_data.py first).
See README — "Running tests" section.
"""

from __future__ import annotations

import sys
sys.path.insert(0, ".")

from pathlib import Path

import pytest
import torch

from src.deepfake_detector.data import _validate_root, get_dataloaders

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_ROOT   = Path("data/working")
BATCH_SIZE  = 8     # small — tests don't need full-size batches
NUM_WORKERS = 0     # avoid multiprocessing in test environment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def loaders():
    """Build dataloaders once per module — expensive to construct each test."""
    if not DATA_ROOT.exists():
        pytest.skip(
            f"data/working not found — run scripts/prepare_data.py first."
        )
    return get_dataloaders(DATA_ROOT, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_loaders_return_three_splits(loaders):
    """get_dataloaders must return exactly the keys train, val, and test."""
    assert set(loaders.keys()) == {"train", "val", "test"}


def test_all_splits_non_empty(loaders):
    """Every split must contain at least one batch."""
    for split, loader in loaders.items():
        assert len(loader) > 0, f"Split '{split}' produced an empty loader."


def test_train_batch_shape_and_dtype(loaders):
    """A training batch must be shape (B, 3, 224, 224) and float32."""
    images, labels = next(iter(loaders["train"]))
    assert images.ndim == 4,                   "Expected 4-D image tensor"
    assert images.shape[1] == 3,               "Expected 3 colour channels"
    assert images.shape[2] == 224,             "Expected height 224"
    assert images.shape[3] == 224,             "Expected width 224"
    assert images.dtype  == torch.float32,     "Expected float32 images"
    assert labels.dtype  == torch.int64,       "Expected int64 labels"
    assert images.shape[0] == labels.shape[0], "Batch size mismatch between images and labels"


def test_normalisation_applied(loaders):
    """
    After ImageNet normalisation, pixel values must extend below 0 and above 1.
    If this fails, normalisation was omitted from the transform pipeline.
    """
    images, _ = next(iter(loaders["train"]))
    assert images.min().item() < 0.0, (
        f"Pixel min is {images.min().item():.4f} — normalisation may be missing. "
        "Expected values below 0 after ImageNet mean subtraction."
    )
    assert images.max().item() > 1.0, (
        f"Pixel max is {images.max().item():.4f} — normalisation may be missing. "
        "Expected values above 1 after ImageNet std division."
    )


def test_validate_root_raises_on_missing_path():
    """_validate_root must raise FileNotFoundError for a non-existent directory."""
    with pytest.raises(FileNotFoundError, match="Dataset root not found"):
        _validate_root(Path("/nonexistent/path/that/cannot/exist"))