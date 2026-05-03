"""
tests/test_model.py
====================
Student ID : 21396816
Tests for src/deepfake_detector/model.py

All tests use pretrained=False — no network download required, fast to run.
"""

from __future__ import annotations

import sys
sys.path.insert(0, ".")

import torch

from src.deepfake_detector.model import build_model, count_parameters

# Expected ResNet18 parameter count (fixed by architecture, not by weights).
# Verify once with check_model.py; thereafter this acts as a regression guard
# that catches accidental architectural changes.
EXPECTED_TOTAL_PARAMS = 11_177_538


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_output_shape():
    """Model must output (N, 2) logits for a standard (N, 3, 224, 224) input."""
    model = build_model(num_classes=2, pretrained=False)
    model.eval()
    out = model(torch.randn(4, 3, 224, 224))
    assert out.shape == (4, 2), (
        f"Expected output shape (4, 2), got {tuple(out.shape)}."
    )


def test_output_is_logits_not_probabilities():
    """
    The model must return raw logits (no softmax).
    Probabilities are constrained to [0, 1] and sum to 1; logits are not.
    CrossEntropyLoss requires logits — softmax inside the loss would double-apply it.
    """
    model = build_model(num_classes=2, pretrained=False)
    model.eval()
    out = model(torch.randn(8, 3, 224, 224))
    row_sums = out.softmax(dim=1).sum(dim=1)
    # Softmax of logits sums to 1; raw output should NOT already sum to 1.
    assert not torch.allclose(out.sum(dim=1), torch.ones(8), atol=1e-3), (
        "Output rows sum to ~1.0 — model may be applying softmax internally. "
        "Remove it; CrossEntropyLoss applies softmax."
    )
    # But softmax of the output should sum to 1 (sanity check on the check).
    assert torch.allclose(row_sums, torch.ones(8), atol=1e-4)


def test_total_parameter_count():
    """Total parameters must match the known ResNet18 count exactly."""
    model = build_model(num_classes=2, pretrained=False)
    total, _ = count_parameters(model)
    assert total == EXPECTED_TOTAL_PARAMS, (
        f"Expected {EXPECTED_TOTAL_PARAMS:,} total params, got {total:,}. "
        "Architectural change detected."
    )


def test_freeze_backbone_reduces_trainable_params():
    """
    freeze_backbone=True must leave fewer trainable parameters than total.
    Only the final fc layer should remain trainable.
    """
    model = build_model(num_classes=2, pretrained=False, freeze_backbone=True)
    total, trainable = count_parameters(model)
    assert trainable < total, (
        f"freeze_backbone=True did not reduce trainable params "
        f"(total={total:,}, trainable={trainable:,})."
    )
    # The head is nn.Linear(512, 2): 512*2 weights + 2 biases = 1026.
    assert trainable == 1_026, (
        f"Expected exactly 1,026 trainable params (fc head only), got {trainable:,}."
    )