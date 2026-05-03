"""
src/deepfake_detector/model.py
================================
Student ID : 21396816
Python     : 3.10 or higher
Purpose    : Construct a ResNet18-based binary classifier for deepfake-face
             detection via transfer learning.

The pretrained backbone supplies general visual features (edges, textures,
facial structure) learned from 1.3 M ImageNet images. Only the final
fully-connected head is replaced and retrained for the real/fake task,
making the model viable on a training set of ~7,000 images.
"""

from __future__ import annotations

from torch import nn
from torchvision import models
from torchvision.models import ResNet18_Weights


def build_model(
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Build and return a ResNet18 binary classifier.

    Parameters
    ----------
    num_classes:
        Number of output logits. Defaults to 2 (real vs fake).
    pretrained:
        If True, initialise the backbone with ImageNet weights
        (ResNet18_Weights.IMAGENET1K_V1). If False, random initialisation
        — useful only for debugging or ablation experiments.
    freeze_backbone:
        If True, set requires_grad=False on every parameter except the new
        classification head. The backbone acts as a fixed feature extractor;
        only the head is updated during training. Reduces training time and
        memory use, but lowers the accuracy ceiling. Leave False unless
        training data is very small (< ~500 images).

    Returns
    -------
    nn.Module
        ResNet18 with its original 1,000-class head replaced by a new
        nn.Linear(in_features, num_classes) layer.
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    # ── Head replacement ─────────────────────────────────────────────────────
    # model.fc is the original 1,000-class output layer.
    # in_features (512 for ResNet18) must be read before the layer is replaced.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # ── Optional backbone freeze ──────────────────────────────────────────────
    if freeze_backbone:
        for name, param in model.named_parameters():
            # "fc." (with dot) avoids matching hypothetical layers named
            # "fc1", "fc_extra", etc. Safe no-op for current ResNet18.
            if not name.startswith("fc."):
                param.requires_grad = False

    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total_params, trainable_params) for *model*.

    Iterates all parameter tensors; sums numel() for every tensor and again
    only for those with requires_grad=True. Useful for logging model capacity
    and verifying that freeze_backbone is working as expected.
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable