from __future__ import annotations
import sys; sys.path.insert(0, ".")
import torch
from src.deepfake_detector.model import build_model, count_parameters

model = build_model(num_classes=2, pretrained=True, freeze_backbone=False)
total, trainable = count_parameters(model)
print(f"Total params:     {total:>12,}")
print(f"Trainable params: {trainable:>12,}")

fake_batch = torch.randn(4, 3, 224, 224)
out = model(fake_batch)
print(f"Output shape:     {tuple(out.shape)}")
print(f"Output sample:    {[round(v, 4) for v in out[0].tolist()]}")