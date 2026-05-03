from __future__ import annotations
from pathlib import Path
import sys; sys.path.insert(0, ".")
from src.deepfake_detector.data import describe_dataset
describe_dataset(Path("data/working"), batch_size=32)