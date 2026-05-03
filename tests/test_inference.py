"""
tests/test_inference.py
========================
Student ID : 21396816
Tests for src/deepfake_detector/inference.py and scripts/run_inference.py

The smoke test (test_smoke_run_inference) requires:
  - models/best_model.pt and models/best_model_config.json (run train.py first)
  - data/sample/test_clip.mov (place a short sample clip there)
Tests that need these files are skipped automatically if they are absent.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, ".")

import pytest
import torch

from src.deepfake_detector.inference import predict_video
from src.deepfake_detector.model import build_model

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SAMPLE_CLIP      = Path("data/sample/test_clip.mov")
CHECKPOINT       = Path("models/best_model.pt")
MODEL_CONFIG     = Path("models/best_model_config.json")
SMOKE_OUTPUT     = Path("/tmp/test_pred.json")
PYTHON           = sys.executable   # uses the active venv interpreter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _untrained_model_in_eval() -> torch.nn.Module:
    """Build a minimal model in eval mode — no weights download needed."""
    model = build_model(num_classes=2, pretrained=False)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Tests — predict_video contract
# ---------------------------------------------------------------------------

def test_predict_video_raises_on_train_mode():
    """predict_video must raise RuntimeError if model is in training mode."""
    model = build_model(num_classes=2, pretrained=False)
    # Deliberately do NOT call model.eval()
    assert model.training, "Pre-condition: model should start in training mode."

    with pytest.raises(RuntimeError, match="eval mode"):
        predict_video(
            video_path=SAMPLE_CLIP if SAMPLE_CLIP.exists() else Path("dummy.mp4"),
            model=model,
            device=torch.device("cpu"),
        )


def test_predict_video_raises_on_missing_video():
    """predict_video must raise FileNotFoundError for a non-existent path."""
    model = _untrained_model_in_eval()
    with pytest.raises(FileNotFoundError):
        predict_video(
            video_path=Path("/nonexistent/clip_that_cannot_exist.mp4"),
            model=model,
            device=torch.device("cpu"),
        )


def test_predict_video_output_schema():
    """
    predict_video must return a dict with all required top-level keys and the
    correct nested structure — even when no faces are detected in the clip.
    This test runs without a real video by checking the error path produces
    the correct exception type before the schema is even reached; the schema
    is verified below via the smoke test if the clip is available.

    If the sample clip is present, runs the full pipeline and validates schema.
    """
    if not SAMPLE_CLIP.exists():
        pytest.skip(
            f"Sample clip not found: {SAMPLE_CLIP} — "
            "place a short .mov/.mp4 there to enable this test."
        )

    model = _untrained_model_in_eval()
    result = predict_video(
        video_path=SAMPLE_CLIP,
        model=model,
        device=torch.device("cpu"),
        frame_step=60,      # sparse sampling — fast, not accurate (untrained model)
    )

    # Top-level keys
    assert "video_level" in result, "Missing key: video_level"
    assert "frame_level" in result, "Missing key: frame_level"
    assert "summary"     in result, "Missing key: summary"

    # video_level sub-keys
    vl = result["video_level"]
    assert "label"            in vl, "Missing key: video_level.label"
    assert "probability_fake" in vl, "Missing key: video_level.probability_fake"
    assert "probability_real" in vl, "Missing key: video_level.probability_real"
    assert vl["label"] in ("REAL", "FAKE"), (
        f"video_level.label must be 'REAL' or 'FAKE', got {vl['label']!r}"
    )

    # summary sub-keys
    sm = result["summary"]
    assert "frames_sampled"       in sm, "Missing key: summary.frames_sampled"
    assert "frames_with_faces"    in sm, "Missing key: summary.frames_with_faces"
    assert "frames_without_faces" in sm, "Missing key: summary.frames_without_faces"
    assert sm["frames_sampled"] == sm["frames_with_faces"] + sm["frames_without_faces"]

    # frame_level is a list
    assert isinstance(result["frame_level"], list), "frame_level must be a list"


# ---------------------------------------------------------------------------
# Smoke test — full end-to-end via subprocess
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not SAMPLE_CLIP.exists() or not CHECKPOINT.exists() or not MODEL_CONFIG.exists(),
    reason=(
        "Smoke test requires data/sample/test_clip.mov, models/best_model.pt, "
        "and models/best_model_config.json — run train.py and place a sample clip first."
    ),
)
def test_smoke_run_inference():
    """
    End-to-end smoke test: invoke run_inference.py as a subprocess, confirm it
    exits with code 0, writes a valid JSON file, and produces a label in
    ('REAL', 'FAKE').

    Uses the trained checkpoint — this is the only test that does so.
    """
    cmd = [
        PYTHON, "scripts/run_inference.py",
        "--input",      str(SAMPLE_CLIP),
        "--out",        str(SMOKE_OUTPUT),
        "--checkpoint", str(CHECKPOINT),
        "--config",     str(MODEL_CONFIG),
        "--frame-step", "30",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    assert proc.returncode == 0, (
        f"run_inference.py exited with code {proc.returncode}.\n"
        f"stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )

    assert SMOKE_OUTPUT.exists(), (
        f"Expected output JSON at {SMOKE_OUTPUT} — file was not created."
    )

    output = json.loads(SMOKE_OUTPUT.read_text(encoding="utf-8"))

    assert "video_level" in output, "JSON missing key: video_level"
    assert output["video_level"]["label"] in ("REAL", "FAKE"), (
        f"Unexpected label: {output['video_level']['label']!r}"
    )
    assert "schema_version" in output, "JSON missing key: schema_version"
    assert "summary" in output,        "JSON missing key: summary"