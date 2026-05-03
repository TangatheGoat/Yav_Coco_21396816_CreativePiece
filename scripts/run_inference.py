"""
scripts/run_inference.py
=========================
Student ID : 21396816
Python     : 3.10 or higher
Purpose    : CLI wrapper around src/deepfake_detector/inference.predict_video.
             Loads the model and config, validates all inputs defensively,
             runs inference, and writes the result as a JSON file.

Usage
-----
.venv/bin/python scripts/run_inference.py --input clip.mp4 --out result.json
.venv/bin/python scripts/run_inference.py \\
    --input      clip.mp4 \\
    --checkpoint models/best_model.pt \\
    --config     models/best_model_config.json \\
    --out        results/prediction.json \\
    --frame-step 15 \\
    --threshold  0.5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, ".")
from src.deepfake_detector.inference import predict_video
from src.deepfake_detector.model import build_model

# File-size sanity cap — prevents accidentally passing a huge file or /dev/zero.
_MAX_INPUT_BYTES = 500 * 1024 * 1024   # 500 MB
_ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run deepfake detection on a video clip.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Path to the input video clip (.mp4 / .mov / .avi).",
    )
    parser.add_argument(
        "--out", type=Path, required=True,
        help="Destination path for the JSON result file.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("models/best_model.pt"),
        help="Path to the saved model state_dict (.pt).",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("models/best_model_config.json"),
        help="Path to best_model_config.json produced by train.py.",
    )
    parser.add_argument(
        "--frame-step", type=int, default=30,
        help="Sample one frame every N frames (30 ≈ 1 fps at 30 fps footage).",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Minimum mean P(fake) to classify a video as FAKE.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_inputs(args: argparse.Namespace) -> None:
    """Exit with a clear message if any required file is missing or invalid.

    Validation is centralised here so main() stays readable and each error
    message names the exact flag that caused it.
    """
    # ── Input video ──────────────────────────────────────────────────────────
    if not args.input.exists():
        sys.exit(f"[ERROR] --input not found: {args.input}")

    ext = args.input.suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        sys.exit(
            f"[ERROR] --input has unsupported extension '{ext}'.\n"
            f"        Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}"
        )

    file_bytes = args.input.stat().st_size
    if file_bytes > _MAX_INPUT_BYTES:
        mb = file_bytes / (1024 * 1024)
        sys.exit(
            f"[ERROR] --input is {mb:.0f} MB; maximum allowed is "
            f"{_MAX_INPUT_BYTES // (1024 * 1024)} MB.\n"
            f"        Use a shorter clip or raise _MAX_INPUT_BYTES in the script."
        )

    # ── Model files ──────────────────────────────────────────────────────────
    if not args.checkpoint.exists():
        sys.exit(
            f"[ERROR] --checkpoint not found: {args.checkpoint}\n"
            f"        Run scripts/train.py first."
        )
    if not args.config.exists():
        sys.exit(
            f"[ERROR] --config not found: {args.config}\n"
            f"        Run scripts/train.py first."
        )

    # ── Numeric args ─────────────────────────────────────────────────────────
    if args.frame_step < 1:
        sys.exit(f"[ERROR] --frame-step must be >= 1, got {args.frame_step}.")
    if not (0.0 < args.threshold < 1.0):
        sys.exit(
            f"[ERROR] --threshold must be in (0.0, 1.0), got {args.threshold}."
        )


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def select_device() -> torch.device:
    """Return the best available device: MPS -> CUDA -> CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Validate inputs, load model, run inference, write JSON result."""
    args = parse_args()
    validate_inputs(args)

    # ── Device & model ───────────────────────────────────────────────────────
    device = select_device()
    print(f"[INFO] Device     : {device.type}")

    model_config = json.loads(args.config.read_text(encoding="utf-8"))
    print(f"[INFO] Checkpoint : {args.checkpoint}")

    model = build_model(num_classes=model_config["num_classes"], pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    model.eval()   # required before predict_video — enforced by RuntimeError there

    # ── Inference ────────────────────────────────────────────────────────────
    print(f"[INFO] Analysing  : {args.input}  (frame_step={args.frame_step})")
    t0 = time.perf_counter()

    prediction = predict_video(
        video_path=args.input,
        model=model,
        device=device,
        frame_step=args.frame_step,
        fake_threshold=args.threshold,
    )

    elapsed = time.perf_counter() - t0

    # ── Attach provenance metadata ───────────────────────────────────────────
    # The library function returns only what it computed. Metadata about how
    # inference was invoked belongs here at the boundary.
    result = {
        "schema_version":    "1.0",
        "input":             str(args.input),
        "model_checkpoint":  str(args.checkpoint),
        "model_config":      str(args.config),
        "device":            device.type,
        "inference_seconds": round(elapsed, 2),
        **prediction,
    }

    # ── Write JSON ───────────────────────────────────────────────────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # ── Human summary ────────────────────────────────────────────────────────
    v  = result["video_level"]
    sm = result["summary"]
    print(
        f"\n  -> {v['label']}  "
        f"P(fake)={v['probability_fake']:.3f}  "
        f"frames_with_faces={sm['frames_with_faces']}/{sm['frames_sampled']}  "
        f"({elapsed:.1f} s)"
    )
    print(f"  Result written -> {args.out}")

    if sm["frames_with_faces"] == 0:
        print(
            "\n[WARN] No faces detected in any sampled frame.\n"
            "       Try --frame-step 10 or verify the clip contains a face."
        )


if __name__ == "__main__":
    main()