"""
src/deepfake_detector/inference.py
====================================
Student ID : 21396816
Python     : 3.10 or higher
Purpose    : Core inference logic for deepfake-face detection from video.
             Stateless library module — no CLI, no file I/O, no side effects.
             All orchestration (model loading, JSON writing) lives in
             scripts/run_inference.py.

Pipeline per call to predict_video()
--------------------------------------
1. cv2.VideoCapture  — open clip, read frames at fixed intervals
2. MTCNN             — detect faces; crop the largest per frame to 224×224
3. eval_transform    — resize + ToTensor + ImageNet normalise
4. ResNet18          — forward pass, raw logits
5. softmax           — convert logits to P(fake), P(real)
6. aggregate         — mean P(fake) across frames that had a detected face
7. label             — "FAKE" if mean P(fake) ≥ threshold, else "REAL"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# Evaluation transform — must match the one used during training exactly.
# Any deviation (different resize, missing normalise) silently degrades
# accuracy without raising an error.
# ---------------------------------------------------------------------------
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


def predict_video(
    video_path: str | Path,
    model: torch.nn.Module,
    device: torch.device("cpu"),
    frame_step: int = 30,
    fake_threshold: float = 0.5,
    mtcnn_margin: int = 20,
    mtcnn_min_face_size: int = 60,
) -> dict[str, Any]:
    """Run the full deepfake detection pipeline on a single video clip.

    Parameters
    ----------
    video_path:
        Path to the input .mp4 (or any format OpenCV can decode).
    model:
        A model returned by build_model(), already moved to *device* and in
        eval mode (model.eval() must be called before this function).
        A RuntimeError is raised immediately if the model is in training mode.
    device:
        Torch device matching the model's location.
    frame_step:
        Sample one frame every *frame_step* frames. Default 30 ≈ 1 fps for
        30 fps footage. Reducing this increases accuracy at the cost of
        runtime; increasing it speeds up inference on long clips.
    fake_threshold:
        Minimum mean P(fake) required to label the video as "FAKE". 0.5 is
        the natural decision boundary; raise it to favour false negatives
        (call things real when uncertain), lower it for false positives.
    mtcnn_margin:
        Pixels of padding added around the detected face bounding box before
        cropping. Small values clip the chin/forehead; large values include
        background noise.
    mtcnn_min_face_size:
        Minimum face size MTCNN will detect (pixels). Filters out tiny
        background faces and detection artefacts.

    Returns
    -------
    dict with the following structure::

        {
            "video_level": {
                "label": "FAKE",
                "probability_fake": 0.872,
                "probability_real": 0.128
            },
            "frame_level": [
                {"frame_index": 0,  "face_detected": True,
                 "probability_fake": 0.851, "probability_real": 0.149},
                {"frame_index": 30, "face_detected": False,
                 "probability_fake": None,  "probability_real": None}
            ],
            "summary": {
                "frames_sampled": 30,
                "frames_with_faces": 28,
                "frames_without_faces": 2
            }
        }

    Raises
    ------
    RuntimeError
        If *model* is in training mode — call model.eval() before passing it.
        Loud failure is preferable to silent wrong predictions.
    RuntimeError
        If the model does not output exactly 2 classes.
    FileNotFoundError
        If *video_path* does not exist or cannot be opened by OpenCV.
    """
    # ── Guard: eval mode ─────────────────────────────────────────────────────
    # Training mode activates Dropout and uses per-batch BatchNorm statistics,
    # both of which produce incorrect and non-deterministic predictions at
    # inference time. Raising here is preferable to silent wrong answers.
    if model.training:
        raise RuntimeError(
            "[inference] model must be in eval mode before predict_video — "
            "call model.eval() after loading weights. "
            "Train mode activates Dropout and per-batch BatchNorm, producing "
            "incorrect predictions."
        )
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(
            f"[inference] Video not found: {video_path}"
        )

    # ── MTCNN face detector ──────────────────────────────────────────────────
    # MTCNN runs on CPU because PyTorch's MPS backend (Apple Silicon) does not
    # yet support adaptive pooling with non-divisible input sizes — see:
    # https://github.com/pytorch/pytorch/issues/96056
    # The CPU detection cost (~50 ms per frame) is negligible vs. classifier
    # inference on MPS, so this has no meaningful impact on overall runtime.
    mtcnn = MTCNN(
        image_size=224,
        margin=mtcnn_margin,
        min_face_size=mtcnn_min_face_size,
        keep_all=False,
        post_process=False,
        device=torch.device("cpu"),
    )

    # ── Open video ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(
            f"[inference] OpenCV could not open video: {video_path}"
        )

    frame_results: list[dict[str, Any]] = []
    frame_index = 0

    try:
        while True:
            ret, bgr_frame = cap.read()
            if not ret:
                break   # end of clip

            if frame_index % frame_step == 0:
                result = _process_frame(
                    bgr_frame, frame_index, mtcnn, model, device
                )
                frame_results.append(result)

            frame_index += 1
    finally:
        cap.release()   # always release, even on exception

    # ── Aggregate ────────────────────────────────────────────────────────────
    scored_frames = [r for r in frame_results if r["face_detected"]]
    frames_with_faces    = len(scored_frames)
    frames_without_faces = len(frame_results) - frames_with_faces

    if scored_frames:
        mean_p_fake = sum(r["probability_fake"] for r in scored_frames) / frames_with_faces
    else:
        # No faces detected in any sampled frame — cannot classify.
        # Return a neutral 0.5 rather than crashing; caller can check
        # summary.frames_with_faces == 0 and warn accordingly.
        mean_p_fake = 0.5

    mean_p_real = 1.0 - mean_p_fake
    label = "FAKE" if mean_p_fake >= fake_threshold else "REAL"

    return {
        "video_level": {
            "label": label,
            "probability_fake": round(mean_p_fake, 4),
            "probability_real": round(mean_p_real, 4),
        },
        "frame_level": frame_results,
        "summary": {
            "frames_sampled":       len(frame_results),
            "frames_with_faces":    frames_with_faces,
            "frames_without_faces": frames_without_faces,
        },
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _process_frame(
    bgr_frame: "cv2.typing.MatLike",
    frame_index: int,
    mtcnn: MTCNN,
    model: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    """Detect a face in *bgr_frame*, classify it, and return a frame record.

    OpenCV reads frames in BGR order; PIL and MTCNN expect RGB. The
    conversion is performed here, once, before any further processing.
    """
    # BGR (OpenCV) → RGB (PIL/MTCNN)
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    # MTCNN returns a float32 tensor [C, H, W] in [0, 255] range if a face
    # is found, or None if no face passes the confidence threshold.
    face_tensor = mtcnn(pil_image)

    if face_tensor is None:
        return {
            "frame_index":      frame_index,
            "face_detected":    False,
            "probability_fake": None,
            "probability_real": None,
        }

    # Convert MTCNN output → uint8 PIL image → eval_transform → model input.
    # Clamp first: MTCNN occasionally produces values marginally outside
    # [0, 255] due to floating-point margin arithmetic.
    face_pil = Image.fromarray(
        face_tensor.permute(1, 2, 0).clamp(0, 255).byte().numpy()
    )
    input_tensor = _eval_transform(face_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)            # (1, 2) raw logits

        # Guard against a mis-configured model (e.g. 3-class head) that would
        # silently produce wrong class-index mappings without this check.
        if logits.shape[1] != 2:
            raise RuntimeError(
                f"[inference] expected a 2-class model (fake / real), "
                f"but model output has {logits.shape[1]} classes. "
                f"Check num_classes in best_model_config.json."
            )

        probs = F.softmax(logits, dim=1)[0]     # (2,) probabilities

    # class_to_idx: {"fake": 0, "real": 1} — alphabetical ImageFolder order.
    p_fake = probs[0].item()
    p_real = probs[1].item()

    return {
        "frame_index":      frame_index,
        "face_detected":    True,
        "probability_fake": round(p_fake, 4),
        "probability_real": round(p_real, 4),
    }