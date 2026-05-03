# Deepfake Video Detection — Frame-Level CNN Classifier

## What this is

This project implements a frame-level binary classifier that scores short video clips as either REAL or FAKE by detecting manipulated faces. It was produced as the Creative Piece component of the undergraduate Software Engineering degree at Manchester Metropolitan University (student ID: 21396816). The approach fine-tunes a ResNet18 backbone, pre-trained on ImageNet, on a balanced sample of face crops derived from the Deepfake Faces dataset on Kaggle, which is itself sourced from the DeepFake Detection Challenge (DFDC).

## Headline results

Evaluated on a held-out test set of 1,500 balanced face crops (750 real, 750 fake), the model achieves an accuracy of 92.47%, a weighted F1 score of 0.925, and an AUC of 0.981. The complete per-class breakdown, including precision and recall for each class, is recorded in `results/evaluation_metrics.json`.

## How to install

The project has been tested on macOS (Apple Silicon and Intel) and Linux with Python 3.9 or higher. On Apple Silicon Macs, PyTorch will automatically select the MPS backend, which gives the fastest training and inference times. On Intel machines with an NVIDIA GPU, CUDA will be selected instead; otherwise the script falls back to CPU.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All dependencies, including their exact versions, are pinned in `requirements.txt`. No additional system packages are required beyond a working Python 3.9+ installation.

## How to run inference

Two equivalent entry points are provided. The first uses the package's `__main__` shim, which satisfies the project brief's required command form. The second calls the script directly and is more convenient during development.

```bash
PYTHONPATH=src python -m deepfake_detector --input <video> --out <result.json>
python scripts/run_inference.py --input <video> --out <result.json>
```

Supported input formats are `.mp4`, `.mov`, and `.avi`. Files larger than 500 MB are rejected at the boundary with a clear error message. Two optional flags control behaviour: `--frame-step` (default 30) sets how many frames are skipped between samples — lower values increase accuracy at the cost of runtime — and `--threshold` (default 0.5) sets the minimum mean P(fake) required to label a video as FAKE.

A worked example using the included sample clip:

```bash
python scripts/run_inference.py \
    --input data/sample/test_clip.mov \
    --out   results/sample_prediction.json \
    --frame-step 30 \
    --threshold  0.5
```

The result JSON is written to the path given by `--out`, and a one-line verdict is printed to the terminal.

## How to reproduce the evaluation

The following sequence of commands takes a cold checkout to the headline numbers. Each step is fully scripted; no manual data manipulation is required.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the dataset from Kaggle to the expected location
#    Dataset: mafiosoquasar/deepfake-faces
#    URL: https://www.kaggle.com/datasets/mafiosoquasar/deepfake-faces
#    Place the unzipped folder at: ~/Downloads/deepfake-faces/
#    It must contain two sub-folders: real/ and fake/

# 3. Prepare the working dataset (samples 5,000 per class; splits 70/15/15)
python scripts/prepare_data.py

# 4. Train for five epochs
python scripts/train.py --epochs 5

# 5. Evaluate on the held-out test set
python scripts/evaluate.py
```

After step 5, open `results/evaluation_metrics.json` and confirm the accuracy, F1, and AUC figures match the headline numbers above. The data split is fully reproducible: re-running `prepare_data.py` and diffing `data/working/manifest.csv` against a previous run produces a zero diff, because the sampling is seeded with a fixed value (seed 42) and the per-class offsets are hard-coded rather than derived from Python's hash function.

## Sample output

The JSON schema is illustrated below using representative values. Actual probabilities will depend on the input clip and the trained model — running inference on the included sample produces a real result file at `results/sample_prediction.json`.

```json
{
  "schema_version": "1.0",
  "input": "data/sample/test_clip.mov",
  "model_checkpoint": "models/best_model.pt",
  "model_config": "models/best_model_config.json",
  "device": "mps",
  "inference_seconds": 4.21,
  "video_level": {
    "label": "FAKE",
    "probability_fake": 0.872,
    "probability_real": 0.128
  },
  "frame_level": [
    {"frame_index": 0,  "face_detected": true,
     "probability_fake": 0.891, "probability_real": 0.109},
    {"frame_index": 30, "face_detected": true,
     "probability_fake": 0.853, "probability_real": 0.147},
    {"frame_index": 60, "face_detected": false,
     "probability_fake": null,  "probability_real": null}
  ],
  "summary": {
    "frames_sampled": 30,
    "frames_with_faces": 28,
    "frames_without_faces": 2
  }
}
```

## Known limitations and design choices

The model was trained exclusively on face crops from the DFDC dataset. Inputs that differ substantially from this distribution — such as webcam selfies recorded under unusual lighting, or faces at extreme angles — produce borderline confidence scores. During end-to-end testing, one real-world clip returned P(fake) = 0.510, which correctly resolves to REAL at the default threshold of 0.5 but would flip to FAKE with only minor threshold adjustment. In-the-wild deployment would require threshold calibration on a representative held-out set.

A slight false-positive bias is present in the evaluation results: recall on the fake class is 0.908, meaning roughly 9% of fake clips are misclassified as real. This asymmetry is typical of transfer-learning classifiers trained on a balanced but domain-specific dataset. Raising the threshold above 0.5 would reduce this at the cost of increased false positives on the real class.

The MTCNN face detector runs on CPU rather than the MPS device, because PyTorch's MPS backend does not yet support adaptive average pooling with non-divisible input sizes. This is a known upstream limitation tracked at [pytorch/pytorch#96056](https://github.com/pytorch/pytorch/issues/96056). In practice the CPU detection cost is approximately 50 ms per frame and has negligible impact on overall runtime, since the ResNet18 classifier — the computationally dominant step — still runs on MPS.

Only ResNet18 was evaluated within the time budget of this project. ResNet50 and Xception are both common choices in the deepfake-detection literature and represent reasonable future comparisons; the `build_model` function in `src/deepfake_detector/model.py` is written to make backbone substitution straightforward. Additionally, the current pipeline aggregates frame-level scores by arithmetic mean, discarding temporal ordering entirely. A 3D convolutional network or an LSTM operating over frame embeddings would likely improve performance on longer clips where inter-frame consistency is a discriminative signal.

## Project structure

```
.
├── data/
│   ├── sample/          # included sample clips for smoke testing
│   └── working/         # generated by prepare_data.py (not in repo)
├── models/              # best_model.pt and best_model_config.json (generated)
├── results/             # evaluation outputs (generated)
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── run_inference.py
├── src/
│   └── deepfake_detector/
│       ├── __init__.py
│       ├── __main__.py
│       ├── data.py
│       ├── inference.py
│       └── model.py
├── tests/
│   ├── test_data.py
│   ├── test_inference.py
│   └── test_model.py
├── DATASETS.md
├── EAN.txt
├── INSTRUCTIONS.txt
├── LICENCE
├── README.md
├── requirements.txt
└── Showcase_video_link.txt
```

## Running the test suite

Three prerequisites must be satisfied before running the full test suite: `data/working/` must exist (run `prepare_data.py`), `models/best_model.pt` must exist (run `train.py`), and `data/sample/test_clip.mov` must be present for the end-to-end smoke test. Tests that require these files are decorated with `pytest.mark.skipif` and will skip cleanly if the prerequisites are absent rather than failing with a confusing error.

```bash
.venv/bin/python -m pytest tests/ -v
```

The expected outcome is 13 tests passing in roughly 17 seconds. Tests that exercise the data pipeline are the slowest, as they construct DataLoaders and pull one real batch; model and inference unit tests complete in under two seconds each.

## Reproducibility

All stochastic operations are seeded with the value 42, applied consistently across Python's `random` module, NumPy, PyTorch CPU, and — where supported — the MPS backend. The manifest at `data/working/manifest.csv` provides a cast-iron record of which image landed in which split; diffing this file between two runs of `prepare_data.py` on the same source data produces no differences. The saved checkpoint is accompanied by `models/best_model_config.json`, which records the architecture name, number of classes, image size, normalisation statistics, and class-to-index mapping, making the checkpoint self-describing without reference to the training script. Minor run-to-run variation may occur on Apple Silicon due to non-deterministic kernels in the MPS backend; CPU-only runs are fully deterministic.

## Ethics

This project was conducted under the ethical sign-off of the project supervisor, Dr Nicholas Costen, in accordance with MMU's Faculty research ethics procedures. The supervisor's authorisation is recorded in `EAN.txt` at the repository root in lieu of a separately issued EthOS reference number. Dataset licences are documented in full in `DATASETS.md`. The project works with images of real human faces; no children are present in the source datasets, no re-identification of individuals has been attempted, and no raw face images from the source datasets are redistributed inside the submission archive. A known fairness limitation is that the demographic balance of the training data has not been audited; this is flagged as a limitation and noted as future work.

## Licence

This project is released under the MIT Licence; see the [LICENCE](LICENCE) file for the full text.

## Author

Coco Yav, student ID 21396816, BSc (Hons) Software Engineering Creative Piece, Manchester Metropolitan University, 2026.