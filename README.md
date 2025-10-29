# ARCES Classification — PyTorch Branch

End-to-end system for seismic event detection and classification on beamformed ARCES array data. This branch uses PyTorch and PyTorch Lightning for training and live inference. Live usage produces early label recommendations for GBF events (either incoming events or a selected bulletin).

## Table of Contents

1. Project Overview
2. Requirements
3. Environment & Config
4. Repository Layout
5. Training (PyTorch)
6. Live Inference (GBF, PyTorch)
7. Validation & Live-Mode Evaluation (Recommendation)
8. Troubleshooting
9. Maintainer Notes
10. Operational Workflows

## 1) Project Overview

- Task: Binary detection (noise vs event) and binary classification (earthquake vs explosion) on array beams.
- Live: Processes GBF events and outputs early label recommendations; optional visualizations.
- Infra: Can run locally or via Docker on a GPU host using provided scripts.

## 2) Requirements

- Python 3.10+ recommended.
- Key packages: PyTorch, PyTorch Lightning, torchmetrics, ObsPy, SeismonPy, scikit-image, imageio.
- Optional: Weights & Biases (wandb) for experiment tracking.

Install locally (example):
```bash
pip install -r requirements.txt
```

## 3) Environment & Config

- Required environment variables (set before importing project code):
  - `PROJECT_DIR` — repository root used to resolve configuration paths.
  - `DATA_DIR` — base directory for data paths.

Example (shell):
```bash
export PROJECT_DIR=/path/to/this/repo
export DATA_DIR=/path/to/data/root
```

Example (Jupyter):
```python
import os
os.environ['PROJECT_DIR'] = '/path/to/this/repo'
os.environ['DATA_DIR'] = '/path/to/data/root'
```

- Configuration files:
  - `config/data_config.yaml` — main run config (live, filters, thresholds, paths, pretrained checkpoints, etc.).
  - `config/models/*.yaml` — model hyperparameters (e.g., alexnet).
  - `global_config.py` / `project_setup.py` — loads YAML into `cfg`/`model_cfg` and sets logging.

Important fields:
- `pretrained_model_name` — path to a Lightning checkpoint (`.ckpt`) used for live inference.
- `live.*` — window length/step/sample rate and beamforming velocities.
- `filters.*` — filter selection and parameters for live preprocessing.
- `data.model_threshold` — sigmoid threshold for binary decisions.
- `project_paths.*` / `data_paths.*` — output/data directories (resolved under `DATA_DIR`/`PROJECT_DIR`).

Optional: `WANDB_API_KEY` for Weights & Biases.

## 4) Repository Layout

- `config/` — YAML configs and logging config.
- `src/` — core modules:
  - `Live.py` — GBF fetch/beamform (`ClassifyGBF`), live pipeline (`LiveClassifier`), plotting helpers.
  - `Models_torch.py`, `Loop_torch.py` — Torch model definitions and Lightning training base.
  - `Utils.py`, `BeamDataset.py`, `LoadData.py`, `Transforms.py`, `Scaler_torch.py` — utilities, datasets, scaling.
- Scripts:
  - Training: `train_torch.py`
  - Live: `gbf_iter_torch.py` (interactive; recommended)
  - Docker wrappers: `run.sh`, `run_predict.sh`, `run_live.sh` (via `common.sh`)
- Legacy TF components (kept for reference only): `src/Models.py`, `src/Callbacks_tf.py`, `src/Scaler_tf.py`, `train.py`, `predict.py`, `gbf_iter.py`

## 5) Training (PyTorch)

Local:
```bash
cd $PROJECT_DIR
python train_torch.py
```

GPU host via Docker:
```bash
cd $PROJECT_DIR
bash run.sh
```

Notes:
- `train_torch.py` builds datasets via `src/Utils.prep_data()`, constructs a Lightning model from `src/Models_torch.py`, and logs metrics (optionally to wandb).
- Inputs are channel-first `(batch, channels, timesteps)`.
- Outputs: `{'detector': logits, 'classifier': logits}`.

## 6) Live Inference (GBF, PyTorch)

Purpose: provide early label recommendations for GBF events (incoming or from a chosen bulletin). Optionally saves MP4 visualizations per event.

Local run:
```bash
cd $PROJECT_DIR
python gbf_iter_torch.py --plots
```

Docker on GPU host:
```bash
cd $PROJECT_DIR
bash run_live.sh
```

Configuration tips:
- Set `pretrained_model_name` in `config/data_config.yaml` to a valid `.ckpt` (Lightning checkpoint with `state_dict`).
- Visualizations go to `cfg.project_paths.live_test_path`.

How it works (high level):
- `ClassifyGBF.get_data_to_predict(...)` fetches events + inventory (SeismonPy/Mongo), preprocesses, and builds P/S beams.
- `LiveClassifier` splits traces into windows (length = `cfg.live.length`, step = `cfg.live.step`), normalizes, and runs per-window inference.
- Ensemble: majority vote across windows + mean probabilities; optional plot saved to video.

## 7) Validation & Live-Mode Evaluation (Recommendation)

Standard validation uses single-window predictions. To better approximate “liveness” on labeled data, simulate the live pipeline on validation/test:

- For each sample, run the same windowing + ensemble steps as in live.
- Aggregate predictions as in live (majority vote; mean probabilities).
- Compute metrics on aggregated predictions to estimate live performance.

This is optional and can be done as time permits (Maikael may own this exploration).

## 8) Troubleshooting

- Environment variables not set → `project_setup.py` raises helpful errors; set `PROJECT_DIR`, `DATA_DIR` before imports.
- Checkpoint loading → ensure `pretrained_model_name` points to a valid `.ckpt` file.
- SeismonPy/Mongo access → the live fetch depends on internal services; configure credentials/endpoints as needed.
- GPU issues → Lightning defaults to all available devices; set CUDA-visible devices or adjust `Trainer` args if necessary.

## 9) Maintainer Notes

- Primary maintainer for productionization: Maikael.
- See `docs/handover_overview.md` for a system overview and TODOs for hardening.
- See `docs/handover_overview.md` → Known Existing Issues for a quick start list of current gaps/quirks.
- See `docs/live_serving.md` for the GBF live pipeline details.
- See `docs/handover_agenda.md` for a suggested handover meeting checklist.

## 10) Operational Workflows

- Option A — develop directly on the GPU machine (simpler):
  - SSH to the GPU host and work in-place (no local→remote transfer).
  - Create/activate your environment, set `PROJECT_DIR` and `DATA_DIR`, then run Python scripts directly (`train_torch.py`, `gbf_iter_torch.py`).
  - For a simpler template of this approach, refer to the minem_arraydetect repo (Tord’s branch): https://github.com/NorwegianSeismicArray/minem_arraydetect/tree/tord

- Option B — use the existing Docker transfer flow (fast start):
  - `run.sh` (training), `run_predict.sh` (predict), `run_live.sh` (live) wrap `common.sh` to:
    - Sync input data from `$DATA_DIR` to a working path on the GPU host.
    - Build the Docker image if the Dockerfile/requirements changed (use `-b` to force rebuild).
    - Run the selected script inside the container with GPU access.
    - Sync outputs back to `$PROJECT_DIR/output`.
  - Typical commands:
    - Training: `bash run.sh`
    - Predict: `bash run_predict.sh`
    - Live: `bash run_live.sh` (defaults to the Torch live script)
