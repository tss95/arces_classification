# ARCES Classification — Handover Overview

This document orients a new maintainer to the core parts of the repository, how configuration is wired, where the main entry points are, and the current state of the PyTorch conversion. It also lists high‑value next steps for production readiness.

## Repository Layout

- `config/`
  - `data_config.yaml` — central run/configuration (model name, paths, live settings, filters, thresholds, etc.). See `config/data_config.yaml:1`.
  - `models/` — model‑specific hyperparameters (e.g., `alexnet.yaml`).
  - `logging_config.py` — Python logging configuration.
- `src/`
  - `Live.py` — GBF data retrieval, beamforming, and live inference utilities (TF‑era code with shared utilities). Key classes/functions:
    - `load_model(...)` — loads a Keras/TF model using `cfg.model_name` weights. See `src/Live.py:1`.
    - `ClassifyGBF` — fetch/prepare data windows; beamforming/filters; uses SeismonPy/ObsPy. See `src/Live.py:92`.
    - `LiveClassifier` — windowing, scaling, ensemble voting, plotting of outputs. See `src/Live.py:416`.
  - `Models_torch.py` — PyTorch/Lightning models (AlexNet1D, CNN_dense) and `get_model(...)`. See `src/Models_torch.py:1`.
  - `Loop_torch.py` — LightningModule with losses, metrics, and optimizer/scheduler wiring. See `src/Loop_torch.py:1`.
  - `Utils.py` — shared utilities (label prep, thresholds, one‑off inference). See `src/Utils.py:1`.
  - Other helpers: `BeamDataset.py`, `LoadData.py`, `Transforms.py`, `Scaler_torch.py`, etc.
- Top‑level scripts
  - Training: `train_torch.py` (Lightning training loop) at `train_torch.py:1`.
  - Live runs (interactive):
    - TF path: `gbf_iter.py` at `gbf_iter.py:1`.
    - Torch path: `gbf_iter_torch.py` at `gbf_iter_torch.py:1`.
  - Shell wrappers for NORSAR GPU host + Docker: `run.sh`, `run_predict.sh`, `run_live.sh` (uses `common.sh`).
- Infra
  - `docker.dockerfile`, `requirements.txt`, `.dockerignore`, `.docker_bashrc`.
  - `common.sh` — syncs data/configs, builds image if needed, and runs a chosen script in a container. See `common.sh:1`.

## Configuration and Environment

- Environment variables (required before import of project modules):
  - `PROJECT_DIR` — repo root path used to resolve config/paths. See `project_setup.py:22`.
  - `DATA_DIR` — base path for data paths in `config/data_config.yaml`. See `project_setup.py:11`.
- `global_config.py` imports `setup_config_and_logging()` which loads the YAMLs and returns `logger`, `cfg`, `model_cfg`. See `global_config.py:1` and `project_setup.py:1`.
- Key config fields in `config/data_config.yaml`:
  - `model_name` (YAML under `models/`), `pretrained_model_name` (weights/ckpt to load for live), `live.*` (window length, step, sample_rate), `filters.*`, `data.model_threshold`, `project_paths.*` (output folders), `wandb.*`.

## Training (PyTorch)

- Entry: `train_torch.py:1`.
  - Loads data via `src.Utils.prep_data()` and constructs `BeamDataset`/`DataLoader`.
  - Builds model via `src.Models_torch.get_model(...)` and prints `torchsummary`.
  - Uses PyTorch Lightning `Trainer` with callbacks (e.g., ModelCheckpoint) and optional WandB.
- Model shapes and outputs:
  - Input tensors are channel‑first `(batch, channels, timesteps)`.
  - Multi‑task outputs: `{'detector': logits, 'classifier': logits}` (both binary).
  - Losses: BCEWithLogits with per‑class weights; classifier loss masked to non‑noise detections (see `src/Loop_torch.py:120`).
- Metrics (torchmetrics): AUROC/AP on probabilities; accuracy/precision/recall/F1 on predictions; logged per stage.

## Inference Utilities

- `src/Utils.one_prediction(...)` converts a single window to a tensor (torch path) and calls the model; logits are passed through sigmoid thresholds to labels via `get_final_labels(...)`. See `src/Utils.py:236` and `src/Utils.py:269`.
- Threshold and label translation use `cfg.data.model_threshold` and the `label_maps` dict.

## Live/GBF Support (Overview)

- Data retrieval/beamforming handled by `ClassifyGBF` using SeismonPy and ObsPy, with rotation to RT, filtering, resampling, and beamforming based on `cfg.live.p_vel/s_vel`. See `src/Live.py:120` and `src/Live.py:300`.
- Live windowing + ensemble handled by `LiveClassifier`:
  - Splits traces into windows of `cfg.live.length` and steps of `cfg.live.step` (seconds). See `src/Live.py:448`.
  - Per‑window prediction → majority vote; aggregates mean probabilities. See `src/Live.py:466`.
  - Optional MP4 visualization using ObsPy plots combined with model outputs. See `src/Live.py:520`.
- Torch entry script: `gbf_iter_torch.py` builds `AlexNet1D`, loads Lightning checkpoint (`cfg.pretrained_model_name`), wraps in `LiveClassifier`, collects user time windows, and runs predictions with optional plots. See `gbf_iter_torch.py:1`.

## How to Run

- Local, PyTorch live (recommended for quick iteration):
  - Set `PROJECT_DIR` and `DATA_DIR` in your shell.
  - Ensure `config/data_config.yaml` points `pretrained_model_name` to a `.ckpt` file.
  - Run: `python gbf_iter_torch.py --plots`.
- Dockerized (GPU server):
  - `run.sh`/`run_predict.sh`/`run_live.sh` call `common.sh` which builds an image, syncs data/output, and runs the selected script.
  - To use Torch live via the wrapper: set `SCRIPT_NAME=gbf_iter_torch.py` in `run_live.sh:2`.

## TODOs For Coworker (Production Readiness)

- Unify live path on PyTorch:
  - `src/Live.load_model(...)` still targets TF/Keras (`src/Models`); Torch live uses checkpoint load in `gbf_iter_torch.py`. Consider adding a Torch `load_model_torch(...)` in `src/Live.py` and deprecating TF usage on this branch.
- Robustness and clarity:
  - Fix a likely typo in `get_data_to_predict`: `if traced is not isinstance(traced, str):` → `if not isinstance(traced, str):` (see `src/Live.py:274`).
  - Label typo in Torch script: `"exlposion"` → `"explosion"` (see `gbf_iter_torch.py:20`).
 - Review `train_torch.py` call to `get_model(...)` for argument order vs signature in `src/Models_torch.py`.
  - Ensure consistent scaling in live path (currently `local_minmax`; consider reusing `Scaler_torch` with a fixed policy).
  - Audit config key usage: several modules refer to `cfg.paths.*` while YAML defines `project_paths`/`data_paths`. Consider adding a compatibility shim or updating references for consistency.
- Operational hardening:
  - Add structured error handling/timeouts for SeismonPy/Mongo and waveform fetch.
  - Centralize logging and reduce print statements.
  - Validate/configure credentials and endpoints used by SeismonPy (Mongo, URLs).
  - Package as a CLI or service (e.g., FastAPI) with a preloaded model and a small stateful worker.
  - Add reproducible environments (conda/uv lock) and a slim runtime image.
  - Add unit/integration tests for windowing, label mapping, and end‑to‑end inference on a tiny sample.

Notes on batching windows:
- You can batch windows per event to shape `(N, C, T)` for a single forward pass; this typically reduces per‑event latency without hurting “live‑ness,” provided you still process at your cadence (e.g., every `cfg.live.step` seconds). Keep batches modest to avoid GPU memory spikes. Micro‑batches are a good compromise.

## Quick Reference (entry points)

- Live data classes: `src/Live.py:92` (ClassifyGBF), `src/Live.py:416` (LiveClassifier)
- Torch live script: `gbf_iter_torch.py:1`
- TF live script: `gbf_iter.py:1`
- Torch model: `src/Models_torch.py:180` (AlexNet1D)
- Training: `train_torch.py:1`
- Config wiring: `project_setup.py:1`, `global_config.py:1`
