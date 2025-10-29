# Onboarding Checklist — Maikael

This is a concise, practical checklist to bring you up to speed quickly. You’re experienced in ML Ops, so this focuses on where things live, how to run them, and what’s worth improving first.

## 1) Big Picture (5 min)
- Goal: Early label recommendations for GBF events (incoming or bulletin-selected).
- Branch: PyTorch-first (training + live in Torch; TensorFlow code kept as legacy).
- Start with: README.md (project overview), then docs/handover_overview.md and docs/live_serving.md.

## 2) Environment & Config (10 min)
- Env vars: set `PROJECT_DIR`, `DATA_DIR` before imports.
- Config files:
  - `config/data_config.yaml` — live windowing, filters, thresholds, `pretrained_model_name` (.ckpt).
  - `config/models/*.yaml` — model HPs (alexnet, etc.).
  - Load flow: `global_config.py` → `project_setup.py` → `cfg`, `model_cfg`.

## 3) How to Run (10 min)
- Preferred workflow (simpler): develop directly on the GPU host.
  - `python train_torch.py`
  - `python gbf_iter_torch.py --plots`
  - Reference template: https://github.com/NorwegianSeismicArray/minem_arraydetect/tree/tord
- Docker workflow (fast start):
  - Training: `bash run.sh`
  - Predict: `bash run_predict.sh`
  - Live: `bash run_live.sh` (now defaults to Torch live script)
  - See README “Operational Workflows” for details.

## 4) Code Orientation (10–15 min)
- Live/GBF: `src/Live.py` → `ClassifyGBF` (fetch/beamform) and `LiveClassifier` (windowing, ensemble, viz).
- Models: `src/Models_torch.py` (AlexNet1D, etc.) + `src/Loop_torch.py` (LightningModule, losses/metrics/opt).
- Utilities: `src/Utils.py` (single-window inference, label translation), `src/Scaler_torch.py` (scaling policies), `src/BeamDataset.py`, `src/LoadData.py`.
- Entry scripts: `train_torch.py`, `gbf_iter_torch.py`.

## 5) Live Serving Details (10 min)
- Windowing and ensemble: `LiveClassifier.prepare_multiple_intervals(...)` and majority vote across windows.
- Visualization: MP4 per event combining ObsPy waveform + probability tracks.
- Output path: `cfg.project_paths.live_test_path`.

## 6) Known Issues / Quick Wins (15–20 min)
- Fixed: skip failed waveform retrievals (src/Live.py:299); label typo in torch live (gbf_iter_torch.py:20).
- Verify/align `train_torch.py` vs `Models_torch.get_model(...)` signature (metrics/label maps/weights order).
- Scheduler config in `Loop_torch.configure_optimizers`: typo (`optimzer`) and warmup logic likely off.
- Live scaling policy: per-window local min-max vs `Scaler_torch`; pick one and standardize.
- `Scaler_torch` transform signatures: align base/children (MinMaxScaler.transform currently expects `cfg`).
- Config consistency: unify `cfg.paths.*` vs `project_paths`/`data_paths` references or add a shim.
- I/O hardening: timeouts/retries/logging around SeismonPy/Mongo.
- Optional performance: micro-batch windows per event with `torch.no_grad()`.
- See docs/handover_overview.md → Known Existing Issues for details.

## 7) Validation Strategy (optional)
- Simulate live-mode evaluation on validation/test: run the same windowed ensemble as live and compute metrics on aggregated predictions.
- Helps approximate expected live performance (since GBF lacks labels).

## 8) Deployment & Ops Considerations
- Decide between in-place on GPU host vs Dockerized flow (both supported now).
- Logging: standardize Python logging, reduce prints, add event/time context.
- Observability: metrics export (TorchMetrics/W&B), basic dashboards, storage quotas for MP4s.
- Credentials/endpoints: SeismonPy/Mongo; secure handling and error handling.

## 9) Decision Log / Next Steps
- Choose scaling policy for live (and apply consistently).
- Confirm LR scheduler policy; fix typos and argument names.
- Align config key usage or add compatibility shim.
- Pick validation approach (one-shot vs live-sim) and document.

