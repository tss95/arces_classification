# Handover Meeting Agenda

Use this as a time‑boxed checklist to bring your coworker up to speed and capture next steps.

## 1) Scope and Goals (5 min)
- What the system does today (event detection/classification on ARCES beams).
- Near‑term goal: productionize live inference using the PyTorch path.

## 2) Architecture Walkthrough (10 min)
- Config loading: `project_setup.py`, `global_config.py`, `config/data_config.yaml`.
- Models and training: `src/Models_torch.py`, `src/Loop_torch.py`, `train_torch.py`.
- Live pipeline: `src/Live.py` (`ClassifyGBF`, `LiveClassifier`), Torch entry `gbf_iter_torch.py`.
- Infra scripts: `common.sh`, `run.sh` family.

## 3) Live Demo Path (10 min)
- Set `PROJECT_DIR`, `DATA_DIR`.
- Ensure `pretrained_model_name` points to a valid `.ckpt`.
- Run: `python gbf_iter_torch.py --plots`.
- Review outputs under `cfg.project_paths.live_test_path`.

## 4) Known Issues Fixed Now (5 min)
- Fixed type check in `src/Live.py:get_data_to_predict` (ignore string error returns).
- Fixed label typo in `gbf_iter_torch.py` ("explosion").

## 5) Open TODOs (15 min)
- Unify live path fully on Torch; add `load_model_torch(...)` and deprecate TF on this branch.
- Stabilize scaling in live (prefer `Scaler_torch` policy over per‑window min‑max).
- Harden I/O: timeouts/retries for SeismonPy/Mongo and waveform fetch.
- Logging cleanup: replace prints, structured logs, consistent levels.
- Optional batching of windows per event; consider micro‑batches to balance latency vs throughput.
- Add tests for windowing/thresholding/label mapping.
- Consider CLI/API service for production (preloaded model, health endpoints).

## 6) Ownership and Next Steps (5 min)
- Assign owners for each TODO.
- Agree on acceptance criteria for “production ready”.
- Decide timeline and environments (dev/stage/prod) and monitoring.

