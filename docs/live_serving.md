# Live Serving — GBF Inference (PyTorch)

This document explains how the live/GBF inference flow works in the PyTorch branch, where data is fetched and beamformed, windows are prepared, the model is run with ensemble voting, and results are optionally visualized.

## Entry Points

- Torch live script: `gbf_iter_torch.py:1`
  - Builds `AlexNet1D`, loads a Lightning checkpoint from `cfg.pretrained_model_name`, wraps in `LiveClassifier`, asks for a time range, fetches data, and runs predictions.
- Legacy TF script (still present): `gbf_iter.py:1`
  - Uses `src/Live.load_model(...)` (Keras) and the same `LiveClassifier` interface. Prefer the Torch script in this branch.

## Configuration

- Core: `config/data_config.yaml:1`
  - `pretrained_model_name` — path to a `.ckpt` file with Lightning `state_dict`.
  - `live.*` — length/step/sample_rate and beamforming velocities.
  - `filters.*` — filter selection and parameters for live preprocessing.
  - `data.model_threshold` — sigmoid threshold used in label translation.
- Environment variables must be set before running the script:
  - `PROJECT_DIR` and `DATA_DIR` (see `project_setup.py:1`).

## Data Fetch and Preprocessing

- `ClassifyGBF.get_data_to_predict(...)` iterates detected events and prepares per‑event traces. See `src/Live.py:278`.
  - Retrieves events + inventory from Mongo/SeismonPy; forms `(start, end)` windows.
  - Calls `get_beam(...)` to fetch raw traces and construct three beams: P‑Z, S‑T, S‑R. See `src/Live.py:139`.
  - Corrects start times, detrends/tapers/filters, resamples, rotates NE→RT, and beamforms with `cfg.live.p_vel/s_vel`.
  - Returns `tracedata` shaped `(3, timesteps)` and the resulting `Stream`.

Important functions and their locations:
- Beam construction: `src/Live.py:139` (get_beam)
- Start‑time fix: `src/Live.py:245` (correct_trace_start_times)
- Event loop: `src/Live.py:278` (get_data_to_predict)

## Model Setup (Torch)

- In `gbf_iter_torch.py:1`:
  - Label maps are defined (noise/event, earthquake/explosion).
  - `AlexNet1D` is created with an example input shape; checkpoint is loaded into `state_dict` (map to CPU/GPU as available).
  - `model.eval()` and wrap in `LiveClassifier`.
  - Note: There is a small label typo (`"exlposion"`) in the script that should be corrected.

## Windowing and Ensemble

- `LiveClassifier` orchestrates live inference and optional plotting. See `src/Live.py:416`.
- Windowing: `prepare_multiple_intervals(...)` splits the `(channels, time_steps)` trace into overlapping windows:
  - Window length = `cfg.live.length * cfg.live.sample_rate`.
  - Step size = `cfg.live.step * cfg.live.sample_rate`.
  - See `src/Live.py:455`.
- Per‑window preprocessing: currently a local min‑max normalization per window. See `src/Live.py:500` and usage at `src/Live.py:448`.
- Per‑window inference: `src/Utils.one_prediction(...)` creates a tensor and runs the Torch model; logits → sigmoid → threshold → labels via `get_final_labels(...)`. See `src/Utils.py:269` and `src/Utils.py:236`.
- Ensemble aggregation: majority vote over window labels; mean probabilities across windows. See `src/Live.py:474`.

## Visualization (optional)

- `plot_predicted_event(...)` renders an ObsPy waveform plot stacked with model output indicators per step and writes an MP4 in `cfg.project_paths.live_test_path`. See `src/Live.py:515`.

## How to Run (Torch)

- Quick local run:
  - Ensure `PROJECT_DIR` and `DATA_DIR` are set.
  - Update `config/data_config.yaml:1` → `pretrained_model_name` to a valid `.ckpt`.
  - Run: `python gbf_iter_torch.py --plots`.
- Dockerized GPU flow:
  - Update `run_live.sh:2` to `SCRIPT_NAME=gbf_iter_torch.py` and run `bash run_live.sh` (uses `common.sh` to build/sync/run inside a container).

## Production Notes and Next Steps

- Unify loader: add `load_model_torch(...)` beside `load_model(...)` in `src/Live.py` and remove TF dependencies on this branch.
- Scaling: consider standardizing on `src/Scaler_torch.Scaler` in live instead of per‑window min‑max for stability.
- Robustness: improve error handling/timeouts around waveform fetch and Mongo queries; centralize logging.
- Performance: preallocate buffers, avoid repeated conversions, and consider batched window inference.
- Interface: expose as a CLI or API (e.g., FastAPI) with a preloaded model and health/status endpoints.
- Testing: add unit tests for windowing, thresholding, and label translation; add an integration test with a tiny sample.

### Batch Window Inference — does it hurt “live” behavior?
- Not necessarily. If you batch only the windows for a single event/time slice and maintain your stepping cadence, batching reduces overhead and can improve responsiveness. Keep batch sizes reasonable and disable autograd for inference. For truly streaming scenarios, you can micro‑batch or process windows sequentially depending on latency targets.
