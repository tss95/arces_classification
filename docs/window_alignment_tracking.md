# Window Alignment & Preprocessing Parity — Tracking Doc

## Goal
Ensure live inference uses the same input window length and preprocessing as training. Remove silent fallbacks (shape‑invariant behavior) so mismatches fail fast.

## Why This Exists
We observed that the model is not crashing even when configs appear mismatched (e.g., 180s vs 80s). The current pipeline tolerates variable lengths or uses unrelated values, which can silently shift the input distribution.

## Current Findings (As of 2026‑02‑04)

### Live Inference (PyTorch GBF path)
- Window length used for inference is `cfg.live.length`.
- `config/data_config.yaml` sets:
  - `live.length = 80` seconds
  - `live.sample_rate = 40` Hz
  - `live.step = 5` seconds
- `src/Live.py` windowing:
  - `interval_length = cfg.live.length * cfg.live.sample_rate` → `80 * 40 = 3200` timesteps
  - `step_size = cfg.live.step * cfg.live.sample_rate` → `5 * 40 = 200` timesteps
  - File: `src/Live.py:454-469`
- Live beamforming/filtering pipeline uses:
  - `stream.detrend('demean')`
  - `stream.taper(...)`
  - `stream.filter('highpass', freq=1.5)`
  - `stream.resample(cfg.live.sample_rate)`
  - Then beamforming and trimming to the requested window
  - File: `src/Live.py:140-237`
- Live visualization uses the same `cfg.live.sample_rate` for trace stats.
  - File: `src/Live.py:517-529`

### Training (PyTorch, recommended entrypoint)
- `README.md` states `code_test.py` is the current PyTorch training entrypoint, and `train_torch.py` is outdated.
  - File: `README.md:90-110`
- `code_test.py` sets model input shape directly from config:
  - `input_shape = (3, cfg.augment.random_crop_kwargs.timesteps)`
  - File: `code_test.py:61-66`
- `RandomCropTransform` uses `cfg.augment.random_crop_kwargs.timesteps` to crop samples to a fixed length.
  - File: `src/Transforms.py:34-68`
- `config/data_config.yaml` sets:
  - `augment.random_crop_kwargs.timesteps = 3200` (comment notes ~80% of a 100s window)
  - `data.sample_rate = 40`
  - File: `config/data_config.yaml:88-92, 40-41`
- Therefore, training samples are cropped to 3200 timesteps (80 s @ 40 Hz).

### Other Config Values in Play
- `data.default_length_seconds = 180` exists but is not referenced anywhere in code.
  - File: `config/data_config.yaml:40`
- `src/Models.py` and TF live path uses `cfg.live.length * cfg.live.sample_rate + 1` for input shape (3201), but live windows are 3200.
  - File: `src/Live.py:46-63`

## Known Mismatches / Risks

1. **Inconsistent model init in live Torch script**
   - `gbf_iter_torch.py` constructs the model with a dummy input of shape `(3, 4000)` (100 s @ 40 Hz), while live windows are 3200.
   - File: `gbf_iter_torch.py:26-39`

2. **Shape‑tolerant model behavior**
   - `AlexNet1D` uses a Transformer head and interpolates positional encodings to accommodate any sequence length.
   - This hides length mismatches and can change behavior silently.
   - File: `src/Models_torch.py:377-404`

3. **Stale/unused length config**
   - `data.default_length_seconds = 180` is not used; it gives the impression that training expects 180s.
   - File: `config/data_config.yaml:40`

4. **Potential preprocessing mismatch**
   - Live path includes a fixed `highpass 1.5` plus additional filter selection logic later in the function (bandpass/highpass). This is not obviously mirrored in training.
   - File: `src/Live.py:200-245`

5. **Scaler fallback risk**
   - `gbf_iter_torch.py` loads `scaler_state` from checkpoint if present; otherwise only logs a warning.
   - This is unsafe for distribution parity.
   - File: `gbf_iter_torch.py:41-49`

## Data Needed To Confirm Canonical Training Distribution
We need to inspect the actual HDF5 training data to confirm the **native sample length before cropping**, and any metadata about the window length used in dataset creation.

When access is granted, inspect:
- `$DATA_DIR/loaded_classifier/*_data.h5` (e.g., `train_full_data.h5`)
- Expected datasets: `data`, `labels`, `windows` in the HDF5 file
- Confirm `data` shape and timesteps, and whether `windows` indicates the original window length.

## Data Inspection Results (GPU host, /nobackup2/tord/arces_classification)

Legacy NumPy dataset (not HDF5) found at:
`/nobackup2/tord/arces_classification/data/loaded/`

Shapes:
- `train_traces_filtered.npy`: `(180930, 3, 9601)` float32
- `val_traces_filtered.npy`: `(27139, 3, 9601)` float32
- `test_traces_filtered.npy`: `(18094, 3, 9601)` float32
- Unfiltered traces are identical in shape (`*_traces.npy`).

Metadata confirms sampling rate:
- `train_metadata.pkl` contains `trace_stats.sampling_rate = 40.0`

Implication:
- Raw training windows are **9601 timesteps ≈ 240.0 seconds at 40 Hz** (4 minutes).
- Training uses `RandomCropTransform` with `timesteps=3200` → **80s crops** for model input.

No `loaded_classifier/` HDF5 directory exists under `/nobackup2/tord/arces_classification` at time of inspection.

## Decisions Required
1. **Canonical window length** (seconds + sample rate) to use across training and live.
2. **Strict enforcement**: should any mismatch be a hard error?
3. **Preprocessing parity**: define a shared pipeline for live and training to ensure identical filtering, tapering, scaling.

## Proposed Next Steps (once data access is available)
1. Read one training HDF5 file to confirm native sample length and window metadata.
2. Document the canonical window length and update `config/data_config.yaml` to expose a single source of truth.
3. Update `gbf_iter_torch.py` to construct model using the canonical length (not a hardcoded 4000).
4. Add strict input‑length checks in live inference and model initialization.
5. Remove or disable positional‑encoding interpolation unless explicitly opted in.
6. Align live preprocessing with training transforms and make it shared where possible.
7. Require scaler state for live inference (fail if missing).

## Status Updates (2026‑02‑04)
- Canonical window length confirmed at **80 seconds** (3200 timesteps @ 40 Hz).
- Config updated to declare `data.window_seconds = 80`, and `default_length_seconds` aligned to 80.
- Added config validation in `project_setup.py` to enforce:
  - `augment.random_crop_kwargs.timesteps == data.window_seconds * data.sample_rate`
  - `live.length == data.window_seconds`
  - `live.sample_rate == data.sample_rate`
  - `default_length_seconds == window_seconds`
- Removed hardcoded `4000` timesteps in `gbf_iter_torch.py`.
- Removed transformer positional‑encoding interpolation fallback and added strict input length checks in `src/Models_torch.AlexNet1D`.

## Files Referenced
- `config/data_config.yaml`
- `src/Live.py`
- `gbf_iter_torch.py`
- `src/Models_torch.py`
- `src/Transforms.py`
- `code_test.py`
- `README.md`
