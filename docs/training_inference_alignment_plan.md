# Training/Inference Convergence Plan (Updated)

Date: 2026-02-11  
Scope: `arces_classification` (training repo) + `ml_array_data_classification` (inference repo)

## 1) Objective

Close the train/validation vs live-performance gap while reducing maintenance duplication.

Target ownership:
- `arces_classification`: training, dataset prep, offline analysis, promotion gates.
- `ml_array_data_classification`: live inference implementation (single source of truth).

## 2) Current State (Confirmed)

### 2.1 New Dataset Pipeline Is In Place

New training data path:
- `data_paths.loaded_path: loaded_classifier_nofilt/`
- `data.source_variant: nofilt`
- Split policy:
  - train: all years excluding val/test
  - val: `2024`
  - test: `2025`
- Dropped labels:
  - `data.drop_event_labels: ["not existing"]`

Implemented in `arces_classification`:
- Config: `config/data_config.yaml`
- Data file discovery + metadata resolution + split handling:
  - `src/Utils_torch.py`
- HDF5 generation for train/val/test:
  - `create_hdf5_files.py`
- GPU host sync for new loaded folder:
  - `common.sh` syncs `loaded_classifier_nofilt`

### 2.2 Preprocessing Alignment Improved

Implemented in both repos:
- Live-style context crop around event indices:
  - `extract_production_like_trace(...)`
- Online filtering before scaling in live inference path:
  - crop -> filter (detrend/taper/highpass|bandpass) -> scale -> model

Relevant files:
- `arces_classification/src/Live.py`
- `ml_array_data_classification/src/live.py`

Observed impact from test run:
- Before fix (live-style subset): `acc 0.4333`, `macro_f1 0.3434`
- After fix: `acc 0.7333`, `macro_f1 0.6902`

### 2.3 Remaining Structural Mismatch

Open gap:
- `arces_classification` live-style validation still imports local `src/Live.py` directly.
- To reach full SSoT, callbacks/scripts must route through an adapter backed by
  `ml_array_data_classification` code.

## 3) Hard Decisions

1. Live inference single source of truth = `ml_array_data_classification`.
2. `arces_classification` must stop owning independent live logic behavior.
3. During transition, two configs are acceptable, but shared keys need explicit governance.
4. Any "live-style validation" in training must call inference behavior from the inference repo, not re-implement it.
5. Live scripts in `arces_classification` should be compatibility shims only during migration, then removed after one stable release cycle.

## 4) Migration Plan

## Phase 0 (Done): Baseline and Discrepancy Reproduction

Completed:
- Frozen baseline artifacts and phase-0 report
- Live-style callback integration
- Discrepancy decomposition (windowing + preprocessing)

Primary finding:
- Random-crop validation inflates metrics relative to production-like context windowing.

## Phase 1: Infrastructure Wiring for Inference SSoT

Goal:
- Ensure GPU run environment always includes a pinned copy of `ml_array_data_classification`.

Tasks:
- Extend `arces_classification/common.sh` to sync inference repo into GPU workdir, e.g.:
  - source: `${INFERENCE_REPO_DIR:-$PROJECT_DIR/../ml_array_data_classification}`
  - target: `$BASE_DIR/inference/ml_array_data_classification`
- Record pinned inference commit hash in run artifacts.
- Fail fast if inference repo path is missing when live-parity validation is enabled.

Status:
- `common.sh` now syncs the inference repo into `$BASE_DIR/inference/ml_array_data_classification`
  when `SYNC_INFERENCE_REPO=1` (default), and writes `.inference_commit`.
- `run_live.sh` now defaults to `live_via_ml_array.py`, which delegates to
  `ml_array_data_classification/inference.py`.
- Legacy live scripts in `arces_classification` now act as compatibility wrappers only.

Exit criteria:
- Every training/eval run can import inference code from synced pinned repo copy.

## Phase 2: Adapter Layer in Training Repo

Goal:
- Replace direct live logic calls in training repo with a stable adapter that delegates to inference repo.

Tasks:
- Add `arces_classification/src/inference_adapter.py`:
  - load `LiveClassifier` from synced inference repo path
  - expose stable `predict_live_style(...)` interface
  - validate required shared config keys
- Switch training live-style callback and phase0 baseline script to adapter use.

Exit criteria:
- No live inference behavior is reimplemented in training callbacks/scripts.

## Phase 3: Dataset Contract Unification

Goal:
- Make inference repo understand the new nofilt dataset contract.

Tasks in `ml_array_data_classification`:
- Add `data.source_variant` support (`filtered|nofilt`)
- Add `data.min_year`, `data.max_year`, optional dropped labels policy for offline HDF5 evaluation
- Update `src/load_process_hdf5.py` file discovery to support `eventclass_nofilt_*` patterns
- Keep backward compatibility with filtered inputs

Status:
- Implemented in `ml_array_data_classification`:
  - `config/data_config.yaml` now includes/uses `source_variant`, year bounds, and `drop_event_labels`.
  - `src/load_process_hdf5.py` now parses `eventclass_nofilt_*` and filtered variants via config.

Exit criteria:
- Inference offline evaluation can run on `nofilt` and `filtered` with explicit config, no code edits.

## Phase 4: Config Governance (Two Configs Without Chaos)

Goal:
- Keep two config files for now, but remove drift risk.

Policy:
- `arces_classification/config/data_config.yaml` owns:
  - dataset construction/splits
  - training-specific augmentation/optimizer/callbacks
- `ml_array_data_classification/config/data_config.yaml` owns:
  - runtime inference endpoints and operational defaults

Shared keys must match (governed list):
- `data.sample_rate`
- `data.model_threshold`
- `live.length`
- `live.step`
- `live.sample_rate`
- `live.event_buffer`
- `filters.use_filters`
- `filters.detrend`
- `filters.taper`
- `filters.taper_max_percentage`
- `filters.highpass_or_bandpass`
- `filters.band_kwargs.*`
- `filters.high_kwargs.*`
- `scaling.scaler_type`
- `scaling.global_or_local`
- `scaling.per_channel`

Implementation approach:
- Add a small bridge script in `arces_classification` to emit inference overrides (YAML/JSON) from training config for shared keys only.
- Add a parity check script to compare shared-key snapshots from both repos and fail on drift.

Exit criteria:
- Reproducible parity without manual copy/paste between config files.

## Phase 5: Duplicate Cleanup

Goal:
- Reduce long-term maintenance by deleting/archiving duplicated live behavior in `arces_classification`.

Tasks:
- Deprecate direct live entrypoints in training repo for production usage.
- Keep compatibility wrappers (`gbf_iter_torch.py`, `gbf_iter.py`, `gbf_live.py`) that delegate to `ml_array_data_classification/inference.py` and emit deprecation notices.
- Keep only adapter + tests for live-parity evaluation.
- Delete compatibility wrappers after one release cycle with no consumers.
- Archive legacy TensorFlow paths and dead scripts after one cleanup cycle.

Exit criteria:
- Clear repo boundaries with no ambiguous live-code ownership.

## 5) Dataset Contract (New `nofilt`)

Source inputs:
- Data files under `eventclass_nofilt_<year>_*.hdf5`
- Metadata with arrivals update CSVs

Loaded artifacts:
- `loaded_classifier_nofilt/train_{full|debug}_data.h5`
- `loaded_classifier_nofilt/val_{full|debug}_data.h5`
- `loaded_classifier_nofilt/test_{full|debug}_data.h5`
- matching `*_index_list.pkl`
- `key_dicts.pkl`

Current split policy:
- val year: `2024`
- test year: `2025`

Operational requirement:
- both repos must read/write this contract consistently for parity experiments.

## 6) Immediate Next Actions

1. Validate `common.sh` inference-repo sync in an end-to-end run and persist commit hash in run artifacts.
2. Add adapter module in `arces_classification` and route live-style callback through it.
3. Add shared-config parity checker and make it part of phase-0/phase-1 run checklist.
4. Re-run baseline on fixed subset and confirm parity of preprocessing path end-to-end.
5. Remove compatibility wrappers after one stable release cycle.

## 7) Definition of Done

This initiative is complete when all are true:
- `ml_array_data_classification` is the only maintained live inference implementation.
- `arces_classification` live-style validation delegates to inference repo code.
- New nofilt dataset contract is supported in both repos.
- Shared config keys are parity-checked automatically.
- Promotion gates rely on live-style metrics and reproducible pinned commits.
