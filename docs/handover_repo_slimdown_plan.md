# ARCES -> ML Array Handover Slimdown Plan

Date: 2026-02-12  
Scope: `/staff/tord/Workspace/arces_classification` with reference to `/staff/tord/Workspace/ml_array_data_classification`

Companion matrix:
- `docs/train_logic_port_and_src_cleanup_matrix.md` (detailed train-logic port checklist + `src/` keep/merge/delete mapping)

## 1) Goal

Prepare `arces_classification` for ownership handoff by:

1. Converging to a single clean training entrypoint (`train.py`) with clear runtime flow.
2. Replacing the current `BeamDatasetHDF5` loading path with a loader strategy aligned to ML-array style (raw per-year eventclass HDF5 inputs rather than pre-baked train/val `.h5` blobs).
3. Reducing repository sprawl from PyTorch migration leftovers so navigation and maintenance are straightforward.

## 2) Current State (Verified)

## 2.1 Training Entry Points

- Canonical entrypoint is now `train.py` (high-level wrapper).
- `train.py` delegates to `code_test.py`, which still contains the full training implementation during transition.
- Legacy scripts removed:
  - `train_torch.py`
  - `sweep_train.py`
  - `predict.py`

## 2.2 Data Loading Split

- Current training path:
  - `src/BeamModule.py` -> `src/BeamDataset.py::BeamDatasetHDF5`
  - Expects pre-generated files:
    - `*_data.h5`
    - `*_index_list.pkl`
    - `key_dicts.pkl`
  - Generated via `create_hdf5_files.py` + `src/Utils_torch.py`.

- ML-array path (latest pulled `master`):
  - `train.py` now exists in `ml_array_data_classification` (commit `4a6f50e`, 2026-02-12).
  - Loader logic centered in `src/load_process_hdf5.py` for direct eventclass files + metadata mapping.
  - `inference.py` remains the fully wired runtime entrypoint; `train.py` currently provides the preferred script shape to mirror.

## 2.3 Redundant Footprint

- `temp_run_dir` duplicates many source/config files and is currently very large (~5.2 GB locally; 120 files, 37 tracked by git).
- Legacy TF modules remain mixed with torch modules in `src/`.
- Script naming and roles are inconsistent for a handoff scenario.

## 3) Target End State

1. One canonical training entrypoint: `train.py` (PyTorch Lightning).
2. One canonical training data path: direct/raw eventclass loader (no mandatory intermediate `*_full_data.h5` build step).
3. Minimal compatibility wrappers only where necessary.
4. Clear directory responsibilities:
  - `src/` only active code paths.
  - `docs/` concise operator/maintainer documentation.
  - Legacy/deprecated assets removed or archived out-of-tree.

## 4) Phased Execution Plan

## Phase 0: Guardrails and Inventory Freeze

Work:

- Create a migration branch dedicated to handoff cleanup.
- Snapshot baseline training behavior (metrics + runtime config + best checkpoint path).
- Pin loader contract assumptions:
  - Event labels
  - Sample shape/orientation
  - Start/end index semantics
  - Year split behavior

Exit criteria:

- Baseline run metadata captured and committed in docs/artifacts.

## Phase 1: Introduce New Canonical `train.py` (Torch)

Work:

- Build new `train.py` by refactoring `code_test.py` into clearer sections:
  - config/CLI override parsing
  - reproducibility setup
  - datamodule/dataloader setup
  - model setup
  - trainer/callback wiring
  - handoff bundle export
- Keep CLI runtime toggles currently used in operations.
- Keep `train.py` as high-level entrypoint and `code_test.py` as implementation while internals are extracted.

Formatting objective:

- Follow the simpler, explicit script style your colleague prefers:
  - clear main guard
  - explicit argument parser
  - no hidden side effects at import time
  - predictable control flow

Status:

- `run.sh` now defaults to `SCRIPT_NAME=train.py`.

## Phase 2: Replace `BeamDatasetHDF5` Path with Raw Loader Strategy

Work:

- Implement a new dataset path (suggested names):
  - `BeamDatasetRawEventClass` (or equivalent) in `src/BeamDataset.py`
  - corresponding datamodule wiring in `src/BeamModule.py`
- Reuse/adapt loader behavior from ML-array `src/load_process_hdf5.py`:
  - year/file discovery by naming convention
  - metadata join logic
  - label normalization/drop policy
  - induced-event handling
- Preserve current transform pipeline (`sample` + `batch` transforms) and scaler fitting workflow.
- Maintain compatibility for train/val/test split policy from config.

Implementation constraints:

- Avoid reopening HDF5 file handles on every sample where possible; prefer worker-local cached handles.
- Keep tensor orientation consistent with model expectations `(C, T)`.

Exit criteria:

- Training no longer depends on `create_hdf5_files.py` artifacts.
- New loader can run both debug and full modes from raw eventclass files.

## Phase 3: Compatibility Window and Validation

Work:

- Keep old and new data paths behind a short-lived toggle (`data.loader_mode: preprocessed|raw`) for comparison.
- Run A/B validation on fixed seed:
  - class counts
  - sample count per split
  - metric parity trend (allowing expected drift from sampling differences)
  - runtime/memory profile

Exit criteria:

- Raw loader mode becomes default.
- Preprocessed mode marked deprecated with a removal date.

## Phase 4: Repo Slimdown (Post-Validation)

Work (high confidence removals first):

- Remove `temp_run_dir` after confirming no runtime dependency.
- Remove duplicate/legacy training entrypoints once wrapper period ends.
- Separate legacy TF assets from active torch path:
  - remove from root flow and docs
  - optionally archive in dedicated `legacy_tf/` branch/tag if history retention is required

Exit criteria:

- Root-level script list is minimal and role-based.
- New maintainer can identify:
  - one train command
  - one inference/live command
  - one config source for each concern

## 5) Proposed Keep/Deprecate Matrix

Keep (active):

- `train.py` (new torch canonical)
- `src/BeamModule.py` (updated raw-loader path)
- `src/BeamDataset.py` (updated dataset classes)
- `src/Utils_torch.py` (or split into loader utility module)
- `create_hdf5_files.py` (temporary only if fallback path retained)

Deprecate then remove:

- `code_test.py` (once extracted helpers are complete and `train.py` owns implementation directly)

Delete candidate (after verification):

- `temp_run_dir/`

Needs explicit decision before removal:

- TF-heavy source modules in `src/` (`Models.py`, `Loop.py`, `Generator.py`, `Callbacks_tf.py`, `Scaler_tf.py`, `S4.py`, `S4D.py`, etc.)

## 6) Risks and Mitigations

Risk: Data semantic drift between preprocessed and raw loader.  
Mitigation: deterministic A/B subset comparison and label/index assertions.

Risk: Slower dataloading from raw files.  
Mitigation: worker-local HDF5 handle caching, tuned `num_workers`, pinned memory profiling.

Risk: Hidden dependencies on old scripts in ops tooling.  
Mitigation: compatibility wrappers for one cycle + explicit deprecation logs.

Risk: Colleague `train.py` is present but still partially wired (contains TODOs).  
Mitigation: mirror its script structure/style, but keep training behavior anchored to the validated `code_test.py` path until raw-loader migration is complete.

## 7) Definition of Done

Handoff-ready when all are true:

1. `train.py` is the only documented training entrypoint.
2. Raw eventclass loading is the default training path.
3. `BeamDatasetHDF5` preprocessed dependency is removed or disabled by default.
4. `temp_run_dir` and other validated duplicates are gone.
5. README + onboarding docs match actual commands and ownership boundaries.

## 8) Immediate Next Actions

1. Extract remaining `code_test.py` internals into helper functions (`do_preamble()`, data/model/trainer builders).
2. Add `data.loader_mode` toggle and scaffold raw-loader dataset class.
3. Port/reuse ML-array loader logic into ARCES training path with minimal divergence.
4. Run first parity check (sample counts + one short training smoke run).
5. Remove `temp_run_dir` in a dedicated cleanup commit once confirmed unused.
