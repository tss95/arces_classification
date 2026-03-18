# Production Sync + Parity Plan

Date: 2026-03-18  
Training repo: `arces_classification`  
Production repo: `ml_array_data_classification`

## Objective

Resolve prediction-performance mismatch between training and production by:
1. Updating `ml_array_data_classification` to Maikael's latest branch state.
2. Rebuilding any broken dependencies caused by that update.
3. Enforcing model-load parity so a model exported from training gives equivalent performance in production.

## Constraints and Working Style

- Pull from Maikael's branch first, regardless of whether it has been merged to `main/master`.
- Prefer existing production functions and patterns (minimal additions in production code).
- Keep MLOps-critical behavior explicit where needed (config integrity, validation checks, reproducibility).

## Phase 1: Safe Sync of Production Repo

1. Capture pre-sync snapshot:
   - `git status --short`
   - `git rev-parse HEAD`
   - `git branch -vv`
2. Fetch remote branches and identify:
   - Maikael branch name
   - Tord branch name
   - main/master branch state
3. Create/update a local tracking branch for Maikael's branch.
4. Pull latest commits from Maikael's branch.
5. Record post-sync snapshot:
   - active branch
   - new commit hash
   - short changelog (`git log --oneline --decorate -n 20`)

## Phase 2: Dependency and Contract Reconciliation

1. Compare dependency declarations and runtime assumptions:
   - Python/package versions
   - Config schema keys
   - Model artifact load paths
2. Repair breakages with minimal production changes:
   - Reuse Maikael's abstractions/functions wherever possible.
   - Add only targeted glue needed for compatibility and reproducibility.
3. Validate production smoke path:
   - model load
   - single prediction pass
   - no silent fallback/default misconfiguration

## Phase 3: Performance Parity Investigation

Primary hypothesis:
- Saved model `cfg` dictionary may contain empty/missing values that change runtime behavior between repos.

Checks (fixed model + fixed input slice):
1. Compare loaded `cfg` content field-by-field in both repos:
   - missing keys
   - empty strings/lists/dicts
   - type mismatches
2. Validate effective runtime config after defaults are applied:
   - filters
   - scaling
   - live window params
   - thresholds and class mappings
3. Verify preprocessing parity before model forward:
   - crop/window indices
   - filtering params
   - scaling mode and statistics
4. Compare post-forward steps:
   - logits/probabilities
   - thresholding
   - label mapping
5. Identify and patch root-cause divergence with minimal code delta.

## Deliverables

1. Production repo synced to Maikael branch with commit hash documented.
2. Dependency/runtime compatibility fixes applied.
3. Reproducible parity check showing aligned outputs/metrics for the same model and input.
4. Brief note in `docs/` summarizing final root cause and applied fix.

## Definition of Done

- Same model artifact produces materially equivalent predictions in both repos under the same input/config.
- No hidden defaulting from empty/missing `cfg` values.
- Production code remains minimal and aligned with Maikael's structure.

## Execution Log (2026-03-18)

### Completed

1. Production repo sync:
   - Remote branches found: `origin/master`, `origin/db-testing`.
   - Local production repo switched to `db-testing` and pulled to `1bb7a0f`.
2. Training repo delegation compatibility:
   - Updated `live_via_ml_array.py`, `gbf_iter.py`, `gbf_iter_torch.py`, `gbf_live.py` to resolve both:
     - `ml_array_classifier/inference.py` (new layout)
     - `inference.py` (legacy fallback)
3. Config/override integrity fixes:
   - Training `code_test.py` now serializes `OmegaConf` containers correctly (no empty `{}` artifacts from `DictConfig/ListConfig`).
   - Training handoff overrides now emit `inference`-compatible keys (mapped from `live`/`data` where needed).
   - Production `live.py` override extraction now supports both schema variants:
     - production-style `cfg.inference`
     - training-style `cfg.live` + `cfg.data` bridge keys.
4. Production policy guard:
   - Added explicit single-head enforcement in production model loading:
     - dual-head checkpoints now fail fast with a clear error.

### Verified

1. Dual-head checkpoint handling:
   - Production now rejects dual checkpoints by design (as required for phase-out).
2. Single-head checkpoint loading:
   - Production successfully loads a single-head handoff checkpoint from:
     - `/nobackup2/tord/arces_classification_pytorch/output/alexnet/analysis_test/handoff_bundle/model.ckpt`
3. `cfg` empty-value hypothesis:
   - Raw checkpoint `cfg` values were present.
   - Empty-value issue was confirmed in exported sidecar snapshots due serialization of OmegaConf types, now fixed.

### Current Parity Result (Single-Head Evaluation Slice)

Validation subset: 300 fixed indices (`100` per class where available), same input files, same checkpoint.

- Training-side predictions vs production-side predictions:
  - raw label agreement: `174/300` (`58.0%`)
  - mapped agreement (`not existing -> noise`): `282/300` (`94.0%`)
- Metrics on same truth labels:
  - training: accuracy `0.8867`, macro-F1 `0.8809`
  - production: accuracy `0.9000`, macro-F1 `0.8960`

### Root Cause Found for Remaining Drift

- Preprocessed model input arrays were confirmed bit-identical for inspected mismatch samples.
- Logits still differed between repos on the same input and checkpoint.
- Therefore the remaining discrepancy is from model implementation drift (`Models_torch.py` vs `ml_array_classifier/models.py`), not from data window/filter/scaler path.

### Additional Parity Hardening (2026-03-18)

1. Production model stem now reads and applies `model_cfg.dilations` (with safe defaults):
   - `ml_array_classifier/models.py` now matches training behavior for:
     - dilation parsing and validation
     - dilation-aware conv padding
     - passing dilation to `nn.Conv1d`
2. This closes a class of silent drift where dilated checkpoints were loaded in production as non-dilated models.
3. Changes were committed and pushed on branch `tord-db-testing-integration`:
   - commit `00c107d` — "Align AlexNet dilation handling with training checkpoints"
4. Training-side checkpoint analysis load (`model_analysis.py`) now applies checkpoint cfg/model_cfg overrides before model construction and clears known missing optional keys (`dilations`) to avoid stale local defaults leaking into checkpoint evaluation.

### Maikael Environment Test Flow (Reproduced)

Observed test entrypoint:
- `ml_array_data_classification/ml_array_classifier/test.py --repro-check`

This is now the recommended parity gate command:

```bash
cd /nobackup2/tord/ml_array_data_classification/ml_array_classifier
python test.py \
  --model-path /path/to/handoff_bundle \
  --loaded-path /path/to/loaded_classifier_nofilt \
  --repro-check --repro-acc-tol 0.02
```

Validated in current environment:

1. Baseline single-head handoff bundle:
   - model: `/nobackup2/tord/arces_classification_pytorch/output/alexnet/analysis_test/handoff_bundle`
   - result: `expected_acc=0.8956`, `actual_acc=0.8956`, `pass=True`
2. Dilated single-head handoff bundle:
   - model: `/nobackup2/tord/arces_classification_pytorch/output/alexnet_dilate_late/analysis_test/handoff_bundle`
   - result: `expected_acc=0.9400`, `actual_acc=0.9400`, `pass=True`

### Next Decision

To fully close parity, choose one:
1. Make production `models.py` numerically equivalent to training model implementation for the production-supported single-head path.
2. Make training parity checks/validation run through production model code as the source of truth and gate on that output.
