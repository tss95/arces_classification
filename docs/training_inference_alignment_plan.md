# Training/Inference Convergence Plan

Date: 2026-02-11  
Scope: `arces_classification` (training-first) and `ml_array_data_classification` (inference-first)

## 1) Objective

Close the observed performance gap between:
- training/validation metrics in `arces_classification`, and
- live inference behavior in production-like usage.

At the same time, reduce duplicated code and make ownership clear:
- `arces_classification`: training, experimentation, offline evaluation.
- `ml_array_data_classification`: live/inference implementation.

## 2) Current-State Map (As-Is)

### 2.1 Active Paths

- Active training entry in `arces_classification`: `code_test.py`
- Active live entry in `arces_classification`: `gbf_iter_torch.py`
- Active inference entry in `ml_array_data_classification`: `inference.py`

### 2.2 High-Overlap Modules (Duplicated)

- `arces_classification/src/Loop_torch.py` ~= `ml_array_data_classification/src/loop.py` (functionally identical).
- `arces_classification/src/InferenceUtils.py` ~= `ml_array_data_classification/src/inference_utils.py` (near-identical).
- `arces_classification/src/Scaler_torch.py` ~= `ml_array_data_classification/src/scaler.py` (no material diff observed).
- `arces_classification/src/Live.py` ~= `ml_array_data_classification/src/live.py` with minor behavior drift.

### 2.3 Notable Drift Between Repos

- `arces_classification/src/Models_torch.py` has stricter timestep and positional-encoding checks than `ml_array_data_classification/src/models.py`.
- `arces_classification/project_setup.py` validates window/sample-rate consistency; `ml_array_data_classification/project_setup.py` currently does not.
- `arces_classification/src/Live.py` is slightly behind `ml_array_data_classification/src/live.py` in small robustness details (origin time handling, some guards, log style).

### 2.4 Legacy/TF Surface Still Present in `arces_classification`

Confirmed legacy TF paths still in tree and/or scripts:
- Scripts: `train.py`, `predict.py`, `gbf_iter.py`, `gbf_live.py`, `sweep_train.py`
- TF-specific modules: `src/Models.py`, `src/Loop.py`, `src/Callbacks_tf.py`, `src/Scaler_tf.py`, `src/S4.py`, `src/S4D.py`, `src/Generator.py`, `src/UMAPCallback.py`, `src/Analysis.py`
- Full duplicate subtree: `temp_run_dir/` (contains copied source/config)

### 2.5 Validation Reality Today

- Both repos use standard batch-level validation in `Loop_torch.py` / `loop.py`.
- No existing live-style ensemble validation pipeline is currently wired into validation loops.
- Live ensemble exists only in live inference (`LiveClassifier.prepare_multiple_intervals` + `ensamble_predict`).

## 3) Gap Hypotheses to Verify

Primary likely causes of train-vs-live discrepancy:
- Single-window validation vs multi-window ensemble voting in live inference.
- Preprocessing mismatch: training transforms vs live preprocessing/filtering path.
- Sampling/context mismatch: random crops in training vs event-window stepping in live.
- Scaling behavior mismatch: checkpoint scaler state handling and fallback behavior.
- Thresholding/label-map inconsistencies or different postprocessing paths.

## 4) Target-State Architecture

### 4.1 Ownership Boundaries

- `ml_array_data_classification` is the single source of truth for inference logic.
- `arces_classification` consumes inference behavior for:
  - offline live-parity evaluation,
  - release validation,
  - model quality checks before handoff.

### 4.2 Practical Integration Rule

- No new inference algorithms should be implemented directly in `arces_classification`.
- `arces_classification` should depend on a pinned inference implementation from `ml_array_data_classification` (commit/tag/version).

## 5) Phased Execution Plan

## Phase 0: Baseline and Freeze (Required)

Deliverables:
- Baseline report with:
  - standard validation metrics (current training path),
  - offline live-style metrics using the same checkpoint(s),
  - confusion matrices by detector/classifier/final label.
- Version pinning document:
  - dataset snapshot id,
  - checkpoint id(s),
  - config files used.

Tasks:
- Define a fixed evaluation set (val/test slice) and freeze it.
- Run current `arces_classification` validation.
- Run current `ml_array_data_classification` inference offline on same set.
- Store artifacts in `arces_classification/output/` with run id.

Exit criteria:
- Reproducible baseline metrics with exact configs and commit hashes.

## Phase 1: Inference Source-of-Truth Adoption

Deliverables:
- Integration decision record: how `arces_classification` consumes inference code.
- Pinned dependency strategy (recommended: commit pin + adapter module).

Recommended implementation:
- Add a thin adapter in `arces_classification` that delegates live-style prediction calls to `ml_array_data_classification` inference components.
- Keep adapter interface stable; avoid duplicating internals.

Tasks:
- Select integration method:
  - Option A: submodule/subtree vendoring with explicit pin (recommended for reproducibility).
  - Option B: packaged dependency pinned to commit/tag.
- Define shared inference contract:
  - input tensor shape/order,
  - scaler requirements,
  - label map schema,
  - output schema for per-window and ensemble outputs.

Exit criteria:
- `arces_classification` offline evaluation uses the inference implementation from `ml_array_data_classification` without copy-pasted logic.

## Phase 2: Ensemble Validation in Training Repo

Deliverables:
- New live-parity validation pipeline in `arces_classification` producing `val_live_*` metrics.
- Side-by-side metric logging:
  - `val_*` (single-window legacy),
  - `val_live_*` (ensemble/live-style).

Design:
- Add a dedicated evaluation stage (callback or post-epoch evaluator) that:
  - takes each validation example (preferably pre-crop/full context when available),
  - runs live-equivalent window slicing (`length`, `step`, `sample_rate`),
  - applies the same scaler path as inference,
  - performs ensemble vote + mean probability aggregation,
  - computes final metrics on aggregated labels.

Tasks:
- Implement evaluator module in `arces_classification` (training-owned code, inference-delegating calls).
- Ensure val loader can provide enough temporal context for windowing.
- Log metrics and confusion matrices for aggregated predictions.

Exit criteria:
- `val_live_*` metrics generated every configured validation interval.
- Metric definitions documented and reproducible.

## Phase 3: Preprocessing and Config Parity Hardening

Deliverables:
- Canonical parity checklist enforced in CI/script.
- Explicitly synchronized preprocessing/scaling path between training evaluation and live inference.

Tasks:
- Keep/extend `verify_live_train_parity.py` to include:
  - windowing params,
  - scaling mode and fitted-state checks,
  - filter settings used in live-style evaluation.
- Enforce fail-fast on missing/invalid scaler state where required.
- Resolve any remaining path/key inconsistencies (`project_paths`/`data_paths` alignment already improved in `arces`).

Exit criteria:
- Parity check fails on any material drift between training live-style eval and inference configuration.

## Phase 4: Duplicate and Legacy Cleanup

Deliverables:
- Reduced maintenance surface in `arces_classification`.
- Deprecated files either removed or moved to explicit archive location.

Cleanup matrix (proposed):

Keep in `arces_classification`:
- `code_test.py`, `src/BeamModule.py`, `src/BeamDataset.py`, `src/Models_torch.py`, `src/Loop_torch.py`, `src/Transforms.py`, `src/Utils_torch.py`, training analyses/tests/docs.

Delegate to `ml_array_data_classification`:
- live/inference core behavior (`live.py`, `inference_utils.py`, inference orchestration).

Deprecate/Archive in `arces_classification`:
- `train.py`, `predict.py`, `gbf_iter.py`, `gbf_live.py`, `sweep_train.py`
- `src/Models.py`, `src/Loop.py`, `src/Callbacks_tf.py`, `src/Scaler_tf.py`, TF-only modules
- `temp_run_dir/` duplicate subtree

Tasks:
- Update scripts/docs so no active workflow points at TF scripts.
- Add explicit deprecation notice before removal (one release cycle if needed).
- Remove dead imports and stale references.

Exit criteria:
- Default workflows are Torch + inference-delegated only.
- No active docs/scripts direct users to deprecated TF paths.

## Phase 5: Release Gating and Handoff

Deliverables:
- Release gate checklist for model promotion from training repo to inference repo.
- Ownership and runbook updates in docs.

Promotion gate (minimum):
- `val_live_*` meets target threshold.
- Live/train parity checks pass.
- Inference smoke test passes on pinned inference commit.
- Artifact package includes scaler state and metadata.

## 6) Testing and Verification Strategy

Required tests to add/update:
- Unit:
  - window slicing (count, boundaries, edge cases),
  - ensemble aggregation behavior,
  - label translation/threshold behavior.
- Integration:
  - offline live-style evaluation on a small fixed sample set,
  - parity check script in CI.
- Regression:
  - compare `val_live_*` trend across key checkpoints.

Recommended metric set:
- Final-label accuracy/F1,
- detector/classifier AUROC/F1,
- class-wise confusion matrices,
- calibration snapshot for ensemble mean probabilities.

## 7) Risks and Mitigations

Risk: over-coupling repos with fragile imports  
Mitigation: thin adapter + explicit version pin + compatibility tests.

Risk: no temporal context in val data for ensemble simulation  
Mitigation: ensure evaluator uses pre-crop/full windows or regenerate val artifacts with adequate context.

Risk: silent drift between repos after cleanup  
Mitigation: parity checks and promotion gates tied to pinned commit hashes.

Risk: cleanup removes still-used legacy path  
Mitigation: deprecation cycle + import/reference audit before deletion.

## 8) Immediate Next Actions (Order)

1. Approve integration method (submodule/subtree/package pin).  
2. Implement Phase 0 baseline report with fixed dataset/checkpoint ids.  
3. Implement Phase 2 live-style ensemble validation in `arces_classification` using inference-delegated logic.  
4. Run parity + regression metrics and set promotion thresholds.  
5. Execute Phase 4 cleanup once gates are green.

## 9) Definition of Done

This initiative is complete when:
- `arces_classification` is clearly training-first with no active TF inference/training paths.
- `ml_array_data_classification` is the sole maintained inference logic source.
- `arces_classification` validation includes live-style ensemble metrics (`val_live_*`) used in model promotion.
- The live-vs-training performance gap is reduced and tracked against a frozen baseline.
