# Train Logic Port + `src/` Cleanup Matrix

Date: 2026-02-12  
Scope: `arces_classification` handoff to teammate workflow (`ml_array_data_classification/train.py` style)

## 1) What We Are Porting (and Why)

Your teammate's `train.py` now exists and is structurally simple, but the robust training behavior still lives in `arces_classification/code_test.py`.

Goal: keep the robust behavior, but present it in a simpler `train.py` flow.

## 2) Training Logic Inventory From `code_test.py`

## 2.1 Must Keep (P0)

These are core behaviors to preserve when populating teammate-facing `train.py`:

1. CLI runtime overrides (`--head-mode`, `--max-epochs`, `--batch-size`, `--num-workers`, `--debug`, etc.).
2. Reproducibility controls:
   - global seeding
   - deterministic toggle handling
   - dilation guard for deterministic slowdown.
3. DataModule pipeline:
   - split-aware sample transforms (`random_crop` / `live_center`)
   - batch transforms from `setup_transforms(...)`
   - scaler fitting when global scaling is enabled.
4. Model construction from `Models_torch.get_model(...)` + current head-mode support.
5. Trainer setup:
   - single vs multi-GPU handling
   - checkpointing.

## 2.2 Keep But Can Be Optional (P1)

1. `LiveStyleValidationCallback` (important for train/live parity, but can be a feature toggle).
2. Handoff bundle export (`model.ckpt`, inference overrides, repro spec).
3. Confusion matrix callback for final-label diagnostics.
4. Post-train `Analysis_torch` package.

## 2.3 Keep Only If Needed (P2)

1. Memory instrumentation prints.
2. Dummy forward pass NaN sanity block.
3. Legacy compatibility knobs that are rarely used in day-to-day runs.

## 3) Recommended Simplified `train.py` Shape

Populate teammate-facing `train.py` with this structure:

1. `parse_train_args()`
2. `apply_runtime_overrides(cfg, model_cfg, args)`
3. `setup_runtime(cfg, model_cfg, args)` (seed/determinism/gpu mode)
4. `build_data_module_and_scaler(cfg)`
5. `build_model(cfg, model_cfg, key_dicts, scaler, data_module)`
6. `build_callbacks(cfg, scaler, data_module, args)`
7. `build_trainer(cfg, callbacks, logger, args)`
8. `run_training(...)`
9. `run_post_training_exports(...)`

This keeps his preferred readability without dropping your training safeguards.

## 4) Active Torch Dependency Closure (Current)

Current `code_test.py` training path depends on this `src` set:

- `src/Utils_torch.py`
- `src/Transforms.py`
- `src/BeamModule.py`
- `src/BeamDataset.py`
- `src/Scaler_torch.py`
- `src/Models_torch.py`
- `src/Loop_torch.py`
- `src/Callbacks.py`
- `src/Analysis_torch.py`
- `src/Live.py` (indirectly via `LiveStyleValidationCallback`)
- `src/InferenceUtils.py` (indirectly via `src/Live.py`)

Everything else is currently legacy, optional, or side-path tooling.

## 5) `src/` Merge/Delete Matrix

## 5.1 Keep (Now)

- `src/BeamDataset.py`
- `src/BeamModule.py`
- `src/Callbacks.py`
- `src/InferenceUtils.py`
- `src/Live.py` (until callback/inference adapter change)
- `src/Loop_torch.py`
- `src/Models_torch.py`
- `src/Scaler_torch.py`
- `src/Transforms.py`
- `src/Utils_torch.py`
- `src/Analysis_torch.py` (optional but currently used by `code_test.py`)

## 5.2 Merge Candidates (Short Term)

1. `src/Callbacks.py`
   - Merge/remove unused custom callback framework classes:
     - `CustomCallback`
     - `EarlyStoppingCallback`
     - `ModelCheckpointCallback`
   - Keep only active Lightning callbacks.
   - Status: done on 2026-02-12 (unused classes removed, no runtime refs).

2. `src/Utils_torch.py`
   - Split into clearer modules:
     - `data_discovery_and_mapping.py` (raw eventclass mapping helpers)
     - `train_runtime_utils.py` (path prep / transform wiring)
   - This reduces "kitchen sink" complexity.

3. `src/Analysis_torch.py` + `model_analysis.py` (root script)
   - Consolidate analysis entrypoint ownership to one path.

## 5.3 Deleted Now (After Torch `train.py` Cutover)

These legacy TF modules were removed on 2026-02-12:

- `src/Analysis.py`
- `src/Callbacks_tf.py`
- `src/Generator.py`
- `src/LoadData.py`
- `src/Loop.py`
- `src/Models.py`
- `src/S4.py`
- `src/S4D.py`
- `src/Scaler_tf.py`
- `src/UMAPCallback.py`
- `src/Utils.py`

Related legacy entry scripts removed on 2026-02-12:

- `predict.py`
- `sweep_train.py`
- `train_torch.py`

## 5.4 Keep For Now, Revisit Later

- `src/Augment.py`  
  Reason: currently mixed TF/torch augmentation code; not in the primary code_test path, but still referenced by legacy modules.

- `src/DataVerification.py`  
  Reason: not in primary training flow now, but may be useful as optional debug tooling.

## 6) Remaining Legacy Cleanup Notes

- `train.py` is now a high-level wrapper that delegates to `code_test.py`.
- `run.sh` now defaults to `SCRIPT_NAME=train.py`.

## 7) First Safe Cleanup Pass (No Behavior Change)

1. Keep `code_test.py` as canonical behavior source.
2. Continue import cleanup / dead-code removal inside active files.
3. Create new simplified `train.py` that delegates to extracted helpers from `code_test.py`.
4. Only then deprecate old entrypoints and remove TF-bound `src` files.

## 8) Next Implementation Slice

1. Extract `code_test.py` main body into 5-8 helper functions (same behavior, clearer structure).
2. Introduce new `train.py` as thin orchestrator calling those helpers.
3. Keep `code_test.py` as wrapper during one transition cycle.
4. Start deleting legacy TF `src` files only after wrapper cycle is complete.
