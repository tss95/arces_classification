# Optional Single-Head Output Plan (Toggleable)

## Goal

Introduce an **optional** single-head output mode while keeping current dual-head behavior as the default and preserving an easy rollback path.

- Current (default): dual-head output (`detector`, `classifier`)
- New (optional): single-head output (3-class: `noise`, `earthquake`, `explosion`)

## Hard Requirements

1. Toggleable at config level without code edits.
2. Dual mode remains default during rollout.
3. Switching back to dual mode must be one config change.
4. Companion repo (`ml_array_data_classification`) must be tracked for parity before any default switch.

## Config Contract

Add model-level config:

- `head_mode: "dual" | "single"` (default: `"dual"`)

Behavior:

- `dual`: unchanged model outputs and loss/metrics pipeline.
- `single`: one multiclass output head, multiclass loss/metrics.

## Implementation Scope (This Repo)

### Phase 1: Core Toggle in Training Graph

Files:

- `config/models/alexnet.yaml`
- `src/Models_torch.py`
- `src/Loop_torch.py`
- `src/BeamDataset.py`
- `src/BeamModule.py`
- `code_test.py`

Work:

- Add `head_mode` config read.
- Build/output either dual heads or single multiclass head.
- Add mode-aware loss and metrics in Lightning loop.
- Make labels mode-aware (include single-label target).
- Keep checkpoints/monitor metric selection mode-aware.

Acceptance:

- `head_mode=dual`: no behavior regression.
- `head_mode=single`: training + validation run without code edits.

### Phase 2: Inference/Validation Callbacks

Files:

- `src/InferenceUtils.py`
- `src/Callbacks.py`
- `src/Live.py` (local live utilities used by callbacks)

Work:

- Make label translation support dual and single prediction payloads.
- Make confusion/live-style validation callbacks support both modes.
- Keep plotting non-blocking if mode-specific outputs differ.

Acceptance:

- Validation callbacks run in both modes.
- Final string labels (`noise|earthquake|explosion`) remain consistent.

### Phase 3: Docs + Rollout Safety

Files:

- `README.md`
- `docs/handover_overview.md` (or successor docs)

Work:

- Document `head_mode` usage and rollback.
- Document metric names and checkpoint monitor differences by mode.
- Keep dual as default until sign-off.

Acceptance:

- Operator can switch modes via config only.
- Operator can revert to dual immediately.

## Rollback Plan

If issues appear in single mode:

1. Set `head_mode: dual` in model config.
2. Re-run with existing dual checkpoints and monitor keys.
3. Keep single-mode artifacts isolated from dual-mode model selection.

No destructive migration steps are planned.

## Companion Repo Watchlist (`ml_array_data_classification`)

Before changing defaults, mirror mode support in companion repo:

- `/staff/tord/Workspace/ml_array_data_classification/src/models.py`
- `/staff/tord/Workspace/ml_array_data_classification/src/loop.py`
- `/staff/tord/Workspace/ml_array_data_classification/src/inference_utils.py`
- `/staff/tord/Workspace/ml_array_data_classification/src/live.py`
- `/staff/tord/Workspace/ml_array_data_classification/inference.py`

Parity checks:

1. Same `head_mode` config contract.
2. Same label semantics and thresholds.
3. Same checkpoint loading expectations.
4. Same final string-label behavior for live outputs.

## Current Rollout Decision

- Implement single mode behind a toggle now.
- Keep `dual` as default until:
  1. internal metrics are validated, and
  2. companion repo parity is complete.
