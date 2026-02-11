# Phase 0 Baseline Report (2026-02-11)

## Run Metadata

- Generated: `2026-02-11`
- Run ID: `20260211T084258Z`
- Baseline artifact folder:
  - `output/phase0_baseline/20260211T084258Z/`
  - `output/phase0_baseline/20260211T084258Z/phase0_baseline.json`
  - `output/phase0_baseline/20260211T084258Z/phase0_baseline.md`
- `arces_classification` commit: `329e5dd77c2daf59d16aa927865c8164abc0e9df`
- `ml_array_data_classification` commit: `215b93ee781fd28c3d0e303409204ce5e663b0a5`
- Data root (`DATA_DIR`): `/projects/restricted/Array/ML_DataSet`
- Checkpoint:
  - `/staff/tord/Workspace/arces_classification/output/alexnet/best/models/alexnet_small-rain-1048_epoch=97_val_total_loss=0.41.ckpt`

## Execution Command

From `arces_classification/`:

```bash
PROJECT_DIR=/staff/tord/Workspace/arces_classification \
DATA_DIR=/projects/restricted/Array/ML_DataSet \
python scripts/phase0_baseline.py --num-workers 8 --per-class 100 --max-events 300
```

## Standard Validation Baseline (Training Path)

- `val_total_loss`: `0.342531`
- `val_detector_accuracy`: `0.998794`
- `val_detector_f1`: `0.998794`
- `val_detector_auroc`: `0.999986`
- `val_classifier_accuracy`: `0.963966`
- `val_classifier_f1`: `0.981081`
- `val_classifier_auroc`: `0.893227`

## Live-Style Ensemble Baseline (Validation Slice)

Setup:
- Validation source: `val_full_data.h5` + `val_full_index_list.pkl`
- Subset policy: balanced slice, `100` per class (`noise`, `earthquake`, `explosion`), total `300` events.
- Inference mode: live-style windowing + ensemble voting via `LiveClassifier`.

Results:
- `sample_size`: `300`
- `elapsed_seconds`: `69.80`
- `events_per_second`: `4.298`
- `accuracy`: `0.586667`
- Macro F1: `0.541577`
- Weighted F1: `0.541577`

Per-class F1:
- `noise`: `0.668896`
- `earthquake`: `0.296875`
- `explosion`: `0.658960`

Confusion matrix (labels order: `noise`, `earthquake`, `explosion`):

```text
[[100,  0,  0],
 [ 65, 19, 16],
 [ 34,  9, 57]]
```

## Observation

The baseline reproduces a substantial train/eval-vs-live-style gap:
- strong standard validation metrics,
- materially lower live-style ensemble performance on the frozen validation slice.

This confirms the discrepancy and provides a reproducible Phase 0 anchor for subsequent remediation work.

## Discrepancy Deep-Dive (2026-02-11)

Additional ablations were run on the same balanced `300`-event validation subset to isolate sources of the gap.

### A) Preprocessing Ablation (Live-Style)

- Live-style current preprocessing (`extract_production_like_trace` + scaling + ensemble):
  - accuracy: `0.8333`
  - macro F1: `0.8172`
- Live-style plus extra validation transforms (`bandpass + taper` before scaling):
  - accuracy: `0.8133`
  - macro F1: `0.7899`

Result: adding `bandpass+taper` did **not** close the gap; it slightly reduced performance on this slice.

### B) Windowing Policy Ablation (Same Subset)

- `live_ensemble` (production-like context + sliding windows + ensemble):
  - accuracy: `0.8333`
  - macro F1: `0.8172`
- `single_randomcrop_valpipeline` (legacy validation-style random event crop + val transforms):
  - accuracy: `0.8933`
  - macro F1: `0.8914`
- `single_centerwindow_valpipeline` (production-like context, center 80s window + val transforms):
  - accuracy: `0.8333`
  - macro F1: `0.8149`

Result: `single_centerwindow` is nearly identical to `live_ensemble`, while `single_randomcrop` is materially higher.  
Primary remaining discrepancy source is **windowing policy** (random crop vs production-like center/context), not only filter/taper preprocessing.

### C) Ensemble Aggregation Check

- Majority-vote ensemble:
  - accuracy: `0.8333`
  - macro F1: `0.8172`
- Mean-probability aggregation (detector/classifier means, then threshold):
  - accuracy: `0.8433`
  - macro F1: `0.8296`

Result: mean-probability aggregation showed a small positive gain on this slice.
