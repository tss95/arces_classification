# Live Performance Tracking (W&B Best Epoch + Stability Window)

- Last updated: 2026-02-12 15:00 UTC
- W&B entity/project: `tordss-university-of-oslo/arces_classification`
- Metric source: W&B run history (`val_live_*`), including in-progress runs.
- Ranking target: best epoch by `val_live_accuracy` (epoch shown 1-indexed).
- Stability window:
  - `avg_pre5_acc` = mean `val_live_accuracy` over epochs `[peak-5, peak-1]`
  - `avg_post5_acc` = mean `val_live_accuracy` over epochs `[peak+1, peak+5]`
  - Same definition for `avg_pre5_eq_f1` / `avg_post5_eq_f1`
  - `n_post` = how many post-peak epochs currently available (important for running runs)

## Dilated Iterations (Latest)

| Run | Model | W&B state | Latest epoch | Peak epoch | Peak live acc | Peak EQ F1 | avg_pre5_acc | avg_post5_acc | avg_pre5_eq_f1 | avg_post5_eq_f1 | n_post | Latest-peak acc |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`zhjyznz3`](https://wandb.ai/tordss-university-of-oslo/arces_classification/runs/zhjyznz3) | `alexnet_dilate_late` | `finished` | 78 | 34 | 0.9400 | 0.9071 | 0.8531 | 0.8567 | 0.7404 | 0.7382 | 5 | -0.1144 |
| [`s8pshcwj`](https://wandb.ai/tordss-university-of-oslo/arces_classification/runs/s8pshcwj) | `alexnet_dilate_late_meanpool` | `finished` | 35 | 21 | 0.9078 | 0.8494 | 0.8556 | 0.8778 | 0.7337 | 0.7814 | 5 | -0.1011 |
| [`3dt6m3qp`](https://wandb.ai/tordss-university-of-oslo/arces_classification/runs/3dt6m3qp) | `alexnet_dilate_late_stride2` | `finished` | 85 | 30 | 0.9000 | 0.8289 | 0.8558 | 0.8389 | 0.7309 | 0.6870 | 5 | -0.0600 |
| [`0amc41hq`](https://wandb.ai/tordss-university-of-oslo/arces_classification/runs/0amc41hq) | `alexnet_dilate_late_stride2_skip_early_pool` | `finished` | 73 | 41 | 0.8844 | 0.7928 | 0.8396 | 0.8293 | 0.6877 | 0.6624 | 5 | -0.0767 |
| [`7x8kxry0`](https://wandb.ai/tordss-university-of-oslo/arces_classification/runs/7x8kxry0) | `alexnet_dilate_late` | `crashed` | - | - | - | - | - | - | - | - | 0 | - |

## Anchor Comparison (Finished)

| Run | Model | W&B state | Latest epoch | Peak epoch | Peak live acc | Peak EQ F1 | avg_pre5_acc | avg_post5_acc | avg_pre5_eq_f1 | avg_post5_eq_f1 | Latest-peak acc |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`nbc7govy`](https://wandb.ai/tordss-university-of-oslo/arces_classification/runs/nbc7govy) | `alexnet` | `finished` | 90 | 42 | 0.8956 | 0.8199 | 0.8111 | 0.8104 | 0.6251 | 0.6263 | -0.1044 |
| [`4gq8yffu`](https://wandb.ai/tordss-university-of-oslo/arces_classification/runs/4gq8yffu) | `alexnet_stride2_deeper` | `finished` | 71 | 22 | 0.8900 | 0.8107 | 0.8367 | 0.8098 | 0.7079 | 0.6181 | -0.0867 |

## Other Tracked Runs (Stability Pull)

| Run | Model | W&B state | Latest epoch | Peak epoch | Peak live acc | Peak EQ F1 | avg_pre5_acc | avg_post5_acc | avg_pre5_eq_f1 | avg_post5_eq_f1 | Latest-peak acc |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`bqdx56md`](https://wandb.ai/tordss-university-of-oslo/arces_classification/runs/bqdx56md) | `alexnet` | `finished` | 61 | 24 | 0.9400 | 0.9032 | 0.8300 | 0.8200 | 0.6801 | 0.6344 | -0.1433 |
| [`kgbyazzh`](https://wandb.ai/tordss-university-of-oslo/arces_classification/runs/kgbyazzh) | `alexnet_stride2_deeper` | `finished` | 37 | 17 | 0.9022 | 0.8364 | 0.8131 | 0.8251 | 0.6320 | 0.6506 | -0.0244 |
| [`ecnlywld`](https://wandb.ai/tordss-university-of-oslo/arces_classification/runs/ecnlywld) | `alexnet_stride2` | `finished` | 66 | 24 | 0.8700 | 0.7711 | 0.7593 | 0.7713 | 0.4612 | 0.4970 | -0.1433 |

## Best-Epoch Confusion Matrix Links

- `zhjyznz3` best epoch 34:
  [`val_live_confusion_matrix_216_ea03dd426e2ddf590bf8.png`](https://wandb.ai/tordss-university-of-oslo/arces_classification/runs/zhjyznz3/files/media/images/val_live_confusion_matrix_216_ea03dd426e2ddf590bf8.png)
- `s8pshcwj` best epoch 21:
  [`val_live_confusion_matrix_132_f67008b33d3816845cb0.png`](https://wandb.ai/tordss-university-of-oslo/arces_classification/runs/s8pshcwj/files/media/images/val_live_confusion_matrix_132_f67008b33d3816845cb0.png)
- `3dt6m3qp` best epoch 30:
  [`val_live_confusion_matrix_190_d78251660139712dc32a.png`](https://wandb.ai/tordss-university-of-oslo/arces_classification/runs/3dt6m3qp/files/media/images/val_live_confusion_matrix_190_d78251660139712dc32a.png)
- `0amc41hq` best epoch 41:
  [`val_live_confusion_matrix_261_c70355919b3cf2025f7a.png`](https://wandb.ai/tordss-university-of-oslo/arces_classification/runs/0amc41hq/files/media/images/val_live_confusion_matrix_261_c70355919b3cf2025f7a.png)
- `nbc7govy` best epoch 42:
  [`val_live_confusion_matrix_267_1c2d1b0f9df29c6d6fb4.png`](https://wandb.ai/tordss-university-of-oslo/arces_classification/runs/nbc7govy/files/media/images/val_live_confusion_matrix_267_1c2d1b0f9df29c6d6fb4.png)
- `4gq8yffu` best epoch 22:
  [`val_live_confusion_matrix_139_85c5ca95ce0d7b54fddf.png`](https://wandb.ai/tordss-university-of-oslo/arces_classification/runs/4gq8yffu/files/media/images/val_live_confusion_matrix_139_85c5ca95ce0d7b54fddf.png)

## Decision Note (`alexnet_dilate_late_stride2`)

- Both `stride2` variants are now finished.
- `alexnet_dilate_late_stride2` beat `stride2_skip_early_pool` on peak live acc (`0.9000` vs `0.8844`) and peak EQ F1 (`0.8289` vs `0.7928`).
- Both show post-peak decay (`avg_post5_acc < avg_pre5_acc` and final below peak), so they improve over some non-dilated anchors but do not beat `alexnet_dilate_late` peak (`0.9400`).
