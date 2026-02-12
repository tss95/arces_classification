# First Single-Head Trial Run

Use `code_test.py` runtime overrides so no YAML edits are required.

## Recommended first trial (quick smoke on real pipeline)

```bash
PROJECT_DIR=/staff/tord/Workspace/arces_classification \
DATA_DIR=/projects/restricted/Array/ML_DataSet \
python code_test.py \
  --head-mode single \
  --single-gpu \
  --debug \
  --max-epochs 1 \
  --batch-size 256 \
  --num-workers 0 \
  --disable-wandb \
  --disable-live-val \
  --limit-train-batches 0.08 \
  --limit-val-batches 0.02 \
  --run-id single_head_trial_01
```

## Notes

- Keep `config/models/alexnet.yaml` default as `head_mode: "dual"` for rollback safety.
- This trial command overrides mode only for the current process.
- Default checkpoint monitor metric when live-style validation is enabled: `val_live_accuracy`.
- For this smoke command (`--disable-live-val`), monitor metric falls back to `val_single_f1`.
- These limits guarantee at least one train and one val batch on the current debug split.
