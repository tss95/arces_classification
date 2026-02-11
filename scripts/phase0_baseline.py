#!/usr/bin/env python3
"""Phase 0 baseline runner.

Produces a frozen baseline artifact with:
1) Standard validation metrics from the training validation path.
2) Offline live-style ensemble metrics on a fixed validation slice.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import h5py
import numpy as np
import torch
from pytorch_lightning import Trainer, seed_everything
from sklearn.metrics import classification_report, confusion_matrix

# Ensure repository root is importable when running from scripts/.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from global_config import cfg, logger
from src.BeamModule import BeamModule
from src.Live import LiveClassifier
from src.Models_torch import get_model
from src.Scaler_torch import Scaler
from src.Transforms import RandomCropTransform, LiveStyleCenterCropTransform, ScalingTransform
from src.Utils_torch import load_preprocessed_data_dict, setup_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 0 baseline metrics.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-workers", type=int, default=8, help="Dataloader workers for baseline run.")
    parser.add_argument(
        "--per-class",
        type=int,
        default=100,
        help="Number of events per class for live-style ensemble subset.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=300,
        help="Maximum total events for live-style ensemble subset.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Optional output directory. Defaults to PROJECT_DIR/output/phase0_baseline/<timestamp>/",
    )
    return parser.parse_args()


def ensure_env() -> Dict[str, str]:
    project_dir = os.environ.get("PROJECT_DIR")
    data_dir = os.environ.get("DATA_DIR")
    if not project_dir:
        raise EnvironmentError("PROJECT_DIR is not set.")
    if not data_dir:
        raise EnvironmentError("DATA_DIR is not set.")
    return {"project_dir": project_dir, "data_dir": data_dir}


def run_git(repo_path: str, *args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", repo_path, *args], text=True).strip()
    except Exception:
        return "unknown"


def normalize_label(value) -> str:
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return str(value.item())
        return str(value.tolist())
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return str(value[0])
        return str(list(value))
    return str(value)


def pick_live_subset(index_list: Sequence[Sequence], per_class: int, max_events: int, seed: int) -> List[int]:
    by_label: Dict[str, List[int]] = defaultdict(list)
    for idx, rec in enumerate(index_list):
        by_label[str(rec[3])].append(idx)

    rng = random.Random(seed)
    selected: List[int] = []
    for label in sorted(by_label.keys()):
        candidates = by_label[label][:]
        rng.shuffle(candidates)
        selected.extend(candidates[:per_class])

    rng.shuffle(selected)
    if max_events > 0:
        selected = selected[:max_events]
    return selected


def build_model_and_data(args: argparse.Namespace):
    cfg.num_workers = int(args.num_workers)
    cfg.wandb.active = False

    key_dicts = load_preprocessed_data_dict(cfg)
    classifier_label_map = key_dicts["classifier_label_map"]
    detector_label_map = key_dicts["detector_label_map"]
    class_weights = key_dicts["class_weights"]

    train_sample_transform = RandomCropTransform(cfg)
    val_sample_mode = str(getattr(cfg.data, "validation_sample_mode", "random_crop")).lower()
    if val_sample_mode == "live_center":
        val_sample_transform = LiveStyleCenterCropTransform(cfg)
    elif val_sample_mode == "random_crop":
        val_sample_transform = RandomCropTransform(cfg)
    else:
        raise ValueError(
            f"Unsupported data.validation_sample_mode='{val_sample_mode}'. "
            "Use one of: ['random_crop', 'live_center']."
        )
    transforms_by_sample = {
        "train": [train_sample_transform],
        "val": [val_sample_transform],
    }
    transforms_by_set = setup_transforms(cfg, add_scaling=False)
    scaler = Scaler(cfg)

    data_module = BeamModule(transforms_by_sample, transforms_by_set, cfg)
    data_module.setup()

    project_dir = Path(os.environ["PROJECT_DIR"])
    ckpt_path = Path(cfg.pretrained_model_name)
    if not ckpt_path.is_absolute():
        ckpt_path = project_dir / ckpt_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if "scaler_state" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state"])
    elif scaler.requires_fit:
        logger.warning("Checkpoint missing scaler_state; fitting scaler from train loader for baseline.")
        scaler.fit_loader(data_module.train_dataloader())

    scaling_transform = ScalingTransform(scaler)
    for split in transforms_by_set:
        transforms_by_set[split].append(scaling_transform)

    input_shape = (3, cfg.augment.random_crop_kwargs.timesteps)
    detector_metrics = ["accuracy", "precision", "recall", "f1", "auroc"]
    classifier_metrics = ["accuracy", "precision", "recall", "f1", "auroc"]

    model = get_model(
        input_shape,
        detector_metrics,
        classifier_metrics,
        detector_label_map,
        classifier_label_map,
        class_weights["detector"],
        class_weights["classifier"],
        cfg,
    )
    model.load_state_dict(ckpt["state_dict"], strict=False)
    if "scaler_state" in ckpt:
        model.scaler_state = ckpt["scaler_state"]
    model.eval()

    return model, scaler, data_module, ckpt_path


def run_standard_validation(model, data_module) -> Dict[str, float]:
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )
    results = trainer.validate(model, datamodule=data_module, verbose=False)
    if not results:
        return {}
    return {k: float(v) for k, v in results[0].items()}


def run_live_style_eval(model, scaler, args: argparse.Namespace) -> Dict[str, object]:
    split = "debug" if cfg.data.debug else "full"
    val_h5 = Path(cfg.data_paths.loaded_path) / f"val_{split}_data.h5"
    val_index = Path(cfg.data_paths.loaded_path) / f"val_{split}_index_list.pkl"
    if not val_h5.exists():
        raise FileNotFoundError(f"Validation HDF5 not found: {val_h5}")
    if not val_index.exists():
        raise FileNotFoundError(f"Validation index list not found: {val_index}")

    with open(val_index, "rb") as handle:
        index_list = pickle.load(handle)

    selected = pick_live_subset(index_list, args.per_class, args.max_events, args.seed)
    if not selected:
        raise RuntimeError("No events selected for live-style evaluation.")

    label_maps = {
        "detector": {0: "noise", 1: "event"},
        "classifier": {0: "earthquake", 1: "explosion"},
    }
    live_model = LiveClassifier(model, scaler, label_maps, cfg)

    y_true: List[str] = []
    y_pred: List[str] = []
    labels = ["noise", "earthquake", "explosion"]

    t0 = time.time()
    old_level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        with h5py.File(val_h5, "r") as handle:
            data = handle["data"]
            for i in selected:
                trace = data[i]
                rec = index_list[i]
                truth = str(rec[3])
                trace = live_model.extract_production_like_trace(
                    trace,
                    label=truth,
                    start_idx=rec[4],
                    end_idx=rec[5],
                )
                pred, _, _, _, _ = live_model.predict(trace)
                y_true.append(truth)
                y_pred.append(normalize_label(pred))
    finally:
        logger.setLevel(old_level)
    elapsed = time.time() - t0

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

    chosen_dist: Dict[str, int] = defaultdict(int)
    for idx in selected:
        chosen_dist[str(index_list[idx][3])] += 1

    return {
        "sample_size": len(selected),
        "elapsed_seconds": elapsed,
        "events_per_second": len(selected) / elapsed if elapsed > 0 else 0.0,
        "selected_label_distribution": dict(chosen_dist),
        "labels_order": labels,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def write_outputs(out_dir: Path, payload: Dict[str, object]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "phase0_baseline.json"
    md_path = out_dir / "phase0_baseline.md"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    std = payload["standard_validation_metrics"]
    live = payload["live_style_metrics"]
    lines = [
        "# Phase 0 Baseline",
        "",
        f"- Generated UTC: `{payload['generated_utc']}`",
        f"- Arces commit: `{payload['git']['arces_classification']}`",
        f"- Inference repo commit: `{payload['git']['ml_array_data_classification']}`",
        f"- Data dir: `{payload['env']['data_dir']}`",
        f"- Checkpoint: `{payload['checkpoint_path']}`",
        "",
        "## Standard Validation",
    ]
    for key in sorted(std.keys()):
        lines.append(f"- `{key}`: `{std[key]:.6f}`")
    lines.extend(
        [
            "",
            "## Live-Style Ensemble (Validation Slice)",
            f"- sample_size: `{live['sample_size']}`",
            f"- elapsed_seconds: `{live['elapsed_seconds']:.2f}`",
            f"- events_per_second: `{live['events_per_second']:.3f}`",
            f"- selected_label_distribution: `{live['selected_label_distribution']}`",
            "",
            "Detailed confusion matrix and classification report are in `phase0_baseline.json`.",
        ]
    )
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    env = ensure_env()

    seed_everything(args.seed, workers=True)
    torch.set_grad_enabled(False)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(env["project_dir"]) / "output" / "phase0_baseline" / timestamp

    logger.info("Phase 0 baseline started. Output: %s", out_dir)
    model, scaler, data_module, ckpt_path = build_model_and_data(args)
    std_metrics = run_standard_validation(model, data_module)
    live_metrics = run_live_style_eval(model, scaler, args)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "env": env,
        "checkpoint_path": str(ckpt_path),
        "config": {
            "model_name": cfg.model_name,
            "window_seconds": cfg.data.window_seconds,
            "sample_rate": cfg.data.sample_rate,
            "live_length": cfg.live.length,
            "live_step": cfg.live.step,
            "model_threshold": cfg.data.model_threshold,
            "scaling": {
                "scaler_type": cfg.scaling.scaler_type,
                "global_or_local": cfg.scaling.global_or_local,
                "per_channel": bool(cfg.scaling.per_channel),
            },
            "val_years": list(cfg.data.val_years),
        },
        "git": {
            "arces_classification": run_git(env["project_dir"], "rev-parse", "HEAD"),
            "ml_array_data_classification": run_git(
                str(Path(env["project_dir"]).parent / "ml_array_data_classification"), "rev-parse", "HEAD"
            ),
        },
        "run_params": {
            "seed": args.seed,
            "num_workers": args.num_workers,
            "live_eval_per_class": args.per_class,
            "live_eval_max_events": args.max_events,
        },
        "standard_validation_metrics": std_metrics,
        "live_style_metrics": live_metrics,
    }

    write_outputs(out_dir, payload)
    logger.info("Phase 0 baseline finished.")
    logger.info("Wrote %s", out_dir / "phase0_baseline.json")
    logger.info("Wrote %s", out_dir / "phase0_baseline.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
