from global_config import logger, cfg, model_cfg
import numpy as np
from haikunator import Haikunator
import argparse
import os
import random
import json
import shutil
import re
import datetime
import pickle
from pathlib import Path

from omegaconf import OmegaConf
from src.Utils_torch import (
    prepare_folders_paths_cfg,
)
from src.Models_torch import get_model
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
from src.Analysis_torch import Analysis
from src.Callbacks import ConfusionMatrixLogger, LiveStyleValidationCallback
from src.train_data import build_data_module_and_scaler
from torchsummary import summary
import torch.multiprocessing as mp
import psutil

try:
    import wandb
    wandb_available = True
    from pytorch_lightning.loggers import WandbLogger
except ImportError:
    wandb_available = False

def print_mem_before():
    mem_before = psutil.virtual_memory().available
    print(f"Available Memory Before: {mem_before / (1024 ** 3):.2f} GB")  
    return mem_before  

def print_mem_after():
    mem_after = psutil.virtual_memory().available
    print(f"Available Memory After: {mem_after / (1024 ** 3):.2f} GB")
    return mem_after

def print_mem_diff(mem_before, mem_after):
    print(f"Memory Difference: {(mem_before - mem_after) / (1024 ** 3):.2f} GB")
    

def setup_reproducibility(cfg, deterministic: bool):
    seed = int(getattr(cfg, "seed", 42))
    seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Recommended by PyTorch when deterministic algorithms are enabled on CUDA.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)
    logger.info(
        "Reproducibility setup: seed=%d deterministic=%s cudnn.benchmark=%s",
        seed,
        deterministic,
        torch.backends.cudnn.benchmark,
    )


def parse_bool_env(name: str):
    raw = os.getenv(name)
    if raw is None:
        return None
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    logger.warning("Ignoring invalid %s value '%s'. Expected true/false.", name, raw)
    return None


def model_has_dilation_gt_one(model_cfg) -> bool:
    raw = getattr(model_cfg, "dilations", None)
    if raw is None:
        return False
    values = raw
    if isinstance(values, str):
        normalized = values.strip().lower()
        if normalized in {"", "none"}:
            return False
        values = values.replace("[", "").replace("]", "")
        values = [v.strip() for v in values.split(",") if v.strip()]
    elif isinstance(values, (int, np.integer)):
        values = [int(values)]
    elif isinstance(values, np.ndarray):
        values = values.tolist()
    elif not isinstance(values, (list, tuple)):
        try:
            values = list(values)
        except TypeError:
            return False

    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value.lower() == "none":
            continue
        try:
            if int(value) > 1:
                return True
        except (TypeError, ValueError):
            logger.warning("Ignoring unparseable dilation value '%s' when evaluating deterministic guard.", value)
    return False


def _to_serializable_dict(obj):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): _to_serializable_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable_dict(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if hasattr(obj, "__dict__"):
        return {k: _to_serializable_dict(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return obj


def _extract_inference_cfg_overrides(cfg_dict):
    overrides = {}
    for top_key in ("model_name", "seed", "deterministic"):
        if top_key in cfg_dict:
            overrides[top_key] = cfg_dict[top_key]

    data_overrides = {}
    for key in ("model_threshold", "sample_rate", "window_seconds", "default_length_seconds"):
        if key in cfg_dict.get("data", {}):
            data_overrides[key] = cfg_dict["data"][key]
    if data_overrides:
        overrides["data"] = data_overrides

    if "live" in cfg_dict:
        overrides["live"] = cfg_dict["live"]
    if "filters" in cfg_dict:
        overrides["filters"] = cfg_dict["filters"]
    if "scaling" in cfg_dict:
        overrides["scaling"] = cfg_dict["scaling"]

    return overrides


def _extract_checkpoint_epoch(best_model_path):
    match = re.search(r"epoch=(\d+)", os.path.basename(str(best_model_path)))
    if not match:
        return None
    return int(match.group(1))


def _load_best_live_payload(output_root: Path, best_model_path: str):
    epoch_zero_indexed = _extract_checkpoint_epoch(best_model_path)
    if epoch_zero_indexed is None:
        return None, None
    payload_path = output_root / "live_style_validation" / f"epoch_{epoch_zero_indexed + 1:03d}.json"
    if not payload_path.exists():
        return None, None
    with open(payload_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload, str(payload_path)


def _build_repro_check_spec(cfg, monitor_metric, best_model_score, live_val_callback, best_live_payload):
    payload = best_live_payload if isinstance(best_live_payload, dict) else {}
    payload_metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}

    selected_indices = []
    selected_label_distribution = {}
    val_data_file = None
    val_index_file = None
    if live_val_callback is not None:
        selected_indices = [int(i) for i in getattr(live_val_callback, "selected_indices", [])]
        selected_label_distribution = dict(getattr(live_val_callback, "selected_label_distribution", {}))
        val_h5_path = getattr(live_val_callback, "val_h5_path", None)
        val_index_path = getattr(live_val_callback, "val_index_path", None)
        val_data_file = os.path.basename(val_h5_path) if val_h5_path else None
        val_index_file = os.path.basename(val_index_path) if val_index_path else None

    if not selected_indices and isinstance(payload.get("selected_indices"), list):
        selected_indices = [int(i) for i in payload["selected_indices"]]
    if not selected_label_distribution and isinstance(payload.get("selected_label_distribution"), dict):
        selected_label_distribution = dict(payload["selected_label_distribution"])
    if val_data_file is None and payload.get("val_data_file"):
        val_data_file = str(payload.get("val_data_file"))
    if val_index_file is None and payload.get("val_index_file"):
        val_index_file = str(payload.get("val_index_file"))

    expected_accuracy = None
    if isinstance(payload_metrics, dict) and "val_live_accuracy" in payload_metrics:
        expected_accuracy = float(payload_metrics["val_live_accuracy"])
    elif str(monitor_metric) == "val_live_accuracy" and best_model_score is not None:
        expected_accuracy = float(best_model_score)

    split = "debug" if bool(getattr(cfg.data, "debug", False)) else "full"
    if val_data_file is None:
        val_data_file = f"val_{split}_data.h5"
    if val_index_file is None:
        val_index_file = f"val_{split}_index_list.pkl"

    # Deterministic fallback: reconstruct the selected subset exactly as LiveStyleValidationCallback.
    if not selected_indices:
        val_index_path = os.path.join(cfg.data_paths.loaded_path, val_index_file)
        if os.path.exists(val_index_path):
            try:
                with open(val_index_path, "rb") as f:
                    val_index_list = pickle.load(f)
                by_label = {}
                for idx, rec in enumerate(val_index_list):
                    by_label.setdefault(str(rec[3]), []).append(idx)

                rng = random.Random(int(cfg.seed))
                per_class = int(getattr(cfg.callbacks, "live_style_val_per_class", 300))
                max_events = int(getattr(cfg.callbacks, "live_style_val_max_events", 900))
                reconstructed = []
                for label in sorted(by_label.keys()):
                    candidates = by_label[label][:]
                    rng.shuffle(candidates)
                    reconstructed.extend(candidates[:per_class])
                rng.shuffle(reconstructed)
                if max_events > 0:
                    reconstructed = reconstructed[:max_events]
                selected_indices = [int(i) for i in reconstructed]
                if selected_indices and not selected_label_distribution:
                    dist = {}
                    for idx in selected_indices:
                        label = str(val_index_list[idx][3])
                        dist[label] = int(dist.get(label, 0) + 1)
                    selected_label_distribution = dist
            except Exception as exc:
                logger.warning("Failed to reconstruct live-style subset from %s: %s", val_index_path, exc)

    labels_order = payload.get("labels_order", ["noise", "earthquake", "explosion"])

    return {
        "version": "live_style_repro_v1",
        "metric": "val_live_accuracy",
        "expected_accuracy": expected_accuracy,
        "default_tolerance_abs_accuracy": 0.02,
        "sample_size": int(len(selected_indices)),
        "selected_indices": selected_indices,
        "selected_label_distribution": selected_label_distribution,
        "labels_order": labels_order,
        "eval_files": {
            "val_data_file": val_data_file,
            "val_index_file": val_index_file,
        },
        "reference_metrics": payload_metrics if isinstance(payload_metrics, dict) else {},
    }


def export_handoff_bundle(cfg, model_cfg, checkpoint_callback, run_id, live_val_callback=None):
    best_model_path = str(getattr(checkpoint_callback, "best_model_path", "") or "")
    if not best_model_path:
        logger.warning("No best checkpoint path available; skipping handoff bundle export.")
        return None
    if not os.path.exists(best_model_path):
        logger.warning("Best checkpoint path does not exist (%s); skipping handoff bundle export.", best_model_path)
        return None

    output_root = Path(cfg.project_paths.output_folder)
    bundle_dir = output_root / "handoff_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_checkpoint_path = bundle_dir / "model.ckpt"
    shutil.copy2(best_model_path, bundle_checkpoint_path)
    best_live_payload, best_live_payload_path = _load_best_live_payload(output_root, best_model_path)
    if best_live_payload_path is not None:
        shutil.copy2(best_live_payload_path, bundle_dir / "reference_live_style_epoch.json")

    checkpoint = torch.load(best_model_path, map_location="cpu", weights_only=False)
    cfg_dict = checkpoint.get("cfg")
    if not isinstance(cfg_dict, dict):
        cfg_dict = _to_serializable_dict(cfg)
    model_cfg_dict = checkpoint.get("model_cfg")
    if not isinstance(model_cfg_dict, dict):
        model_cfg_dict = _to_serializable_dict(model_cfg)

    cfg_dict = _to_serializable_dict(cfg_dict)
    model_cfg_dict = _to_serializable_dict(model_cfg_dict)
    scaler_state = checkpoint.get("scaler_state", None)

    inference_overrides = {
        "cfg_overrides": _extract_inference_cfg_overrides(cfg_dict),
        "model_cfg_overrides": model_cfg_dict,
    }
    OmegaConf.save(
        config=OmegaConf.create(inference_overrides),
        f=str(bundle_dir / "inference_overrides.yaml"),
    )
    OmegaConf.save(
        config=OmegaConf.create(cfg_dict),
        f=str(bundle_dir / "training_cfg_snapshot.yaml"),
    )
    OmegaConf.save(
        config=OmegaConf.create(model_cfg_dict),
        f=str(bundle_dir / "training_model_cfg_snapshot.yaml"),
    )
    if scaler_state is not None:
        torch.save(scaler_state, bundle_dir / "scaler_state.pt")

    best_model_score = getattr(checkpoint_callback, "best_model_score", None)
    if isinstance(best_model_score, torch.Tensor):
        best_model_score = float(best_model_score.detach().cpu().item())
    elif isinstance(best_model_score, np.generic):
        best_model_score = float(best_model_score.item())

    monitor_metric = str(getattr(checkpoint_callback, "monitor", "unknown"))
    bundle_contents = [
        "model.ckpt",
        "inference_overrides.yaml",
        "training_cfg_snapshot.yaml",
        "training_model_cfg_snapshot.yaml",
        "scaler_state.pt",
        "manifest.json",
        "README.txt",
    ]
    if best_live_payload_path is not None:
        bundle_contents.append("reference_live_style_epoch.json")

    repro_spec = _build_repro_check_spec(
        cfg=cfg,
        monitor_metric=monitor_metric,
        best_model_score=best_model_score,
        live_val_callback=live_val_callback,
        best_live_payload=best_live_payload,
    )
    if repro_spec.get("expected_accuracy") is not None and repro_spec.get("sample_size", 0) > 0:
        with open(bundle_dir / "repro_check_spec.json", "w", encoding="utf-8") as f:
            json.dump(repro_spec, f, indent=2, sort_keys=True)
        bundle_contents.append("repro_check_spec.json")
    else:
        logger.warning(
            "Repro check spec incomplete (expected_accuracy=%s sample_size=%s); skipping repro_check_spec.json",
            repro_spec.get("expected_accuracy"),
            repro_spec.get("sample_size"),
        )

    manifest = {
        "exported_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "run_id": run_id,
        "model_name": cfg.model_name,
        "monitor_metric": monitor_metric,
        "best_model_score": best_model_score,
        "best_checkpoint_source_path": best_model_path,
        "best_checkpoint_bundle_path": str(bundle_checkpoint_path),
        "bundle_contents": bundle_contents,
    }
    with open(bundle_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    readme = (
        "ARCES Handoff Bundle\n"
        "====================\n\n"
        "Use this directory as a self-contained inference handoff package.\n\n"
        "Recommended in ml_array_data_classification:\n"
        "1. Set pretrained_model_name to this bundle's model.ckpt (or the bundle directory).\n"
        "2. Run inference.py; checkpoint and inference_overrides.yaml are used to align runtime config.\n\n"
        "Files:\n"
        "- model.ckpt: best Lightning checkpoint with state_dict/cfg/model_cfg/scaler_state\n"
        "- inference_overrides.yaml: inference-relevant cfg and full model_cfg overrides\n"
        "- training_cfg_snapshot.yaml: full training cfg snapshot\n"
        "- training_model_cfg_snapshot.yaml: full model cfg snapshot\n"
        "- scaler_state.pt: scaler state extracted from checkpoint (convenience copy)\n"
        "- repro_check_spec.json: optional reproducibility check spec (live-style subset + expected accuracy)\n"
        "- manifest.json: metadata about this bundle export\n"
    )
    with open(bundle_dir / "README.txt", "w", encoding="utf-8") as f:
        f.write(readme)

    logger.info("Exported handoff bundle to %s", bundle_dir)
    return str(bundle_dir)


def parse_train_args():
    parser = argparse.ArgumentParser(description="Train/evaluate ARCES classification model.")
    parser.add_argument("--head-mode", choices=["dual", "single"], default=None,
                        help="Override model head mode for this run.")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Override optimizer.max_epochs for this run.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override optimizer.batch_size for this run.")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Override cfg.num_workers for dataloaders.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode (uses reduced data subset).")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Override cfg.run_id.")
    parser.add_argument("--single-gpu", action="store_true",
                        help="Force single-GPU trainer mode even if multiple GPUs are available.")
    parser.add_argument("--disable-wandb", action="store_true",
                        help="Disable Weights & Biases logging for this run.")
    parser.add_argument("--disable-live-val", action="store_true",
                        help="Disable live-style validation callback for this run.")
    parser.add_argument("--limit-train-batches", type=float, default=None,
                        help="PyTorch Lightning limit_train_batches override.")
    parser.add_argument("--limit-val-batches", type=float, default=None,
                        help="PyTorch Lightning limit_val_batches override.")
    parser.add_argument("--deterministic", dest="deterministic", action="store_true",
                        help="Force deterministic training behavior for reproducibility.")
    parser.add_argument("--non-deterministic", dest="deterministic", action="store_false",
                        help="Disable deterministic behavior to maximize speed.")
    parser.add_argument(
        "--allow-slow-dilated-deterministic",
        action="store_true",
        help="Keep deterministic mode even when model dilations > 1 (can be extremely slow).",
    )
    parser.set_defaults(deterministic=None)
    return parser.parse_args()


def apply_runtime_overrides(trial_args):
    if trial_args.head_mode is not None:
        model_cfg.head_mode = str(trial_args.head_mode).lower()
    if trial_args.max_epochs is not None:
        cfg.optimizer.max_epochs = int(trial_args.max_epochs)
    if trial_args.batch_size is not None:
        cfg.optimizer.batch_size = int(trial_args.batch_size)
    if trial_args.num_workers is not None:
        cfg.num_workers = int(trial_args.num_workers)
    if trial_args.debug:
        cfg.data.debug = True
    if trial_args.run_id is not None:
        cfg.run_id = trial_args.run_id
    if trial_args.disable_wandb:
        cfg.wandb.active = False
    if trial_args.disable_live_val:
        cfg.callbacks.live_style_validation = False


def do_preamble():
    trial_args = parse_train_args()
    apply_runtime_overrides(trial_args)

    env_deterministic = parse_bool_env("DETERMINISTIC_OVERRIDE")
    deterministic_default = bool(getattr(cfg, "deterministic", True))
    deterministic_run = (
        deterministic_default if env_deterministic is None else bool(env_deterministic)
    )
    if trial_args.deterministic is not None:
        deterministic_run = bool(trial_args.deterministic)
    if (
        deterministic_run
        and model_has_dilation_gt_one(model_cfg)
        and not trial_args.allow_slow_dilated_deterministic
    ):
        logger.warning(
            "Detected model dilations > 1 with deterministic mode enabled. "
            "This combination can be extremely slow on this stack; overriding to deterministic=False. "
            "Use --allow-slow-dilated-deterministic to force deterministic anyway."
        )
        deterministic_run = False
    cfg.deterministic = deterministic_run
    setup_reproducibility(cfg, deterministic=deterministic_run)

    if cfg.data.debug and trial_args.max_epochs is None:
        cfg.optimizer.max_epochs = 1
    multi_gpu = False
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1 and not trial_args.single_gpu:
            cfg.optimizer.batch_size = int(cfg.optimizer.batch_size / num_gpus)
            logger.info("Detected %s GPUs, setting batch size to %s", num_gpus, cfg.optimizer.batch_size)
            multi_gpu = True
        elif num_gpus > 1 and trial_args.single_gpu:
            logger.info("Detected %s GPUs, forcing single-GPU mode due to --single-gpu.", num_gpus)

    run_id = datetime.datetime.now().strftime("%y%m%d_%H%M%S") if not cfg.run_id else cfg.run_id
    mp.set_start_method("spawn", force=True)
    os.environ["WANDB_START_METHOD"] = "thread"
    logger.info(
        "Run ID: %s, debug mode: %s, num_epochs: %s, multi_gpu: %s, deterministic: %s",
        run_id,
        cfg.data.debug,
        cfg.optimizer.max_epochs,
        multi_gpu,
        cfg.deterministic,
    )
    prepare_folders_paths_cfg(run_id, cfg, make_folders=True)
    return trial_args, run_id, multi_gpu


def compute_single_class_weights_from_index_list(index_list):
    label_counts = {"noise": 0, "earthquake": 0, "explosion": 0}
    for rec in index_list:
        label = str(rec[3])
        if label in label_counts:
            label_counts[label] += 1
    total = sum(label_counts.values())
    if total == 0:
        return {"noise": 1.0, "earthquake": 1.0, "explosion": 1.0}
    return {
        label: (total / count if count > 0 else 0.0)
        for label, count in label_counts.items()
    }
    

if __name__ == "__main__":
    trial_args, run_id, multi_gpu = do_preamble()
    mem_before = print_mem_before()
    key_dicts, data_module, scaler, _, _ = build_data_module_and_scaler(cfg)
    label_dict = key_dicts["label_dict"]
    classifier_label_map = key_dicts["classifier_label_map"]
    detector_label_map = key_dicts["detector_label_map"]
    class_weights = key_dicts["class_weights"]
    single_label_map = key_dicts.get(
        "single_label_map",
        {"noise": 0, "earthquake": 1, "explosion": 2},
    )
    mem_after = print_mem_after()
    print_mem_diff(mem_before, mem_after)
    
    logger.info(f"Class weights: {class_weights}")
    input_shape = (3, cfg.augment.random_crop_kwargs.timesteps)

    wandb_logger = None
    haikunator = Haikunator()
    model_name = f"{cfg.model_name}_{haikunator.haikunate()}"
    if wandb_available and cfg.wandb.active:
        config_dict = {}
        for key, value in vars(cfg).items():
            config_dict[key] = value
        wandb_logger = WandbLogger(name = model_name, project=cfg.wandb.project, entity= cfg.wandb.entity, config=config_dict)
    
    detector_metrics_list = ["accuracy", "precision", "recall", "f1", "auroc"]
    classifier_metrics_list = ["accuracy", "precision", "recall", "f1", "auroc"]
    head_mode = str(getattr(model_cfg, "head_mode", "dual")).lower()
    if head_mode not in {"dual", "single"}:
        raise ValueError(f"Unsupported model head_mode '{head_mode}'. Use 'dual' or 'single'.")
    single_class_weights = class_weights.get("single", None) if isinstance(class_weights, dict) else None
    if head_mode == "single" and single_class_weights is None:
        single_class_weights = compute_single_class_weights_from_index_list(data_module.full_list["train"])
        logger.info("Computed single-head class weights from train index list: %s", single_class_weights)
    
    logger.info("Setting up model")
    mem_before = print_mem_before()
    model = get_model(input_shape,
                      detector_metrics_list,
                      classifier_metrics_list,
                      detector_label_map, 
                      classifier_label_map,
                      class_weights["detector"], 
                      class_weights["classifier"],
                      single_label_map=single_label_map,
                      single_class_weights=single_class_weights,
                      cfg=cfg)
    model.scaler_state = scaler.state_dict()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    summary(model, input_size=input_shape)
    mem_after = print_mem_after()
    print_mem_diff(mem_before, mem_after)
    
    
    # Generate dummy input
    dummy_input = torch.randn(128, *input_shape, device="cuda" if torch.cuda.is_available() else "cpu")

    # Perform a forward pass
    with torch.no_grad():
        model.eval()
        outputs = model(dummy_input)

    # Check if any of the outputs are NaN
    for key, output in outputs.items():
        if torch.isnan(output).any():
            print(f"NaN detected in output: {key}")
        else:
            print(f"No NaN detected in output: {key}")
    
    
    callbacks = []
    callbacks.append(ConfusionMatrixLogger(cfg))
    live_val_enabled = bool(getattr(cfg.callbacks, "live_style_validation", True))
    live_val_callback = None
    if live_val_enabled:
        live_val_interval = int(getattr(cfg.callbacks, "live_style_validation_interval", 1))
        live_val_per_class = int(getattr(cfg.callbacks, "live_style_val_per_class", 300))
        live_val_max_events = int(getattr(cfg.callbacks, "live_style_val_max_events", 900))
        live_val_callback = LiveStyleValidationCallback(
            cfg,
            scaler_state=scaler.state_dict(),
            n_epochs=live_val_interval,
            per_class=live_val_per_class,
            max_events=live_val_max_events,
        )
        callbacks.append(live_val_callback)
        logger.info(
            "Enabled live-style validation callback: interval=%s per_class=%s max_events=%s",
            live_val_interval,
            live_val_per_class,
            live_val_max_events,
        )
    else:
        logger.info("Live-style validation callback disabled via cfg.callbacks.live_style_validation.")

    base_monitor_metric = "val_classifier_f1" if head_mode == "dual" else "val_single_f1"
    monitor_metric = "val_live_accuracy" if live_val_enabled else base_monitor_metric
    logger.info("Model checkpoint monitor metric: %s", monitor_metric)
    checkpoint_callback = ModelCheckpoint(monitor = monitor_metric, mode = "max", save_top_k = 5, 
                                          dirpath = cfg.project_paths.output_folders.model_save_folder,
                                          save_weights_only= True,
                                          filename = model_name + "_{epoch}_{val_total_loss:.2f}"
                                          )
    callbacks.append(checkpoint_callback)
    #logger.warning("Early stopping needs to be changed before full training")
    #callbacks.append(EarlyStopping(monitor = "val_total_loss", mode = "min", patience = cfg.callbacks.early_stopping_patience))
    logger.info("Setting up trainer")
    mem_before = print_mem_before()
    limit_train_batches = (
        float(trial_args.limit_train_batches)
        if trial_args.limit_train_batches is not None
        else 1.0
    )
    limit_val_batches = (
        float(trial_args.limit_val_batches)
        if trial_args.limit_val_batches is not None
        else 1.0
    )
    single_gpu_devices = 1 if trial_args.single_gpu else (-1 if torch.cuda.is_available() else 1)
    if not multi_gpu:
        trainer = Trainer(max_epochs = cfg.optimizer.max_epochs,
                        devices = single_gpu_devices,
                        precision="16-mixed",  # Mixed precision
                        accelerator="gpu" if torch.cuda.is_available() else "cpu",
                        num_nodes=1,
                        deterministic=cfg.deterministic,
                        limit_train_batches=limit_train_batches,
                        limit_val_batches=limit_val_batches,
                        check_val_every_n_epoch=cfg.callbacks.validation_interval,
                        callbacks = callbacks,
                        logger = wandb_logger)
    else:
        trainer = Trainer(max_epochs = cfg.optimizer.max_epochs,
                        devices = -1,
                        accelerator = "gpu",
                        strategy = "ddp",
                        num_nodes = 1,
                        precision="16-mixed",  # Mixed precision
                        deterministic=cfg.deterministic,
                        limit_train_batches=limit_train_batches,
                        limit_val_batches=limit_val_batches,
                        check_val_every_n_epoch=cfg.callbacks.validation_interval,
                        callbacks = callbacks,
                        logger = wandb_logger)
    mem_after = print_mem_after()
    print_mem_diff(mem_before, mem_after)
    
    trainer.fit(model, data_module)
    
    logger.info("Training complete")
    if bool(getattr(cfg.callbacks, "export_handoff_bundle", True)):
        export_handoff_bundle(cfg, model_cfg, checkpoint_callback, run_id, live_val_callback=live_val_callback)
    if head_mode == "dual":
        logger.info("Performing analysis")
        analysis = Analysis(model, data_module.val_dataloader(),label_dict, classifier_label_map, detector_label_map, data_module.full_list["val"], cfg)
        analysis.analysis_package(5)
    else:
        logger.info("Skipping legacy Analysis_torch package for single-head mode.")
