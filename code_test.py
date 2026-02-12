from global_config import logger, cfg, model_cfg
import numpy as np
from haikunator import Haikunator
import argparse

from global_config import logger, cfg, model_cfg
import numpy as np
from haikunator import Haikunator
from src.Utils_torch import *
from src.BeamDataset import BeamDataset
from src.Models_torch import get_model
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer
from src.Transforms import RandomCropTransform, LiveStyleCenterCropTransform, MinMaxPerChannelTransform, ScalingTransform
from src.Scaler_torch import Scaler
from src.Analysis_torch import Analysis
from src.DataVerification import Verficiation
from src.Callbacks import ConfusionMatrixLogger, LiveStyleValidationCallback
from torchsummary import summary
import torch.multiprocessing as mp
from functools import partial
from torch.utils.data.distributed import DistributedSampler
from src.BeamModule import BeamModule
from pytorch_lightning.profilers import PyTorchProfiler
import psutil

try:
    import wandb
    wandb_available = True
    from pytorch_lightning.loggers import WandbLogger
except ImportError:
    wandb_available = False

# Moved outside and added an extra `transforms` parameter
def batch_collate_fn(batch, transforms=None):
    batched_data = [item[0] for item in batch]
    label_keys = list(batch[0][1].keys()) if batch else []
    batched_labels = {key: [] for key in label_keys}
    batched_ids = [item[2] for item in batch]
    
    for _, labels, _ in batch:
        for key in label_keys:
            batched_labels[key].append(labels[key])

    batched_data = torch.stack(batched_data)
    for key in label_keys:
        batched_labels[key] = torch.stack(batched_labels[key])

    if transforms:
        for transform in transforms:
            batched_data = transform(batched_data)

    return batched_data, batched_labels, batched_ids

def get_collate_fn_with_transforms(transforms=None):
    # Using `partial` to bind the transforms to the collate function
    return partial(batch_collate_fn, transforms=transforms)

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
    

def parse_trial_args():
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
    return parser.parse_args()


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
    trial_args = parse_trial_args()

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

    if cfg.data.debug and trial_args.max_epochs is None:
        cfg.optimizer.max_epochs = 1
    multi_gpu = False
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1 and not trial_args.single_gpu:
            cfg.optimizer.batch_size = int(cfg.optimizer.batch_size / num_gpus)
            logger.info(f"Detected {num_gpus} GPUs, setting batch size to {cfg.optimizer.batch_size}")
            multi_gpu = True
        elif num_gpus > 1 and trial_args.single_gpu:
            logger.info("Detected %s GPUs, forcing single-GPU mode due to --single-gpu.", num_gpus)
  
    run_id = datetime.datetime.now().strftime("%y%m%d_%H%M%S") if not cfg.run_id else cfg.run_id
    mp.set_start_method('spawn', force=True)
    os.environ['WANDB_START_METHOD'] = 'thread'
    logger.info(f"Run ID: {run_id}, debug mode: {cfg.data.debug}, num_epochs: {cfg.optimizer.max_epochs}, multi_gpu: {multi_gpu}")
    cfg = prepare_folders_paths_cfg(run_id, cfg, make_folders=True)
    mem_before = print_mem_before()
    key_dicts = load_preprocessed_data_dict(cfg)
    label_dict = key_dicts["label_dict"]
    classifier_label_map = key_dicts["classifier_label_map"]
    detector_label_map = key_dicts["detector_label_map"]
    class_weights = key_dicts["class_weights"]
    single_label_map = key_dicts.get(
        "single_label_map",
        {"noise": 0, "earthquake": 1, "explosion": 2},
    )
    
        # Print memory usage after the operation
    mem_after = print_mem_after()
    print_mem_diff(mem_before, mem_after)
    
    logger.info(f"Class weights: {class_weights}")
    input_shape = (3, cfg.augment.random_crop_kwargs.timesteps)

        
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
    # Build transforms without scaling first so we can fit a global scaler when needed
    transforms_by_set = setup_transforms(cfg, add_scaling=False)
    scaler = Scaler(cfg)
    logger.info("Setting up modules:")
    mem_before = print_mem_before()
    data_module = BeamModule(transforms_by_sample, transforms_by_set, cfg)
    data_module.setup()
    if scaler.requires_fit:
        # Fit on the training loader without scaling applied yet
        fit_transforms = transforms_by_set["train"] if getattr(cfg.data, "set_transforms_on_device", False) else None
        scaler.fit_loader(data_module.train_dataloader(), batch_transforms=fit_transforms)
    scaling_transform = ScalingTransform(scaler)
    for key in transforms_by_set:
        transforms_by_set[key].append(scaling_transform)
    logger.info("Datamodule created")
    mem_after = print_mem_after()
    print_mem_diff(mem_before, mem_after)
    #verify = Verficiation(dataloaders["val"], val_events, cfg)
    #verify.check_one_batch()
    #verify.check_raw_data()
    #verify.check_processed_data()
    
    
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
    monitor_metric = "val_classifier_f1" if head_mode == "dual" else "val_single_f1"
    callbacks.append(ModelCheckpoint(monitor = monitor_metric, mode = "max", save_top_k = 5, 
                                         dirpath = cfg.project_paths.output_folders.model_save_folder,
                                         save_weights_only= True,
                                         filename = model_name + "_{epoch}_{val_total_loss:.2f}"
                                         ))
    
    callbacks.append(ConfusionMatrixLogger(cfg))
    live_val_enabled = bool(getattr(cfg.callbacks, "live_style_validation", True))
    if live_val_enabled:
        live_val_interval = int(getattr(cfg.callbacks, "live_style_validation_interval", 1))
        live_val_per_class = int(getattr(cfg.callbacks, "live_style_val_per_class", 100))
        live_val_max_events = int(getattr(cfg.callbacks, "live_style_val_max_events", 300))
        callbacks.append(
            LiveStyleValidationCallback(
                cfg,
                scaler_state=scaler.state_dict(),
                n_epochs=live_val_interval,
                per_class=live_val_per_class,
                max_events=live_val_max_events,
            )
        )
        logger.info(
            "Enabled live-style validation callback: interval=%s per_class=%s max_events=%s",
            live_val_interval,
            live_val_per_class,
            live_val_max_events,
        )
    else:
        logger.info("Live-style validation callback disabled via cfg.callbacks.live_style_validation.")
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
                        limit_train_batches=limit_train_batches,
                        limit_val_batches=limit_val_batches,
                        check_val_every_n_epoch=cfg.callbacks.validation_interval,
                        callbacks = callbacks,
                        logger = wandb_logger)
    mem_after = print_mem_after()
    print_mem_diff(mem_before, mem_after)
    
    trainer.fit(model, data_module)
    
    logger.info("Training complete")
    if head_mode == "dual":
        logger.info("Performing analysis")
        analysis = Analysis(model, data_module.val_dataloader(),label_dict, classifier_label_map, detector_label_map, data_module.full_list["val"], cfg)
        analysis.analysis_package(5)
    else:
        logger.info("Skipping legacy Analysis_torch package for single-head mode.")
    
    

    
    
