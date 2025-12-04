from global_config import logger, cfg, model_cfg
import numpy as np
from haikunator import Haikunator

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
from src.Transforms import RandomCropTransform, MinMaxPerChannelTransform, ScalingTransform
from src.Scaler_torch import Scaler
from src.Analysis_torch import Analysis
from src.DataVerification import Verficiation
from src.Callbacks import ConfusionMatrixLogger
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
    batched_labels = {'detector': [], 'classifier': []}
    batched_ids = [item[2] for item in batch]
    
    for _, labels, _ in batch:
        batched_labels['detector'].append(labels['detector'])
        batched_labels['classifier'].append(labels['classifier'])

    batched_data = torch.stack(batched_data)
    batched_labels['detector'] = torch.stack(batched_labels['detector'])
    batched_labels['classifier'] = torch.stack(batched_labels['classifier'])

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
    

if __name__ == "__main__":
    if cfg.data.debug:
        cfg.optimizer.max_epochs = 1
    multi_gpu = False
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            cfg.optimizer.batch_size = int(cfg.optimizer.batch_size / num_gpus)
            logger.info(f"Detected {num_gpus} GPUs, setting batch size to {cfg.optimizer.batch_size}")
            multi_gpu = True
  
    run_id = datetime.datetime.now().strftime("%y%m%d_%H%M%S") if not cfg.run_id else cfg.run_id
    mp.set_start_method('spawn', force=True)
    os.environ['WANDB_START_METHOD'] = 'thread'
    logger.info(f"Run ID: {run_id}, debug mode: {cfg.data.debug}, num_epochs: {cfg.optimizer.max_epochs}, multi_gpu: {multi_gpu}")
    cfg = prepare_folders_paths_cfg(run_id, cfg, make_folders=True)
    mem_before = print_mem_before()
    label_dict, classifier_label_map, detector_label_map, class_weights = load_preprocessed_data_dict(cfg).values()
    
        # Print memory usage after the operation
    mem_after = print_mem_after()
    print_mem_diff(mem_before, mem_after)
    
    logger.info(f"Class weights: {class_weights}")
    input_shape = (3, cfg.augment.random_crop_kwargs.timesteps)

        
    transforms_by_sample = [RandomCropTransform(cfg)]
    # Build transforms without scaling first so we can fit a global scaler when needed
    transforms_by_set = setup_transforms(cfg, add_scaling=False)
    scaler = Scaler(cfg)
    logger.info("Setting up modules:")
    mem_before = print_mem_before()
    data_module = BeamModule(transforms_by_sample, transforms_by_set, cfg)
    data_module.setup()
    if scaler.requires_fit:
        # Fit on the training loader without scaling applied yet
        scaler.fit_loader(data_module.train_dataloader())
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
    
    logger.info("Setting up model")
    mem_before = print_mem_before()
    model = get_model(input_shape,
                      detector_metrics_list,
                      classifier_metrics_list,
                      detector_label_map, 
                      classifier_label_map,
                      class_weights["detector"], 
                      class_weights["classifier"],
                      cfg)
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
    callbacks.append(ModelCheckpoint(monitor = "val_classifier_f1", mode = "max", save_top_k = 5, 
                                         dirpath = cfg.project_paths.output_folders.model_save_folder,
                                         save_weights_only= True,
                                         filename = model_name + "_{epoch}_{val_total_loss:.2f}"
                                         ))
    
    callbacks.append(ConfusionMatrixLogger(cfg))
    #logger.warning("Early stopping needs to be changed before full training")
    #callbacks.append(EarlyStopping(monitor = "val_total_loss", mode = "min", patience = cfg.callbacks.early_stopping_patience))
    logger.info("Setting up trainer")
    mem_before = print_mem_before()
    if not multi_gpu:
        trainer = Trainer(max_epochs = cfg.optimizer.max_epochs,
                        devices = -1 if torch.cuda.is_available() else 1,
                        precision="16-mixed",  # Mixed precision
                        accelerator="gpu" if torch.cuda.is_available() else "cpu",
                        num_nodes=1,
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
                        check_val_every_n_epoch=cfg.callbacks.validation_interval,
                        callbacks = callbacks,
                        logger = wandb_logger)
    mem_after = print_mem_after()
    print_mem_diff(mem_before, mem_after)
    
    trainer.fit(model, data_module)
    
    logger.info("Training complete")
    logger.info("Performing analysis")
    analysis = Analysis(model, data_module.val_dataloader(),label_dict, classifier_label_map, detector_label_map, data_module.full_list["val"], cfg)
    analysis.analysis_package(5)
    
    

    
    
