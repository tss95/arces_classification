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
    key_dicts = load_preprocessed_data_dict(cfg)
    label_dict = key_dicts["label_dict"]
    classifier_label_map = key_dicts["classifier_label_map"]
    detector_label_map = key_dicts["detector_label_map"]
    class_weights = key_dicts["class_weights"]
    single_label_map = key_dicts.get("single_label_map", {"noise": 0, "earthquake": 1, "explosion": 2})
    
    logger.info(f"Class weights: {class_weights}")
    input_shape = (3, cfg.augment.random_crop_kwargs.timesteps)

        
    transforms_by_sample = [RandomCropTransform(cfg)]
    transforms_by_set = setup_transforms(cfg, add_scaling=False)
    scaler = Scaler(cfg)
    logger.info("Setting up modules:")
    data_module = BeamModule(transforms_by_sample, transforms_by_set, cfg)
    data_module.setup()
    ckpt = torch.load(cfg.pretrained_model_name, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if "scaler_state" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state"])
    elif scaler.requires_fit:
        fit_transforms = transforms_by_set["train"] if getattr(cfg.data, "set_transforms_on_device", False) else None
        scaler.fit_loader(data_module.train_dataloader(), batch_transforms=fit_transforms)
    scaling_transform = ScalingTransform(scaler)
    for key in transforms_by_set:
        transforms_by_set[key].append(scaling_transform)
    detector_metrics_list = ["accuracy", "precision", "recall", "f1", "auroc"]
    classifier_metrics_list = ["accuracy", "precision", "recall", "f1", "auroc"]
    model = get_model(input_shape,
                    detector_metrics_list,
                    classifier_metrics_list,
                    detector_label_map, 
                    classifier_label_map,
                    class_weights["detector"], 
                    class_weights["classifier"],
                    cfg,
                    single_label_map=single_label_map,
                    single_class_weights=class_weights.get("single"))
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    summary(model, input_shape)
    

    # Load the checkpoint
    # Load the state dict into the model
    # Make sure to use 'load_state_dict' instead of 'load_from_checkpoint'
    # Also, correct the typo from 'stict' to 'strict'
    model.load_state_dict(ckpt["state_dict"], strict=False)
    if "scaler_state" in ckpt:
        model.scaler_state = ckpt["scaler_state"]
    model.eval()
    
    trainer = Trainer(max_epochs = cfg.optimizer.max_epochs,
                        devices = -1 if torch.cuda.is_available() else 1,
                        precision="16-mixed",  # Mixed precision
                        accelerator="gpu" if torch.cuda.is_available() else "cpu",
                        num_nodes=1)
    
    val_result = trainer.validate(model, datamodule=data_module)
    
    for result in val_result:  # or test_result
        print(result)
    
    head_mode = str(getattr(model_cfg, "head_mode", "dual")).lower()
    if head_mode == "dual":
        analysis = Analysis(model, 
                            data_module.val_dataloader(), 
                            label_dict, 
                            classifier_label_map, 
                            detector_label_map, 
                            data_module.full_list["val"], 
                            cfg)
        analysis.analysis_package(5)
    else:
        logger.info("Skipping Analysis_torch package in single-head mode.")
