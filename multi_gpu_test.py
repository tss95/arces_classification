from global_config import logger, cfg, model_cfg
import numpy as np
from haikunator import Haikunator

from global_config import logger, cfg, model_cfg
import numpy as np
from haikunator import Haikunator
from src.Utils_torch import *
from src.BeamDataset import BeamDataset
from src.Models_torch import get_model
from src.BeamModule import BeamModule
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer
from src.Transforms import RandomCropTransform, MinMaxPerChannelTransform
from src.Scaler_torch import Scaler
from src.Analysis_torch import Analysis
from src.DataVerification import Verficiation
from torchsummary import summary
import torch.multiprocessing as mp
from functools import partial
from torch.utils.data.distributed import DistributedSampler

try:
    import wandb
    wandb_available = True
    from pytorch_lightning.loggers import WandbLogger
except ImportError:
    wandb_available = False


if __name__ == "__main__":
    if cfg.data.debug:
        cfg.optimizer.max_epochs = 8
    multi_gpu = False
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        cfg.optimizer.batch_size = int(cfg.optimizer.batch_size / num_gpus)
        logger.info(f"Detected {num_gpus} GPUs, setting batch size to {cfg.optimizer.batch_size}")
        multi_gpu = True
  
    run_id = datetime.datetime.now().strftime("%y%m%d_%H%M%S") if not cfg.run_id else cfg.run_id
    mp.set_start_method('spawn', force=True)
    os.environ['WANDB_START_METHOD'] = 'thread'
    logger.info(f"Run ID: {run_id}, debug mode: {cfg.data.debug}, num_epochs: {cfg.optimizer.max_epochs}, multi_gpu: {multi_gpu}")
    cfg = prepare_folders_paths_cfg(run_id, cfg, make_folders=True)
    train_events, val_events, test_events, all_events, label_dict, class_weights, classifier_label_map, detector_label_map, single_label_map = preprocessing_pipeline(cfg)
    logger.info("Data loaded")
    
    logger.info(f"Class weights: {class_weights}")
    input_shape = (3, cfg.augment.random_crop_kwargs.timesteps)
    #scaler = Scaler(cfg).scaler
    data_dict = {"train" : train_events, "val" : val_events, "test" : test_events}
        
    transforms_by_sample = [RandomCropTransform(cfg)]
    transforms_by_set = setup_transforms(cfg)


    
    logger.info(f"Transforms by set: {transforms_by_set}")

    data_module = BeamModule(data_dict, label_dict, transforms_by_sample, transforms_by_set, cfg)
        
    logger.info("Datamodule created")
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
    
    detector_metrics_list = ["accuracy", "precision", "recall"]
    classifier_metrics_list = ["accuracy", "precision", "recall"]
    #detector_metrics_list = []
    #classifier_metrics_list = []
    
    model = get_model(input_shape,
                      detector_metrics_list,
                      classifier_metrics_list,
                      detector_label_map, 
                      classifier_label_map,
                      class_weights["detector"], 
                      class_weights["classifier"],
                      cfg)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    summary(model, input_size=input_shape)
    
    
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
    #callbacks.append(ModelCheckpoint(monitor = "val_detector_recall", mode = "min", save_top_k = 1, 
    #                                     dirpath = cfg.project_paths.output_folders.model_save_folder,
    #                                     save_weights_only= True,
    #                                     filename = model_name + "_{epoch}_{val_total_loss:.2f}"
    #                                     ))
    logger.warning("Early stopping needs to be changed before full training")
    #callbacks.append(EarlyStopping(monitor = "val_total_loss", mode = "min", patience = cfg.callbacks.early_stopping_patience))

        
    if not multi_gpu:
        trainer = Trainer(max_epochs = cfg.optimizer.max_epochs,
                        devices = -1 if torch.cuda.is_available() else 1,
                        precision=32,  # Mixed precision
                        callbacks = callbacks,
                        logger = wandb_logger)
    else:
        trainer = Trainer(max_epochs = cfg.optimizer.max_epochs,
                        devices = -1,
                        accelerator = "gpu",
                        strategy = "ddp",
                        num_nodes = 1,
                        precision=32,  # Mixed precision
                        callbacks = callbacks,
                        logger = wandb_logger)
    
    trainer.fit(model, data_module)
    
    logger.info("Training complete")
    logger.info("Performing analysis")
    analysis = Analysis(model, data_module.val_dataloader,label_dict, classifier_label_map, detector_label_map, val_events, cfg)
    analysis.analysis_package(5)
    
    

    
    
