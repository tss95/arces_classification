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
from src.Transforms import RandomCropTransform, MinMaxPerChannelTransform
from src.Scaler_torch import Scaler
try:
    import wandb
    wandb_available = True
    from pytorch_lightning.loggers import WandbLogger
except ImportError:
    wandb_available = False


def batch_transforms_collate_fn(transforms=None):
    def batch_collate_fn(batch):
        # Initialize containers for batched components
        batched_data = [item[0] for item in batch]
        # Preparing a dictionary to hold stacked labels for each key
        batched_labels = {'detector': [], 'classifier': []}

        # Assuming IDs are strings, so we collect them into a list
        batched_ids = [item[2] for item in batch]
        
        # Process labels to organize them by key
        for _, labels, _ in batch:
            batched_labels['detector'].append(labels['detector'])
            batched_labels['classifier'].append(labels['classifier'])

        # Stack the data to form a single batch tensor
        batched_data = torch.stack(batched_data)

        # Stack labels for each key to form tensors
        batched_labels['detector'] = torch.stack(batched_labels['detector'])
        batched_labels['classifier'] = torch.stack(batched_labels['classifier'])

        # Apply each transform in the list to the batched data
        if transforms:
            for transform in transforms:
                batched_data = transform(batched_data)

        # Return the batched data, labels, and IDs in the expected structure
        return batched_data, batched_labels, batched_ids

    return batch_collate_fn


if __name__ == "__main__":
    run_id = datetime.datetime.now().strftime("%y%m%d_%H%M%S") if not cfg.run_id else cfg.run_id
    os.environ['WANDB_START_METHOD'] = 'thread'
    logger.info(f"Run ID: {run_id}, debug mode: {cfg.data.debug}")
    train_events, val_events, test_events, all_events, label_dict, class_weights, classifier_label_map, detector_label_map = preprocessing_pipeline(cfg)
    logger.info("Data loaded")
    input_shape = (3, cfg.augment.random_crop_kwargs.timesteps)
    #scaler = Scaler(cfg).scaler
    data_dict = {"train" : train_events, "val" : val_events, "test" : test_events}
        
    transforms_by_sample = [RandomCropTransform(cfg), MinMaxPerChannelTransform(cfg)]
    transforms_by_set = setup_transforms(cfg)
    datasets = {}
    datasets["train"] = BeamDataset(data_dict["train"], label_dict, transforms = transforms_by_sample)
    datasets["val"] = BeamDataset(data_dict["val"], label_dict, transforms = transforms_by_sample)
    if cfg.data.load_testset:
        datasets["test"] = BeamDataset(data_dict["test"], label_dict, transforms = transforms_by_sample)
    
    dataloaders = {}
    for key in datasets.keys():
        current_collate_fn = batch_transforms_collate_fn(transforms=transforms_by_set[key])
        dataloaders[key] = DataLoader(datasets[key], 
                                      batch_size=cfg.optimizer.batch_size, 
                                      shuffle=True if key == "train" else False, 
                                      num_workers = cfg.num_workers,
                                      collate_fn=current_collate_fn)
        
    logger.info("Dataloaders created")
    
    wandb_logger = None
    haikunator = Haikunator()
    model_name = f"{cfg.model_name}_{haikunator.haikunate()}"
    if wandb_available and cfg.wandb.active:
        config_dict = {}
        for key, value in vars(cfg).items():
            config_dict[key] = value
        wandb_logger = WandbLogger(name = model_name, project=cfg.wandb.project, entity= cfg.wandb.entity, config=config_dict)
    
    #detector_metrics_list = ["accuracy", "precision", "recall"]
    #classifier_metrics_list = ["accuracy", "precision", "recall"]
    detector_metrics_list = []
    classifier_metrics_list = []
    
    model = get_model(input_shape,
                      detector_metrics_list,
                      classifier_metrics_list,
                      detector_label_map, 
                      classifier_label_map,
                      class_weights["detector"], 
                      class_weights["classifier"],
                      cfg)
    callbacks = []
    callbacks.append(ModelCheckpoint(monitor = "val_total_loss", mode = "min", save_top_k = 1, 
                                         dirpath = cfg.project_paths.output_folders.model_save_folder,
                                         save_weights_only= True,
                                         filename = model_name + "_{epoch}_{val_total_loss:.2f}"
                                         ))
    callbacks.append(EarlyStopping(monitor = "val_total_loss", mode = "min", patience = cfg.callbacks.early_stopping_patience))
    
    trainer = Trainer(max_epochs = cfg.optimizer.max_epochs,
                      devices = -1 if torch.cuda.is_available() else 1,
                      strategy='ddp',  # Distributed Data Parallel
                      precision=16,  # Mixed precision
                      callbacks = callbacks,
                      logger = wandb_logger)
    
    trainer.fit(model, dataloaders["train"], dataloaders["val"])
    
    
    # Trainer setup
    
    

    
    
