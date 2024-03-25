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
try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False


def batch_transforms_collate_fn(transforms):
    def batch_collate_fn(batch):
        samples, labels, event_ids = zip(*batch)
        samples = torch.stack(samples, dim=0)
        
        if transforms:
            samples = transforms(samples)
        
        labels = torch.tensor(labels)
        
        return samples, labels, event_ids
    return batch_collate_fn


if __name__ == "__main__":
    run_id = datetime.datetime.now().strftime("%y%m%d_%H%M%S") if not cfg.run_id else cfg.run_id
    os.environ['WANDB_START_METHOD'] = 'thread'
    logger.info(f"Run ID: {run_id}, cfg: {cfg}, model_cfg: {model_cfg}")
    train_events, val_events, test_events, all_events, label_dict, class_weights, train_dict, val_dict, test_dict, classifier_label_map, detector_label_map = preprocessing_pipeline(cfg)
    input_shape = (3, cfg.augment.random_crop_kwargs.timesteps)
    transforms_by_set = setup_transforms(cfg)
    datasets = {}
    datasets["train"] = BeamDataset(train_events, label_dict, train_dict)
    datasets["val"] = BeamDataset(val_events, label_dict, val_dict)
    datasets["test"] = BeamDataset(test_events, label_dict, test_dict)
    
    dataloaders = {}
    for key in datasets.keys():
        dataloaders[key] = DataLoader(datasets[key], 
                                      batch_size=cfg.optimizer.batch_size, 
                                      shuffle=True, 
                                      num_workers = cfg.num_workers,
                                      collate_fn=batch_transforms_collate_fn(transforms_by_set[key]))
        
    logger.info("Dataloaders created")
    
    haikunator = Haikunator()
    model_name = f"{cfg.model_name}_{haikunator.haikunate()}"
    if wandb_available and cfg.wandb.active:
        config_dict = {}
        for key, value in vars(cfg).items():
            config_dict[key] = value
        wandb.init(name = model_name, project=cfg.wandb.project, entity= cfg.wandb.entity, config=config_dict)
    
    detector_metrics_list = ["accuracy", "precision", "recall", "f1", "auroc", "average_precision"]
    classifier_metrics_list = ["accuracy", "precision", "recall", "f1", "auroc", "average_precision"]
    
    
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
                      devices = 2 if torch.cuda.is_available() else 0,
                      strategy='ddp',  # Distributed Data Parallel
                      precision=16,  # Mixed precision
                      callbacks = callbacks)
    
    trainer.fit(model, dataloaders["train"], dataloaders["val"])
    
    
    # Trainer setup
    
    

    
    
