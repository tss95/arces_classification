import math
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from global_config import cfg, logger
import copy
import os
try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False
    

class Callback:
    def __init__(self, cfg):
        self.name = "callback"
        self.cfg = cfg
    
    def on_epoch_begin(self, epoch, logs = None):
        pass
    
    def on_epoch_end(self, epoch, logs = None):
        pass
            
    def on_train_end(self, epoch, logs = None):
        pass
    
    def is_improvement(self, logs, mode, target_metric, best_metric):
        if mode == "min":
            if logs[target_metric] < best_metric:
                return True
            else:
                return False
        elif mode == "max":
            if logs[target_metric] > best_metric:
                return True
            else:
                return False
        else:
            raise ValueError(f"Mode must be either 'min' or 'max', not {mode}")
    
        
        
        
class EarlyStoppingCallback(Callback):
    def __init__(self, cfg, patience, target_metric = "val_avg_loss", mode = "min"):
        super().__init__(cfg)
        self.name = "EarlyStoppingCallback"
        self.patience = patience
        self.target_metric = target_metric
        self.stop_training = False
        self.mode = mode
        self.counter = 0
        self.best_metric = float("inf") if mode == "min" else float("-inf")
        
    
    def on_epoch_end(self, epoch, logs):
        if logs is None:
            raise ValueError("Logs must be provided")
        if self.is_improvement(logs, self.mode, self.target_metric, self.best_metric):
            self.best_metric = logs[self.target_metric]
            self.counter = 0
            logger.info(f"Epoch {epoch}: {self.target_metric} improved to {self.best_metric}")
        else:
            self.counter += 1
        if self.counter > self.patience:
            logger.info(f"No improvement in {self.target_metric} for {self.patience} epochs, stopping early")
            self.stop_training = True
    


        
class ModelCheckpointCallback(Callback):
    def __init__(self, cfg, model, model_name, target_metric = "val_avg_loss", mode = "min"):
        super().__init__(cfg)
        self.name = "ModelCheckpointCallback"
        self.model = model
        self.model_name = model_name
        self.target_metric = target_metric
        self.mode = mode
        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self.best_weights = None
        

        
    def on_epoch_end(self, epoch, logs):
        if logs is None:
            raise ValueError("Logs must be provided")
        current_metric = logs[self.target_metric]
        if self.is_improvement(logs, self.mode, self.target_metric, self.best_metric):
            self.best_metric = current_metric
            self.best_weights = copy.deepcopy(self.model.state_dict())
            logger.info(f"Epoch {epoch}: {self.target_metric} improved to {self.best_metric}, temporary storing model state.")
            
    def on_train_end(self, epoch, logs):
        if self.best_weights is not None:
            model_path = os.path.join(self.cfg.output_paths.model_weights_folder, f"{self.model_name}_best_weights.pth")
            logger.info(f"Saving best model weights to {model_path}")
            torch.save(self.best_weights, model_path)
            self.model.load_state_dict(self.best_weights)
            logger.info("Loaded the best model weights after training.")
        else:
            logger.info("No improvements were made, no model weights to save.")
            
            
