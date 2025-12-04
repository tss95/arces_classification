import math
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from global_config import cfg, logger
import copy
import os
from pytorch_lightning import Trainer, LightningModule
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback
import wandb
try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False
    

class CustomCallback:
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
    
        


class ConfusionMatrixLogger(Callback):
    def __init__(self, cfg, n_epochs = 1):
        super().__init__()
        self.cfg = cfg
        self.n_epochs = n_epochs
        self._val_preds = []
        self._val_trues = []

    def process_predictions(self, y_pred):
        final_preds = []
        detector_preds = y_pred['detector'].sigmoid().detach().cpu()
        classifier_preds = y_pred['classifier'].sigmoid().detach().cpu()

        for detector_pred, classifier_pred in zip(detector_preds, classifier_preds):
            if detector_pred.item() >= self.cfg.data.model_threshold:
                final_pred = 'Earthquake' if classifier_pred.item() < 0.5 else 'Explosion'
            else:
                final_pred = 'Noise'
            final_preds.append(final_pred)
        
        return final_preds

    def plot_confusion_matrix(self, y_true, y_pred):
        unique_labels = ['Noise', 'Earthquake', 'Explosion']
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
        fig, ax = plt.subplots(figsize=(10, 7))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.title('Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return fig
            
    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx, dataloader_idx: int = 0):
        # Only collect outputs on epochs where we plan to log
        if (trainer.current_epoch + 1) % self.n_epochs != 0:
            return
        if outputs is None or 'y_pred' not in outputs or 'y_true' not in outputs:
            return
        preds = self.process_predictions(outputs['y_pred'])
        trues = self.translate_true_labels(outputs['y_true'])
        self._val_preds.extend(preds)
        self._val_trues.extend(trues)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if (trainer.current_epoch + 1) % self.n_epochs != 0:
            self._val_preds.clear()
            self._val_trues.clear()
            return
        if not self._val_preds or not self._val_trues:
            return

        fig = self.plot_confusion_matrix(self._val_trues, self._val_preds)

        if wandb_available:
            wandb.log({"confusion_matrix": wandb.Image(fig)}, commit=False)
        plt.close(fig)

        self._val_preds.clear()
        self._val_trues.clear()
            
    def translate_true_labels(self, true_labels):
        string_labels = []
        detector_trues = true_labels['detector'].detach().cpu()
        classifier_trues = true_labels['classifier'].detach().cpu()
        
        for detector_true, classifier_true in zip(detector_trues, classifier_trues):
            # Use .item() to convert each element (which is a scalar tensor) to a Python scalar
            if detector_true.item() == 0:
                string_labels.append('Noise')
            else:
                if classifier_true.item() == 0:
                    string_labels.append('Earthquake')
                else:
                    string_labels.append('Explosion')
        return string_labels
    
class EarlyStoppingCallback(CustomCallback):
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
    


        
class ModelCheckpointCallback(CustomCallback):
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
            

