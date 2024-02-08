import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
import numpy as np
import wandb
from typing import List, Dict, Any, Tuple, Union


class Loop(pl.LightningModule):
    def __init__(
        self, 
        label_map_detector: Dict[str, int], 
        label_map_classifier: Dict[str, int],
        detector_metrics: List[str], 
        classifier_metrics: List[str], 
        detector_class_weights: Dict[str, float], 
        classifier_class_weights: Dict[str, float]
    ):
        super().__init__()
        # Reverse mapping for label maps
        self.label_map_detector = {v: k for k, v in label_map_detector.items()}
        self.label_map_classifier = {v: k for k, v in label_map_classifier.items()}
        
        # Class weights as PyTorch tensors
        self.detector_class_weights = torch.tensor(
            [detector_class_weights[key] for key in sorted(detector_class_weights.keys())]
        )
        self.classifier_class_weights = torch.tensor(
            [classifier_class_weights[key] for key in sorted(classifier_class_weights.keys())]
        )

        
    def training_step(self, batch, batch_idx):
        x, y = batch
        total_loss, loss_detector, loss_classifier, y_pred = self.calculate_loss(x, y)
            
        self.log(f'train_total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'train_detector_loss', loss_detector, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log(f'train_classifier_loss', loss_classifier, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        return self._evaluate_step(batch, batch_idx, prefix='val')

    
    def _evaluate_step(self, batch, batch_idx, prefix):
        x, y = batch
        total_loss, loss_detector, loss_classifier, y_pred = self.calculate_loss(x, y)
        
        # Logging the losses
        self.log(f'{prefix}_total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{prefix}_detector_loss', loss_detector, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log(f'{prefix}_classifier_loss', loss_classifier, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        return total_loss

    def test_step(self, batch, batch_idx):
        return self._evaluate_step(batch, batch_idx, prefix='test')

    # def configure_optimizers(self): define inside models.
        

    def training_epoch_end(self, outputs):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def test_epoch_end(self, outputs):
        pass
    

    def calculate_loss(self, x: torch.Tensor, y: Dict[str, torch.Tensor], training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate the loss for the current batch.

        Args:
            x (torch.Tensor): Input features.
            y (Dict[str, torch.Tensor]): Ground truth labels.
            training (bool, optional): Flag indicating whether the model is in training mode. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing total loss, detector loss, classifier loss, and predicted labels.
        """
        y_pred = self(x, training=training)

        # Loss computation for detector with class weights
        loss_detector = torch.nn.functional.binary_cross_entropy_with_logits(y_pred['detector'], y['detector'])
        detector_weights = self.detector_class_weights[y['detector'].long()]
        loss_detector *= detector_weights

        # Filtering for classifier loss computation
        mask_not_noise = y["detector"] != self.label_map_detector["noise"]
        y_true_classifier = y["classifier"][mask_not_noise]
        y_pred_classifier = y_pred["classifier"][mask_not_noise]

        # Loss computation for classifier with class weights
        loss_classifier = torch.nn.functional.binary_cross_entropy_with_logits(y_pred_classifier, y_true_classifier)
        classifier_weights = self.classifier_class_weights[y_true_classifier.long()]
        loss_classifier *= classifier_weights

        # Total loss calculation
        loss_detector = loss_detector.mean()
        loss_classifier = loss_classifier.mean()
        total_loss = loss_detector + loss_classifier
        
        return total_loss, loss_detector, loss_classifier, y_pred

