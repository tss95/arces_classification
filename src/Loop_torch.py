import torch
import pytorch_lightning as pl
from global_config import cfg, model_cfg
from torch import nn
from torch.nn import functional as F
import numpy as np
import wandb
from typing import List, Dict, Any, Tuple, Union


class Loop(pl.LightningModule):
    def __init__(
        self, 
        input_shape,
        label_map_detector: Dict[str, int], 
        label_map_classifier: Dict[str, int],
        detector_class_weights: Dict[str, float], 
        classifier_class_weights: Dict[str, float]
    ):
        super().__init__()
        # Reverse mapping for label maps
        self.channels, self.timesteps = input_shape
        self.label_map_detector = {v: k for k, v in label_map_detector.items()}
        self.label_map_classifier = {v: k for k, v in label_map_classifier.items()}
        
        # Class weights as PyTorch tensors
        self.detector_class_weights = torch.tensor(
            [detector_class_weights[key] for key in sorted(detector_class_weights.keys())]
        )
        self.classifier_class_weights = torch.tensor(
            [classifier_class_weights[key] for key in sorted(classifier_class_weights.keys())]
        )

    def configure_optimizers(self):
        if cfg.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=cfg.optimizer.optimizer_kwargs.max_lr, 
                                         weight_decay = cfg.optimizer.optimizer_kwargs.weight_decay)
        elif cfg.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=cfg.optimizer.optimizer_kwargs.max_lr, 
                                        momentum=cfg.optimizer.optimizer_kwargs.momentum, 
                                        weight_decay = cfg.optimizer.optimizer_kwargs.weight_decay)
        if cfg.optimizer.scheduler.use_scheduler:
            if cfg.optimizer.optimizer_kwargs.warmup:
                T_0 = cfg.optimizer.scheduler.warmup_epochs
                total_epochs = cfg.optimizer.max_epochs
                T_mult = total_epochs / cfg.optimzer.scheduler.warmup_epochs
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                                                T_0 = T_0, 
                                                                                T_mult = T_mult,
                                                                                eta_min = cfg.optimizer.optimizer_kwargs.min_lr)
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                    T_max = cfg.optimizer.max_epochs, 
                                                                    eta_min = cfg.optimizer.optimizer_kwargs.min_lr)
            return [optimizer], [scheduler]
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        total_loss, loss_detector, loss_classifier, _ = self.calculate_loss(x, y, training=True)
            
        self.log(f'train_total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'train_detector_loss', loss_detector, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log(f'train_classifier_loss', loss_classifier, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        return self._evaluate_step(batch, batch_idx, prefix='val')

    
    def _evaluate_step(self, batch, batch_idx, prefix):
        x, y = batch
        total_loss, loss_detector, loss_classifier, y_pred = self.calculate_loss(x, y, training=False)
        y_pred_detector = torch.argmax(y_pred['detector'], dim=1)
        y_pred_classifier = torch.argmax(y_pred['classifier'], dim=1)
    
    # Assuming y also contains true labels for both detector and classifier in a similar dictionary format
    
        # Logging the losses
        self.log(f'{prefix}_total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{prefix}_detector_loss', loss_detector, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log(f'{prefix}_classifier_loss', loss_classifier, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        return {'y_true_detector': y['detector'], 'y_pred_detector': y_pred_detector,
                'y_true_classifier': y['classifier'], 'y_pred_classifier': y_pred_classifier}

    def test_step(self, batch, batch_idx):
        return self._evaluate_step(batch, batch_idx, prefix='test')

    # def configure_optimizers(self): define inside models.
        

    def training_epoch_end(self, outputs):
        pass

    # TODO: Adapt the output for callbacks so that the output is essentially in strings(?) to simplify metrics and downstream processes.
    # TODO: Also add the probabilities.
    def validation_epoch_end(self, validation_step_outputs):
        y_true_detector = torch.cat([x['y_true_detector'] for x in validation_step_outputs], dim=0)
        y_pred_detector = torch.cat([x['y_pred_detector'] for x in validation_step_outputs], dim=0)
        
        y_true_classifier = torch.cat([x['y_true_classifier'] for x in validation_step_outputs], dim=0)
        y_pred_classifier = torch.cat([x['y_pred_classifier'] for x in validation_step_outputs], dim=0)
        
        # Store these for later use in callbacks or further processing
        self.val_y_true_detector = y_true_detector
        self.val_y_pred_detector = y_pred_detector
        self.val_y_true_classifier = y_true_classifier
        self.val_y_pred_classifier = y_pred_classifier

    def test_epoch_end(self, test_step_outputs):
        y_true_detector = torch.cat([x['y_true_detector'] for x in test_step_outputs], dim=0)
        y_pred_detector = torch.cat([x['y_pred_detector'] for x in test_step_outputs], dim=0)
        
        y_true_classifier = torch.cat([x['y_true_classifier'] for x in test_step_outputs], dim=0)
        y_pred_classifier = torch.cat([x['y_pred_classifier'] for x in test_step_outputs], dim=0)
        
        # Store these for later use in callbacks or further processing
        self.test_y_true_detector = y_true_detector
        self.test_y_pred_detector = y_pred_detector
        self.test_y_true_classifier = y_true_classifier
        self.test_y_pred_classifier = y_pred_classifier
    

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

