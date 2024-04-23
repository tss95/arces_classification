import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
import numpy as np
import wandb
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, AUROC, AveragePrecision
from sklearn.metrics import matthews_corrcoef
from typing import List, Dict, Any, Tuple, Union

# Dictionary to map metric names to their respective torchmetrics classes
prob_metrics = {
    'auroc': AUROC(task='binary', average='macro'),
    'average_precision': AveragePrecision(task='binary', average='macro')
}

pred_metrics = {
    'accuracy': Accuracy(task='binary'),
    'precision': Precision(task='binary', average='macro'),
    'recall': Recall(task='binary', average='macro'),
    'f1': F1Score(task='binary', average='macro'),
    'matthews': matthews_corrcoef,
}

class Loop(pl.LightningModule):
    def __init__(
        self, 
        input_shape,
        detector_metrics_list: List[str],
        classifier_metrics_list: list[str],
        label_map_detector: Dict[str, int], 
        label_map_classifier: Dict[str, int],
        detector_class_weights: Dict[str, float], 
        classifier_class_weights: Dict[str, float],
        cfg
    ):
        super().__init__()
        self.cfg = cfg
               # Initialize MetricCollections for detector
        self.detector_prob_metrics = MetricCollection({name: prob_metrics[name] for name in detector_metrics_list if name in prob_metrics})
        self.detector_pred_metrics = MetricCollection({name: pred_metrics[name] for name in detector_metrics_list if name in pred_metrics})

        # Initialize MetricCollections for classifier
        self.classifier_prob_metrics = MetricCollection({name: prob_metrics[name] for name in classifier_metrics_list if name in prob_metrics})
        self.classifier_pred_metrics = MetricCollection({name: pred_metrics[name] for name in classifier_metrics_list if name in pred_metrics})
        # Reverse mapping for label maps
        self.channels, self.timesteps = input_shape
        #self.label_map_detector = {v: k for k, v in label_map_detector.items()}
        #self.label_map_classifier = {v: k for k, v in label_map_classifier.items()}
        self.label_map_detector = label_map_detector
        self.label_map_classifier = label_map_classifier
        # Initialize class weights as tensors and register them as buffers
        detector_weights_tensor = torch.tensor(
            [detector_class_weights[key] for key in sorted(detector_class_weights.keys())],
            dtype=torch.float
        )
        classifier_weights_tensor = torch.tensor(
            [classifier_class_weights[key] for key in sorted(classifier_class_weights.keys())],
            dtype=torch.float
        )
        
        self.register_buffer("detector_class_weights", detector_weights_tensor)
        self.register_buffer("classifier_class_weights", classifier_weights_tensor)
        self.setup()
        
    def setup(self, stage=None):
        torch.set_float32_matmul_precision('medium')
        
        


    def configure_optimizers(self):
        if self.cfg.optimizer.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.optimizer.optimizer_kwargs.max_lr, 
                                         weight_decay = self.cfg.optimizer.optimizer_kwargs.weight_decay)
        elif self.cfg.optimizer.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.cfg.optimizer.optimizer_kwargs.max_lr, 
                                        momentum=self.cfg.optimizer.optimizer_kwargs.momentum, 
                                        weight_decay = self.cfg.optimizer.optimizer_kwargs.weight_decay)
        if self.cfg.optimizer.scheduler.use_scheduler:
            if self.cfg.optimizer.optimizer_kwargs.warmup:
                T_0 = self.cfg.optimizer.scheduler.warmup_epochs
                total_epochs = self.cfg.optimizer.max_epochs
                T_mult = total_epochs / self.cfg.optimzer.scheduler.warmup_epochs
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                                                T_0 = T_0, 
                                                                                T_mult = T_mult,
                                                                                eta_min = self.cfg.optimizer.optimizer_kwargs.min_lr)
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                    T_max = self.cfg.optimizer.max_epochs, 
                                                                    eta_min = self.cfg.optimizer.optimizer_kwargs.min_lr)
            return [optimizer], [scheduler]
        return optimizer
    
    def training_step(self, batch, batch_idx):
        self.train()
        x, y, _ = batch
        y_pred = self(x)
        total_loss, loss_detector, loss_classifier = self.calculate_loss(y, y_pred)
        self.log('train_total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=self.cfg.optimizer.batch_size)
        self.log('train_detector_loss', loss_detector, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=self.cfg.optimizer.batch_size)
        self.log('train_classifier_loss', loss_classifier, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=self.cfg.optimizer.batch_size)

        # Save outputs for later aggregation
        if not hasattr(self, '_train_outputs'):
            self._train_outputs = []
        self._train_outputs.append({
            'y_true': y,  # Assuming y is a dictionary like {'detector': ..., 'classifier': ...}
            'y_pred': y_pred,
            'loss': total_loss
        })
        return total_loss
    
    def on_train_epoch_end(self):
        if hasattr(self, '_train_outputs') and self._train_outputs:
            # Aggregate and compute metrics using saved outputs
            y_true_aggregated, y_pred_aggregated = self.aggregate_outputs(self._train_outputs)
            self.compute_metrics(y_true_aggregated, y_pred_aggregated, stage='train')
            # Clear the saved outputs after processing
            del self._train_outputs

    def validation_step(self, batch, batch_idx):
        self.eval()
        outputs = self._evaluate_step(batch, batch_idx, prefix='val')
        if not hasattr(self, '_val_outputs'):
            self._val_outputs = []
        self._val_outputs.append(outputs)
        return outputs['losses']['total_loss']
    
    def _evaluate_step(self, batch, batch_idx, prefix: str):
        x, y, _ = batch
        # Perform the forward pass
        y_pred = self(x)
        total_loss, loss_detector, loss_classifier = self.calculate_loss(y, y_pred)
        
        # Prepare the outputs. This structure allows for easy aggregation in the epoch-end step.
        outputs = {
            'y_true': y,  # Assuming y is a dictionary like {'detector': ..., 'classifier': ...}
            'y_pred': y_pred,  # Similarly structured dictionary
            'losses': {  # Optional: Include if you plan to use these for epoch-level logging or analysis
                'total_loss': total_loss,
                'loss_detector': loss_detector,
                'loss_classifier': loss_classifier,
            }
        }
        
        # Optionally log losses here if you want them logged per-step, but as mentioned, aggregation is better
        if prefix == 'val':  # Example conditional logging based on prefix
            self.log(f'{prefix}_total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=self.cfg.optimizer.batch_size)
            self.log(f'{prefix}_detector_loss', loss_detector, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=self.cfg.optimizer.batch_size)
            self.log(f'{prefix}_classifier_loss', loss_classifier, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=self.cfg.optimizer.batch_size)
        
        return outputs

        
    def on_validation_epoch_end(self):
        # Make sure the attribute exists and has content
        if hasattr(self, '_val_outputs') and self._val_outputs:
            y_true_aggregated, y_pred_aggregated = self.aggregate_outputs(self._val_outputs)
            self.compute_metrics(y_true_aggregated, y_pred_aggregated, stage='val')
            # Clear the saved outputs after processing them
            del self._val_outputs

        
    
    def aggregate_outputs(self, outputs):
        y_true_aggregated = {'detector': [], 'classifier': []}
        y_pred_aggregated = {'detector': [], 'classifier': []}

        for output in outputs:
            y_pred = output['y_pred']  # This assumes `output` is a dict with a 'y_pred' key
            y_true = output['y_true']  # Similarly, assumes a 'y_true' key

            # Aggregate predictions and true labels
            y_true_aggregated['detector'].append(y_true['detector'])
            y_pred_aggregated['detector'].append(y_pred['detector'])
            y_true_aggregated['classifier'].append(y_true['classifier'])
            y_pred_aggregated['classifier'].append(y_pred['classifier'])

        # Process aggregated data for metrics computation
        for key in y_true_aggregated:
            y_true_aggregated[key] = torch.cat(y_true_aggregated[key], dim=0)
            y_pred_aggregated[key] = torch.cat(y_pred_aggregated[key], dim=0)
        return y_true_aggregated, y_pred_aggregated
    

    def test_step(self, batch, batch_idx):
        # Your _evaluate_step logic, simplified to focus on collecting outputs
        outputs = self._evaluate_step(batch, batch_idx, prefix='test')
        return outputs
    
    def on_test_epoch_end(self, test_step_outputs):
        y_true_aggregated, y_pred_aggregated = self.aggregate_outputs(test_step_outputs)
        self.compute_metrics(y_true_aggregated, y_pred_aggregated, stage='test')

        

    

    def calculate_loss(self, y: Dict[str, torch.Tensor], y_pred: Dict[str, torch.Tensor]):
        """
        Calculate the loss for the current batch.

        Args:
            y (Dict[str, torch.Tensor]): Input features.
            y_pred (Dict[str, torch.Tensor]): Ground truth labels.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing total loss, detector loss, classifier loss.
        """

        # Loss computation for detector with class weights
        instance_weights_detector = self.detector_class_weights[y['detector'].long()]
        loss_detector = torch.nn.functional.binary_cross_entropy_with_logits(y_pred['detector'], y['detector'], weight= instance_weights_detector)

        # Filtering for classifier loss computation
        mask_not_noise = y["detector"] != self.label_map_detector["noise"]
        y_true_classifier = y["classifier"][mask_not_noise]
        y_pred_classifier = y_pred["classifier"][mask_not_noise]

        # Loss computation for classifier with class weights
        instance_weights_classifier = self.classifier_class_weights[y_true_classifier.long()]
        if mask_not_noise.any():
            # Proceed with classifier loss computation
            loss_classifier = torch.nn.functional.binary_cross_entropy_with_logits(
                y_pred_classifier, y_true_classifier, weight=instance_weights_classifier)
        else:
            # Default classifier loss to 0 or another placeholder value when not applicable
            loss_classifier = 0.0
            # Convert to a tensor if necessary, matching the device and dtype of loss_detector
            loss_classifier = torch.tensor(loss_classifier).to(loss_detector.device, dtype=loss_detector.dtype)

        total_loss = loss_detector + loss_classifier
        
        # Check for NaN in detector loss
        if torch.isnan(loss_detector).any():
            print("NaN detected in loss_detector")
            # Log or print additional info as needed
            #print(f"y['detector']: {y['detector']}")
            #print(f"y_pred['detector']: {y_pred['detector']}")
            #print(f"Instance weights (detector): {instance_weights_detector}")
        # Check for NaN in classifier loss
        if torch.isnan(loss_classifier).any():
            print("NaN detected in loss_classifier")
            # Log or print additional info as needed
            #print(f"y['classifier']: {y['classifier']}")
            #print(f"y_pred['classifier']: {y_pred['classifier']}")
            #print(f"Instance weights (classifier): {instance_weights_classifier}")
        
        return total_loss, loss_detector, loss_classifier
    
    def compute_metrics(self, y_true, y_pred, stage: str):
        # Compute probabilities
        detector_prob = torch.sigmoid(y_pred['detector'])
        classifier_prob = torch.sigmoid(y_pred['classifier'])

        # Compute binary predictions
        detector_pred = (detector_prob > 0.5).int()
        classifier_pred = (classifier_prob > 0.5).int()

        # Detector Metrics: Update then Compute
        self.detector_prob_metrics.update(detector_prob, y_true['detector'])
        self.detector_pred_metrics.update(detector_pred, y_true['detector'])
        
        detector_prob_results = self.detector_prob_metrics.compute()
        detector_pred_results = self.detector_pred_metrics.compute()
        
        # Reset metrics for next computation
        self.detector_prob_metrics.reset()
        self.detector_pred_metrics.reset()

        # Log detector results
        for name, result in {**detector_prob_results, **detector_pred_results}.items():
            self.log(f'{stage}_detector_{name}', result, on_step=False, on_epoch=True, logger=True, sync_dist=True, batch_size=self.cfg.optimizer.batch_size)

        # Classifier Metrics
        mask_not_noise = detector_pred != self.label_map_detector["noise"]
        if mask_not_noise.any():
            classifier_true_filtered = y_true['classifier'][mask_not_noise]
            classifier_prob_filtered = classifier_prob[mask_not_noise]
            classifier_pred_filtered = classifier_pred[mask_not_noise]

            classifier_results = self.classifier_prob_metrics(classifier_prob_filtered, classifier_true_filtered)
            classifier_results.update(self.classifier_pred_metrics(classifier_pred_filtered, classifier_true_filtered))
            for name, result in classifier_results.items():
                self.log(f'{stage}_classifier_{name}', result, on_step=False, on_epoch=True, logger=True, sync_dist=True, batch_size=self.cfg.optimizer.batch_size)
            
            
            

