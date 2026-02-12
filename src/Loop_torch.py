import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
import numpy as np
import wandb
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, AUROC, AveragePrecision
from typing import List, Dict, Any, Tuple, Union, Optional
from types import SimpleNamespace


def _to_plain(obj):
    """Recursively convert namespaces/tensors to Python containers for checkpoint storage."""
    if isinstance(obj, SimpleNamespace):
        return {k: _to_plain(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_plain(v) for v in obj)
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    return obj

def _build_binary_prob_metric(name: str):
    if name == "auroc":
        return AUROC(task="binary", average="macro")
    if name == "average_precision":
        return AveragePrecision(task="binary", average="macro")
    return None


def _build_binary_pred_metric(name: str):
    if name == "accuracy":
        return Accuracy(task="binary")
    if name == "precision":
        return Precision(task="binary", average="macro")
    if name == "recall":
        return Recall(task="binary", average="macro")
    if name == "f1":
        return F1Score(task="binary", average="macro")
    return None


def _build_multiclass_prob_metric(name: str, num_classes: int):
    if name == "auroc":
        return AUROC(task="multiclass", num_classes=num_classes, average="macro")
    if name == "average_precision":
        return AveragePrecision(task="multiclass", num_classes=num_classes, average="macro")
    return None


def _build_multiclass_pred_metric(name: str, num_classes: int):
    if name == "accuracy":
        return Accuracy(task="multiclass", num_classes=num_classes, average="micro")
    if name == "precision":
        return Precision(task="multiclass", num_classes=num_classes, average="macro")
    if name == "recall":
        return Recall(task="multiclass", num_classes=num_classes, average="macro")
    if name == "f1":
        return F1Score(task="multiclass", num_classes=num_classes, average="macro")
    return None


def _build_metric_collection(metric_names: List[str], builder, *builder_args) -> MetricCollection:
    metrics = {}
    for name in metric_names:
        metric = builder(name, *builder_args) if builder_args else builder(name)
        if metric is not None:
            metrics[name] = metric
    return MetricCollection(metrics)

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
        cfg,
        head_mode: str = "dual",
        single_label_map: Optional[Dict[str, int]] = None,
        single_class_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.head_mode = str(head_mode).lower()
        if self.head_mode not in {"dual", "single"}:
            raise ValueError(f"Unsupported head_mode '{self.head_mode}'. Use 'dual' or 'single'.")
        # Reverse mapping for label maps
        self.channels, self.timesteps = input_shape
        self.label_map_detector = label_map_detector
        self.label_map_classifier = label_map_classifier
        self.single_label_map = single_label_map or {"noise": 0, "earthquake": 1, "explosion": 2}
        self.num_single_classes = len(self.single_label_map)
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
        default_single_weights = {
            label: 1.0 for label, _ in sorted(self.single_label_map.items(), key=lambda item: item[1])
        }
        if single_class_weights is not None:
            default_single_weights.update(single_class_weights)
        ordered_single_labels = [
            label for label, _ in sorted(self.single_label_map.items(), key=lambda item: item[1])
        ]
        single_weights_tensor = torch.tensor(
            [float(default_single_weights[label]) for label in ordered_single_labels],
            dtype=torch.float,
        )
        self.register_buffer("single_class_weights", single_weights_tensor)

        if self.head_mode == "dual":
            # Initialize MetricCollections for detector/classifier heads.
            self.detector_prob_metrics = _build_metric_collection(detector_metrics_list, _build_binary_prob_metric)
            self.detector_pred_metrics = _build_metric_collection(detector_metrics_list, _build_binary_pred_metric)
            self.classifier_prob_metrics = _build_metric_collection(classifier_metrics_list, _build_binary_prob_metric)
            self.classifier_pred_metrics = _build_metric_collection(classifier_metrics_list, _build_binary_pred_metric)
        else:
            single_metrics = detector_metrics_list if detector_metrics_list else classifier_metrics_list
            self.single_prob_metrics = _build_metric_collection(
                single_metrics, _build_multiclass_prob_metric, self.num_single_classes
            )
            self.single_pred_metrics = _build_metric_collection(
                single_metrics, _build_multiclass_pred_metric, self.num_single_classes
            )
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
        losses = self.calculate_loss(y, y_pred)
        total_loss = losses["total_loss"]
        self._log_losses("train", losses, on_step=True, on_epoch=True)

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
        return outputs
    
    def _evaluate_step(self, batch, batch_idx, prefix: str):
        x, y, _ = batch
        # Perform the forward pass
        y_pred = self(x)
        losses = self.calculate_loss(y, y_pred)
        
        # Prepare the outputs. This structure allows for easy aggregation in the epoch-end step.
        outputs = {
            'y_true': y,  # Assuming y is a dictionary like {'detector': ..., 'classifier': ...}
            'y_pred': y_pred,  # Similarly structured dictionary
            'losses': losses,
        }
        
        # Optionally log losses here if you want them logged per-step, but as mentioned, aggregation is better
        if prefix == 'val':  # Example conditional logging based on prefix
            self._log_losses(prefix, losses, on_step=False, on_epoch=True)
        
        return outputs

        
    def on_validation_epoch_end(self):
        # Make sure the attribute exists and has content
        if hasattr(self, '_val_outputs') and self._val_outputs:
            y_true_aggregated, y_pred_aggregated = self.aggregate_outputs(self._val_outputs)
            self.compute_metrics(y_true_aggregated, y_pred_aggregated, stage='val')
            # Clear the saved outputs after processing them
            del self._val_outputs

        
    
    def aggregate_outputs(self, outputs):
        if not outputs:
            return {}, {}
        pred_keys = list(outputs[0]["y_pred"].keys())
        y_true_aggregated = {key: [] for key in pred_keys}
        y_pred_aggregated = {key: [] for key in pred_keys}

        for output in outputs:
            y_pred = output['y_pred']  # This assumes `output` is a dict with a 'y_pred' key
            y_true = output['y_true']  # Similarly, assumes a 'y_true' key

            for key in pred_keys:
                y_true_aggregated[key].append(y_true[key])
                y_pred_aggregated[key].append(y_pred[key])

        # Process aggregated data for metrics computation
        for key in y_true_aggregated:
            y_true_aggregated[key] = torch.cat(y_true_aggregated[key], dim=0)
            y_pred_aggregated[key] = torch.cat(y_pred_aggregated[key], dim=0)
        return y_true_aggregated, y_pred_aggregated
    

    def test_step(self, batch, batch_idx):
        outputs = self._evaluate_step(batch, batch_idx, prefix='test')
        if not hasattr(self, "_test_outputs"):
            self._test_outputs = []
        self._test_outputs.append(outputs)
        return outputs
    
    def on_test_epoch_end(self):
        if hasattr(self, "_test_outputs") and self._test_outputs:
            y_true_aggregated, y_pred_aggregated = self.aggregate_outputs(self._test_outputs)
            self.compute_metrics(y_true_aggregated, y_pred_aggregated, stage='test')
            del self._test_outputs

        

    

    def _log_losses(self, prefix: str, losses: Dict[str, torch.Tensor], on_step: bool, on_epoch: bool) -> None:
        self.log(
            f"{prefix}_total_loss",
            losses["total_loss"],
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=self.cfg.optimizer.batch_size,
        )
        if self.head_mode == "dual":
            self.log(
                f"{prefix}_detector_loss",
                losses["loss_detector"],
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                batch_size=self.cfg.optimizer.batch_size,
            )
            self.log(
                f"{prefix}_classifier_loss",
                losses["loss_classifier"],
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                batch_size=self.cfg.optimizer.batch_size,
            )
        else:
            self.log(
                f"{prefix}_single_loss",
                losses["loss_single"],
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                batch_size=self.cfg.optimizer.batch_size,
            )

    def _log_metric_results(self, stage: str, prefix: str, results: Dict[str, torch.Tensor]) -> None:
        for name, result in results.items():
            self.log(
                f"{stage}_{prefix}_{name}",
                result,
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
                batch_size=self.cfg.optimizer.batch_size,
            )

    def _log_zero_metrics(self, stage: str, prefix: str, collection: MetricCollection) -> None:
        for name in collection.keys():
            self.log(
                f"{stage}_{prefix}_{name}",
                torch.tensor(0.0, device=self.device),
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
                batch_size=self.cfg.optimizer.batch_size,
            )

    def calculate_loss(self, y: Dict[str, torch.Tensor], y_pred: Dict[str, torch.Tensor]):
        """Calculate training/validation loss for the active head mode."""
        if self.head_mode == "single":
            y_true_single = y["single"].long().view(-1)
            logits_single = y_pred["single"]
            loss_single = F.cross_entropy(
                logits_single,
                y_true_single,
                weight=self.single_class_weights,
            )
            return {
                "total_loss": loss_single,
                "loss_single": loss_single,
            }

        detector_true = y["detector"].float()
        detector_logits = y_pred["detector"]
        detector_index = detector_true.long().view(-1)
        instance_weights_detector = self.detector_class_weights[detector_index].view_as(detector_true)
        loss_detector = F.binary_cross_entropy_with_logits(
            detector_logits,
            detector_true,
            weight=instance_weights_detector,
        )

        mask_not_noise = detector_index != self.label_map_detector["noise"]
        classifier_true_full = y["classifier"].float().view(-1)
        classifier_logits_full = y_pred["classifier"].view(-1)
        y_true_classifier = classifier_true_full[mask_not_noise]
        y_pred_classifier = classifier_logits_full[mask_not_noise]

        if y_true_classifier.numel() > 0:
            instance_weights_classifier = self.classifier_class_weights[y_true_classifier.long()]
            loss_classifier = F.binary_cross_entropy_with_logits(
                y_pred_classifier,
                y_true_classifier,
                weight=instance_weights_classifier,
            )
        else:
            loss_classifier = torch.tensor(0.0, device=loss_detector.device, dtype=loss_detector.dtype)

        total_loss = loss_detector + loss_classifier
        return {
            "total_loss": total_loss,
            "loss_detector": loss_detector,
            "loss_classifier": loss_classifier,
        }
    
    def compute_metrics(self, y_true, y_pred, stage: str):
        if self.head_mode == "single":
            single_logits = y_pred["single"]
            single_true = y_true["single"].long().view(-1)
            single_prob = torch.softmax(single_logits, dim=1)
            single_pred = torch.argmax(single_prob, dim=1)

            self.single_prob_metrics.update(single_prob, single_true)
            self.single_pred_metrics.update(single_pred, single_true)
            single_prob_results = self.single_prob_metrics.compute()
            single_pred_results = self.single_pred_metrics.compute()
            self.single_prob_metrics.reset()
            self.single_pred_metrics.reset()
            self._log_metric_results(stage, "single", {**single_prob_results, **single_pred_results})
            return

        detector_prob = torch.sigmoid(y_pred['detector'])
        classifier_prob = torch.sigmoid(y_pred['classifier'])

        detector_true = y_true["detector"].int()
        classifier_true = y_true["classifier"].int()
        detector_pred = (detector_prob > 0.5).int()
        classifier_pred = (classifier_prob > 0.5).int()

        self.detector_prob_metrics.update(detector_prob, detector_true)
        self.detector_pred_metrics.update(detector_pred, detector_true)
        detector_prob_results = self.detector_prob_metrics.compute()
        detector_pred_results = self.detector_pred_metrics.compute()
        self.detector_prob_metrics.reset()
        self.detector_pred_metrics.reset()
        self._log_metric_results(stage, "detector", {**detector_prob_results, **detector_pred_results})

        mask_not_noise = detector_pred.view(-1) != self.label_map_detector["noise"]
        classifier_true_filtered = classifier_true.view(-1)[mask_not_noise]
        classifier_prob_filtered = classifier_prob.view(-1)[mask_not_noise]
        classifier_pred_filtered = classifier_pred.view(-1)[mask_not_noise]

        if classifier_true_filtered.numel() > 0:
            classifier_prob_filtered = classifier_prob_filtered.unsqueeze(1)
            classifier_pred_filtered = classifier_pred_filtered.unsqueeze(1)
            classifier_true_filtered = classifier_true_filtered.unsqueeze(1)
            self.classifier_prob_metrics.update(classifier_prob_filtered, classifier_true_filtered)
            self.classifier_pred_metrics.update(classifier_pred_filtered, classifier_true_filtered)
            classifier_prob_results = self.classifier_prob_metrics.compute()
            classifier_pred_results = self.classifier_pred_metrics.compute()
            self.classifier_prob_metrics.reset()
            self.classifier_pred_metrics.reset()
            self._log_metric_results(stage, "classifier", {**classifier_prob_results, **classifier_pred_results})
        else:
            self._log_zero_metrics(stage, "classifier", self.classifier_prob_metrics)
            self._log_zero_metrics(stage, "classifier", self.classifier_pred_metrics)
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["cfg"] = _to_plain(self.cfg)
        if hasattr(self, "model_cfg"):
            checkpoint["model_cfg"] = _to_plain(self.model_cfg)
        if hasattr(self, "scaler_state"):
            checkpoint["scaler_state"] = self.scaler_state
            
