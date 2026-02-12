import torch
from global_config import cfg, logger
import os
import time
import json
import pickle
import random
import logging
from collections import defaultdict
from typing import Any, Dict, List
from pytorch_lightning import Trainer, LightningModule
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback
import h5py
import wandb
import numpy as np
from src.Live import LiveClassifier
from src.Scaler_torch import Scaler
try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False
    
class ConfusionMatrixLogger(Callback):
    def __init__(self, cfg, n_epochs = 1):
        super().__init__()
        self.cfg = cfg
        self.n_epochs = n_epochs
        self._val_preds = []
        self._val_trues = []

    def process_predictions(self, y_pred):
        final_preds = []
        if "single" in y_pred:
            single_probs = torch.softmax(y_pred["single"], dim=1).detach().cpu()
            idx_to_label = {0: "Noise", 1: "Earthquake", 2: "Explosion"}
            for class_idx in torch.argmax(single_probs, dim=1).tolist():
                final_preds.append(idx_to_label[int(class_idx)])
        else:
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
        report = classification_report(
            self._val_trues,
            self._val_preds,
            labels=["Noise", "Earthquake", "Explosion"],
            output_dict=True,
            zero_division=0,
        )
        single_metrics = {
            "val_single_final_accuracy": float(report.get("accuracy", 0.0)),
            "val_single_final_macro_f1": float(report["macro avg"]["f1-score"]),
            "val_single_final_weighted_f1": float(report["weighted avg"]["f1-score"]),
            "val_single_final_noise_f1": float(report["Noise"]["f1-score"]),
            "val_single_final_earthquake_f1": float(report["Earthquake"]["f1-score"]),
            "val_single_final_explosion_f1": float(report["Explosion"]["f1-score"]),
        }
        for key, value in single_metrics.items():
            pl_module.log(
                key,
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=key in {"val_single_final_accuracy", "val_single_final_macro_f1"},
                logger=True,
                sync_dist=False,
            )
        logger.info(
            "Single-window final-label validation epoch %d: acc=%.4f macro_f1=%.4f",
            trainer.current_epoch + 1,
            single_metrics["val_single_final_accuracy"],
            single_metrics["val_single_final_macro_f1"],
        )

        if wandb_available and getattr(wandb, "run", None) is not None:
            wandb.log({"confusion_matrix": wandb.Image(fig)}, commit=False)
        plt.close(fig)

        self._val_preds.clear()
        self._val_trues.clear()
            
    def translate_true_labels(self, true_labels):
        string_labels = []
        if "single" in true_labels:
            idx_to_label = {0: "Noise", 1: "Earthquake", 2: "Explosion"}
            for class_idx in true_labels["single"].detach().cpu().view(-1).tolist():
                string_labels.append(idx_to_label[int(class_idx)])
        else:
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


class LiveStyleValidationCallback(Callback):
    """Run live-style ensemble validation on a fixed balanced subset of val data."""

    def __init__(self, cfg, scaler_state: Dict[str, Any], n_epochs: int = 1, per_class: int = 300, max_events: int = 900):
        super().__init__()
        self.cfg = cfg
        self.scaler_state = scaler_state
        self.n_epochs = n_epochs
        self.per_class = per_class
        self.max_events = max_events
        self.labels_order = ["noise", "earthquake", "explosion"]
        self.selected_indices: List[int] = []
        self.selected_label_distribution: Dict[str, int] = {}
        self.val_index_list = None
        self.val_index_path = None
        self.val_h5_path = None
        self.enabled = True
        self._initialized = False

    def _normalize_label(self, pred):
        if isinstance(pred, np.ndarray):
            if pred.size == 1:
                return str(pred.item())
            return str(pred.tolist())
        if isinstance(pred, (list, tuple)):
            if len(pred) == 1:
                return str(pred[0])
            return str(list(pred))
        return str(pred)

    def _select_balanced_subset(self):
        by_label = defaultdict(list)
        for idx, rec in enumerate(self.val_index_list):
            by_label[str(rec[3])].append(idx)

        rng = random.Random(int(self.cfg.seed))
        selected: List[int] = []
        for label in sorted(by_label.keys()):
            candidates = by_label[label][:]
            rng.shuffle(candidates)
            selected.extend(candidates[: self.per_class])

        rng.shuffle(selected)
        if self.max_events > 0:
            selected = selected[: self.max_events]
        self.selected_indices = selected

        chosen_dist: Dict[str, int] = defaultdict(int)
        for idx in self.selected_indices:
            chosen_dist[str(self.val_index_list[idx][3])] += 1
        self.selected_label_distribution = dict(chosen_dist)

    def _initialize(self):
        split = "debug" if self.cfg.data.debug else "full"
        val_index_path = os.path.join(self.cfg.data_paths.loaded_path, f"val_{split}_index_list.pkl")
        self.val_h5_path = os.path.join(self.cfg.data_paths.loaded_path, f"val_{split}_data.h5")
        self.val_index_path = val_index_path

        if not os.path.exists(val_index_path):
            logger.warning("Live-style validation disabled: missing %s", val_index_path)
            self.enabled = False
            return
        if not os.path.exists(self.val_h5_path):
            logger.warning("Live-style validation disabled: missing %s", self.val_h5_path)
            self.enabled = False
            return

        with open(val_index_path, "rb") as handle:
            self.val_index_list = pickle.load(handle)
        self._select_balanced_subset()
        if not self.selected_indices:
            logger.warning("Live-style validation disabled: no selected validation indices.")
            self.enabled = False
            return

        self._initialized = True
        logger.info(
            "Live-style validation enabled with %d samples (%s).",
            len(self.selected_indices),
            self.selected_label_distribution,
        )

    def _write_epoch_artifact(self, trainer: Trainer, payload: Dict[str, Any]):
        out_dir = os.path.join(self.cfg.project_paths.output_folder, "live_style_validation")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"epoch_{trainer.current_epoch + 1:03d}.json")
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def _plot_confusion_matrix(self, cm: np.ndarray) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.labels_order)
        disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
        ax.set_title("Live-Style Validation Confusion Matrix")
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        return fig

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):
        if not trainer.is_global_zero:
            return
        if not self._initialized:
            self._initialize()

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if not trainer.is_global_zero:
            return
        if not self.enabled:
            return
        if getattr(trainer, "sanity_checking", False):
            return
        if (trainer.current_epoch + 1) % self.n_epochs != 0:
            return
        if not self._initialized:
            self._initialize()
            if not self.enabled:
                return

        scaler = Scaler(self.cfg)
        scaler.load_state_dict(self.scaler_state)
        label_maps = {
            "detector": {0: "noise", 1: "event"},
            "classifier": {0: "earthquake", 1: "explosion"},
            "single": {0: "noise", 1: "earthquake", 2: "explosion"},
        }
        live_model = LiveClassifier(pl_module, scaler, label_maps, self.cfg)

        y_true: List[str] = []
        y_pred: List[str] = []
        trace_lengths: List[int] = []

        start = time.time()
        old_level = logger.level
        logger.setLevel(logging.WARNING)
        try:
            with torch.no_grad():
                pl_module.eval()
                with h5py.File(self.val_h5_path, "r") as handle:
                    data = handle["data"]
                    for idx in self.selected_indices:
                        trace = data[idx]
                        rec = self.val_index_list[idx]
                        truth = str(rec[3])
                        trace = live_model.extract_production_like_trace(
                            trace,
                            label=truth,
                            start_idx=rec[4],
                            end_idx=rec[5],
                        )
                        trace_lengths.append(int(trace.shape[1]))
                        pred, _, _, _, _ = live_model.predict(trace)
                        y_true.append(truth)
                        y_pred.append(self._normalize_label(pred))
        finally:
            logger.setLevel(old_level)
        elapsed = time.time() - start

        report = classification_report(
            y_true,
            y_pred,
            labels=self.labels_order,
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(y_true, y_pred, labels=self.labels_order)

        live_metrics = {
            "val_live_accuracy": float(report.get("accuracy", 0.0)),
            "val_live_macro_f1": float(report["macro avg"]["f1-score"]),
            "val_live_weighted_f1": float(report["weighted avg"]["f1-score"]),
            "val_live_noise_f1": float(report["noise"]["f1-score"]),
            "val_live_earthquake_f1": float(report["earthquake"]["f1-score"]),
            "val_live_explosion_f1": float(report["explosion"]["f1-score"]),
        }

        for key, value in live_metrics.items():
            pl_module.log(
                key,
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=key in {"val_live_accuracy", "val_live_macro_f1"},
                logger=True,
                sync_dist=False,
            )

        payload = {
            "epoch": int(trainer.current_epoch + 1),
            "trace_mode": "event_cropped_v1",
            "sample_size": len(self.selected_indices),
            "elapsed_seconds": elapsed,
            "events_per_second": len(self.selected_indices) / elapsed if elapsed > 0 else 0.0,
            "selected_label_distribution": self.selected_label_distribution,
            "trace_length_stats_samples": {
                "mean": float(np.mean(trace_lengths)),
                "median": float(np.median(trace_lengths)),
                "min": int(np.min(trace_lengths)),
                "max": int(np.max(trace_lengths)),
            },
            "trace_length_stats_seconds": {
                "mean": float(np.mean(trace_lengths) / self.cfg.data.sample_rate),
                "median": float(np.median(trace_lengths) / self.cfg.data.sample_rate),
                "min": float(np.min(trace_lengths) / self.cfg.data.sample_rate),
                "max": float(np.max(trace_lengths) / self.cfg.data.sample_rate),
            },
            "selected_indices": list(self.selected_indices),
            "val_data_file": os.path.basename(self.val_h5_path) if self.val_h5_path else None,
            "val_index_file": os.path.basename(self.val_index_path) if self.val_index_path else None,
            "labels_order": self.labels_order,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "metrics": live_metrics,
        }
        self._write_epoch_artifact(trainer, payload)

        if wandb_available and getattr(wandb, "run", None) is not None:
            fig = self._plot_confusion_matrix(cm)
            try:
                wandb.log({"val_live_confusion_matrix": wandb.Image(fig)}, commit=False)
            finally:
                plt.close(fig)

        logger.info(
            "Live-style validation epoch %d: acc=%.4f macro_f1=%.4f (%d events, %.2fs).",
            trainer.current_epoch + 1,
            live_metrics["val_live_accuracy"],
            live_metrics["val_live_macro_f1"],
            len(self.selected_indices),
            elapsed,
        )
