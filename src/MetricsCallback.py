from tensorflow.keras.callbacks import Callback
from typing import Dict, Any
from global_config import logger, cfg, model_cfg
from src.Utils import get_y_and_ypred
import numpy as np
import wandb
import tensorflow as tf


class MetricsCallback(Callback):
     """
    A custom callback for logging various metrics during training and validation.

    This callback computes and logs precision, recall, F1 score, and accuracy for both training and validation phases.
    It supports both binary and multi-class classification tasks.

    Attributes:
        val_gen (Sequence): The validation data generator.
        label_map (Dict[str, int]): A dictionary mapping label names to class indices.
        label_name (str): The name of the label for logging purposes, default is 'binary'.
        precision (tf.keras.metrics.Precision): Precision metric.
        recall (tf.keras.metrics.Recall): Recall metric.
        accuracy (tf.keras.metrics.BinaryAccuracy): Accuracy metric.
        is_binary (bool): Flag to determine if the task is binary classification.
        min_class_id (int): The class id of the least frequent class.
    """
    #TODO Rewrite this class to account for the output shape
    def __init__(self, val_gen, label_map: Dict[str, int]):
        """
        Initializes the MetricsCallback instance.

        Args:
            val_gen (Sequence): The validation data generator.
            label_map (Dict[str, int]): A dictionary mapping label names to class indices.
        """
        super().__init__()
        y_train = {"detector": np.array(y_train["detector"]),
                   "classifier": np.array(y_train["classifier"])}

        # For binary classification, no need to find the least frequent class

        self.label_name = 'binary'
        
        self.precision = tf.keras.metrics.Precision(name=f"val_precision_{self.label_name}")
        self.recall = tf.keras.metrics.Recall(name=f"val_recall_{self.label_name}")
        self.accuracy = tf.keras.metrics.BinaryAccuracy(name="val_accuracy")

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        """
        Called at the end of an epoch during training.

        Computes and logs precision, recall, F1 score, and accuracy for the validation set.

        Args:
            epoch (int): The index of the epoch.
            logs (Dict[str, Any]): The dictionary of logs from Keras.
        """
        x_val, y_val = self.validation_data[0], self.validation_data[1]
        y_pred = self.model.predict(x_val)


        if not self.is_binary:
            y_true_least = y_val[:, self.min_class_id]
            y_pred_least = np.argmax(y_pred, axis=-1)
            y_pred_least = (y_pred_least == self.min_class_id).astype(np.float32)
        else:
            y_true_least = y_val
            y_pred_least = (y_pred > 0.5).astype(np.float32)
        
        self.precision.update_state(y_true_least, y_pred_least)
        self.recall.update_state(y_true_least, y_pred_least)
        self.accuracy.update_state(y_val, y_pred)

        precision_result = self.precision.result().numpy()
        recall_result = self.recall.result().numpy()
        accuracy_result = self.accuracy.result().numpy()
        f1_score_result = 2 * ((precision_result * recall_result) / (precision_result + recall_result + 1e-5))

        # Log metrics to wandb
        wandb_metrics = {
            f'val_precision_{self.label_name}': precision_result,
            f'val_recall_{self.label_name}': recall_result,
            f'val_f1_score_{self.label_name}': f1_score_result,
            'val_accuracy': accuracy_result
        }
        wandb.log(wandb_metrics)

        # Log metrics to Keras logs as well
        logs[f'val_precision_{self.label_name}'] = precision_result
        logs[f'val_recall_{self.label_name}'] = recall_result
        logs[f'val_f1_score_{self.label_name}'] = f1_score_result
        logs['val_accuracy'] = accuracy_result

        self.precision.reset_states()
        self.recall.reset_states()
        self.accuracy.reset_states()

    def on_train_batch_end(self, batch: int, logs: Dict[str, Any] = None):
        """
        Called at the end of a training batch.

        Optionally logs metrics to wandb if the batch number meets the specified interval.

        Args:
            batch (int): The index of the batch.
            logs (Dict[str, Any]): The dictionary of logs from Keras.
        """
        if batch % cfg.callbacks.wandb_n_batches_per_update == 0:
            x_train, y_train = self.model.train_function.inputs[0], self.model.train_function.targets[0]
            y_pred = self.model.predict(x_train)
            
            if not self.is_binary:
                y_true_least = tf.gather(y_train, [self.min_class_id], axis=1)
                y_pred_least = tf.argmax(y_pred, axis=-1)
                y_pred_least = tf.cast(tf.equal(y_pred_least, self.min_class_id), dtype=tf.float32)
            else:
                y_true_least = y_train
                y_pred_least = tf.cast(y_pred > 0.5, dtype=tf.float32)
            
            # Update states
            self.precision.update_state(y_true_least, y_pred_least)
            self.recall.update_state(y_true_least, y_pred_least)
            self.accuracy.update_state(y_train, y_pred)
            
            # Compute metrics
            precision_result = self.precision.result().numpy()
            recall_result = self.recall.result().numpy()
            accuracy_result = self.accuracy.result().numpy()
            f1_score_result = 2 * ((precision_result * recall_result) / (precision_result + recall_result + 1e-5))

            # Log metrics to wandb
            wandb_metrics = {
                f'train_precision_{self.label_name}': precision_result,
                f'train_recall_{self.label_name}': recall_result,
                f'train_f1_score_{self.label_name}': f1_score_result,
                'train_accuracy': accuracy_result
            }
            wandb.log(wandb_metrics, step=batch)
            
            # Reset metrics
            self.precision.reset_states()
            self.recall.reset_states()
            self.accuracy.reset_states()


