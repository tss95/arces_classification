from tensorflow.keras.callbacks import Callback
from global_config import logger, cfg, model_cfg
import numpy as np
import wandb
import tensorflow as tf


class MetricsCallback(Callback):
    def __init__(self, y_train, label_map):
        super().__init__()
        y_train = np.array(y_train)

        # For binary classification, no need to find the least frequent class

        self.label_name = 'binary'
        
        self.precision = tf.keras.metrics.Precision(name=f"val_precision_{self.label_name}")
        self.recall = tf.keras.metrics.Recall(name=f"val_recall_{self.label_name}")
        self.accuracy = tf.keras.metrics.BinaryAccuracy(name="val_accuracy")

    def on_epoch_end(self, epoch, logs=None):
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

    def on_train_batch_end(self, batch, logs=None):
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


