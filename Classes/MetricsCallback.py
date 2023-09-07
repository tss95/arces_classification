from tensorflow.keras.callbacks import Callback
from global_config import logger, cfg, model_cfg
import numpy as np
import wandb
import tensorflow as tf


class MetricsCallback(Callback):
    def __init__(self, y_train, label_map):
        super().__init__()
        y_train = np.array(y_train)
        print(y_train)
        # Determine if it's a binary or multi-class classification
        self.is_binary = True if cfg.data.num_classes < 3 else False
        
        # For binary classification, no need to find the least frequent class
        if not self.is_binary:
            class_counts = np.sum(y_train, axis=0)
            self.min_class_id = np.argmin(class_counts)
            if label_map:
                self.label_name = label_map.get(self.min_class_id, str(self.min_class_id)).split(' ')[0]
            else:
                self.label_name = str(self.min_class_id)
        else:
            self.label_name = 'binary'
        
        self.precision = tf.keras.metrics.Precision(name=f"val_precision_{self.label_name}")
        self.recall = tf.keras.metrics.Recall(name=f"val_recall_{self.label_name}")
        self.accuracy = tf.keras.metrics.CategoricalAccuracy(name="val_accuracy") if not self.is_binary else tf.keras.metrics.BinaryAccuracy(name="val_accuracy")

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


