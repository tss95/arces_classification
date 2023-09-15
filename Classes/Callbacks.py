import numpy as np
import tensorflow as tf
import wandb
from sklearn.metrics import confusion_matrix
import wandb
from Classes.Utils import get_y_and_ypred, plot_confusion_matrix  # Assuming this is in Classes.Utils

class ValidationConfusionMatrixCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_gen, label_maps):
        super().__init__()
        self.val_gen = val_gen
        self.label_maps = label_maps

    def on_epoch_end(self, epoch, logs=None):
        # Get true labels and predicted labels
        y_true, y_pred, _ = get_y_and_ypred(self.model, self.val_gen, self.label_maps)
        
        # Compute the confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred, labels=["noise", "earthquake", "explosion"])
        
        # Normalize the confusion matrix
        conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        
        # Convert to integer labels
        label_to_int = {"noise": 0, "earthquake": 1, "explosion": 2}
        y_true_int = [label_to_int[label] for label in y_true]
        y_pred_int = [label_to_int[label] for label in y_pred]
        
        plt = plot_confusion_matrix(conf_matrix, conf_matrix_normalized, ["noise", "earthquake", "explosion"])
        wandb.log({"confusion_matrix": plt})


class InPlaceProgressCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        self.steps_per_epoch = self.params['steps']
        
    def on_epoch_begin(self, epoch, logs=None):
        self.current_step = 0
        self.cumulative_train_loss = 0.0  # Initialize the cumulative train loss for the epoch
        print(f"\n Epoch {epoch + 1}/{self.epochs}")

    def on_batch_end(self, batch, logs=None):
        self.current_step += 1
        self.cumulative_train_loss += logs['train_total_loss']  # Accumulate the batch train loss
        
        avg_train_loss = self.cumulative_train_loss / self.current_step  # Compute the average train loss so far
        
        progbar = "=" * (self.current_step * 50 // self.steps_per_epoch)
        progbar += "-" * (50 - len(progbar))
        
        metrics_str = f" avg_train_total_loss: {avg_train_loss:.4f}"
        
        print(f"\r[{progbar}] {self.current_step}/{self.steps_per_epoch}{metrics_str}", end="")
        
    def on_epoch_end(self, epoch, logs=None):
        print(f"\n Epoch {epoch + 1}/{self.epochs} completed")


class WandbLoggingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            train_metrics = {k: logs[k] for k in logs if not k.startswith('val_')}
            val_metrics = {k: logs[k] for k in logs if k.startswith('val_')}
            wandb.log({"epoch": epoch, **train_metrics, **val_metrics})

# Usage example with model.fit()
# model.fit(x, y, epochs=10, verbose=0, callbacks=[InPlaceProgressCallback()])
