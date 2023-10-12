import numpy as np
import tensorflow as tf
import wandb
from global_config import logger, cfg, model_cfg
from sklearn.metrics import confusion_matrix
import wandb
from PIL import Image
import io
import matplotlib.pyplot as plt
from collections import Counter
from Classes.Utils import get_y_and_ypred, plot_confusion_matrix  # Assuming this is in Classes.Utils

class ValidationConfusionMatrixCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_gen, label_maps, unswapped_labels):
        super().__init__()
        self.val_gen = val_gen
        self.label_maps = label_maps
        self.unswapped_labels = np.array(unswapped_labels)

    def on_epoch_end(self, epoch, logs=None):
        # Get true labels and predicted labels
        y_true, y_pred, y_prob = get_y_and_ypred(self.model, self.val_gen, self.label_maps)
        self.wandb_conf_matrix(y_true, y_pred)
        self.explore_and_log_distributions(y_true, np.array(y_pred), y_prob)
        

    def wandb_conf_matrix(self, y_true, y_pred):
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


    def explore_and_log_distributions(self, y_true, y_pred, final_pred_probs):
        fig, axs = plt.subplots(4 if cfg.data.include_induced else 3, 1, figsize=(10, 16))

        self.explore_regular_events(axs[0], y_true, y_pred, final_pred_probs, "noise")
        self.explore_regular_events(axs[1], y_true, y_pred, final_pred_probs, "earthquake")
        self.explore_regular_events(axs[2], y_true, y_pred, final_pred_probs, "explosion")
        if cfg.data.include_induced:
            self.explore_induced_events(axs[3], y_true, y_pred, final_pred_probs)


        plt.tight_layout()
        # Save the plot to a BytesIO object
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)

        # Create a wandb.Image object from the buffer
        img = wandb.Image(Image.open(img_buffer))
        wandb.log({"probability_distributions": img})

    def explore_induced_events(self, ax, y_true, y_pred, final_pred_probs):
        n_samples = len(y_true)
        unswapped_labels = np.array(self.unswapped_labels)[:n_samples]
        relevant_idx = np.where(unswapped_labels == 'induced or triggered event')[0]

        predictions_on_induced_events = y_pred[relevant_idx]
        final_pred_probs_on_induced_events = {
            "classifier": np.array(final_pred_probs["classifier"])[relevant_idx],
            "detector": np.array(final_pred_probs["detector"])[relevant_idx]
        }

        self.plot_prob_distributions(ax, final_pred_probs_on_induced_events, predictions_on_induced_events, "Induced Earthquakes")

    def explore_regular_events(self, ax, y_true, y_pred, final_pred_probs, target_label):
        n_samples = len(y_true)
        y_pred = np.array(y_pred)
        shortened_unswapped_labels = np.array(self.unswapped_labels)[:n_samples]

        relevant_idx = np.where(shortened_unswapped_labels == target_label)[0]
        predictions_on_target_events = y_pred[relevant_idx]
        final_pred_probs_on_target_events = {
            "classifier": np.array(final_pred_probs["classifier"])[relevant_idx],
            "detector": np.array(final_pred_probs["detector"])[relevant_idx]
        }

        self.plot_prob_distributions(ax, final_pred_probs_on_target_events, predictions_on_target_events, f'{str(target_label).title()} Events')

    def plot_prob_distributions(self, ax, pred_probs, final_pred_labels, title):
        stage_names = ['detector', 'classifier']
        for i, stage in enumerate(stage_names):
            stage_probs = pred_probs[stage]
            ax.hist(stage_probs, bins=np.linspace(0, 1, 50), alpha=0.5, label=f"{stage} stage")

        ax.set_title(title)
        ax.set_xlabel("Probability")
        ax.set_ylabel("Frequency")
        ax.legend('upper center')

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
