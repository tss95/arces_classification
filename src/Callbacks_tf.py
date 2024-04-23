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
from src.Utils import get_y_and_ypred, plot_confusion_matrix  # Assuming this is in Classes.Utils
from typing import List, Dict, Any, Optional

class ValidationConfusionMatrixCallback(tf.keras.callbacks.Callback):
    """
    A TensorFlow Keras Callback to compute and log confusion matrices and distribution of predictions.

    Attributes:
        val_gen (tf.keras.utils.Sequence): The validation data generator.
        label_maps (Dict[str, int]): Mapping of labels to integers.
        unswapped_labels (np.ndarray): Original labels before any swapping.
    """  
    def __init__(self, val_gen, label_maps: Dict[str, int], unswapped_labels: List[str]):
        """
        Initialize the ValidationConfusionMatrixCallback.

        Args:
            val_gen (tf.keras.utils.Sequence): The validation data generator.
            label_maps (Dict[str, int]): Mapping of labels to integers.
            unswapped_labels (List[str]): Original labels before any swapping.
        """
        super().__init__()
        self.val_gen = val_gen
        self.label_maps = label_maps
        self.unswapped_labels = np.array(unswapped_labels)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """
        Called at the end of an epoch during training. Computes and logs confusion matrix and prediction distributions.

        Args:
            epoch (int): Current epoch number.
            logs (Optional[Dict[str, Any]]): Logs for the training epoch.
        """
        # Get true labels and predicted labels
        y_true, y_pred, y_prob = get_y_and_ypred(self.model, self.val_gen, self.label_maps)
        self.wandb_conf_matrix(y_true, y_pred, epoch)
        self.explore_and_log_distributions(y_true, np.array(y_pred), y_prob, epoch)
        

    def wandb_conf_matrix(self, y_true: List[str], y_pred: List[str], epoch: int):
        """
        Computes and logs the confusion matrix to Weights and Biases (wandb).

        This method computes the confusion matrix from the true and predicted labels, normalizes it, and logs it using wandb.

        Args:
            y_true (List[str]): True labels of the validation set.
            y_pred (List[str]): Predicted labels of the validation set.
            epoch (int): The current epoch number.
        """
        # Compute the confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred, labels=["noise", "earthquake", "explosion"])
        
        # Normalize the confusion matrix
        conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        
        # Convert to integer labels
        label_to_int = {"noise": 0, "earthquake": 1, "explosion": 2}
        y_true_int = [label_to_int[label] for label in y_true]
        y_pred_int = [label_to_int[label] for label in y_pred]
        
        plt = plot_confusion_matrix(conf_matrix, conf_matrix_normalized, ["noise", "earthquake", "explosion"])
        wandb.log({"confusion_matrix": plt}, step = epoch)


    def explore_and_log_distributions(self, y_true: List[str], y_pred: List[str], final_pred_probs: np.ndarray, epoch: int):
        """
        Explores and logs the distributions of prediction probabilities for different event types.

        This method visualizes the prediction probability distributions for each type of event and logs them using wandb.

        Args:
            y_true (List[str]): True labels of the validation set.
            y_pred (List[str]): Predicted labels of the validation set.
            final_pred_probs (np.ndarray): The array of prediction probabilities.
            epoch (int): The current epoch number.
        """
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
        wandb.log({"probability_distributions": img}, step = epoch)

    def explore_induced_events(self, ax: plt.Axes, y_true: List[str], y_pred: List[str], final_pred_probs: Dict[str, np.ndarray]):
        """
        Explores and visualizes prediction probabilities specifically for induced or triggered events.

        Args:
            ax (plt.Axes): The matplotlib Axes object to plot on.
            y_true (List[str]): True labels of the validation set.
            y_pred (List[str]): Predicted labels of the validation set.
            final_pred_probs (Dict[str, np.ndarray]): Dictionary containing prediction probabilities from different model stages.
        """
        n_samples = len(y_true)
        unswapped_labels = np.array(self.unswapped_labels)[:n_samples]
        relevant_idx = np.where(unswapped_labels == 'induced or triggered event')[0]

        predictions_on_induced_events = y_pred[relevant_idx]
        final_pred_probs_on_induced_events = {
            "classifier": np.array(final_pred_probs["classifier"])[relevant_idx],
            "detector": np.array(final_pred_probs["detector"])[relevant_idx]
        }

        self.plot_prob_distributions(ax, final_pred_probs_on_induced_events, predictions_on_induced_events, "Induced Earthquakes")

    def explore_regular_events(self, ax: plt.Axes, y_true: List[str], y_pred: List[str], final_pred_probs: Dict[str, np.ndarray], target_label: str):
        """
        Explores and visualizes prediction probabilities for regular events (non-induced).

        Args:
            ax (plt.Axes): The matplotlib Axes object to plot on.
            y_true (List[str]): True labels of the validation set.
            y_pred (List[str]): Predicted labels of the validation set.
            final_pred_probs (Dict[str, np.ndarray]): Dictionary containing prediction probabilities from different model stages.
            target_label (str): The label of the target event type to explore.
        """
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

    def plot_prob_distributions(self, ax: plt.Axes, pred_probs: Dict[str, np.ndarray], final_pred_labels: List[str], title: str):
        """
        Plots the probability distributions for given predictions.

        Args:
            ax (plt.Axes): The matplotlib Axes object to plot on.
            pred_probs (Dict[str, np.ndarray]): Dictionary containing prediction probabilities from different model stages.
            final_pred_labels (List[str]): The final predicted labels.
            title (str): The title of the plot.
        """
        stage_names = ['detector', 'classifier']
        for i, stage in enumerate(stage_names):
            stage_probs = pred_probs[stage]
            ax.hist(stage_probs, bins=np.linspace(0, 1, 50), alpha=0.5, label=f"{stage} stage")

        ax.set_title(title)
        ax.set_xlabel("Probability")
        ax.set_ylabel("Frequency")
        ax.legend('upper center')

class InPlaceProgressCallback(tf.keras.callbacks.Callback):
    """
    A TensorFlow Keras Callback to monitor and print the progress of training in-place.

    Attributes:
        epochs (int): Total number of epochs.
        steps_per_epoch (int): Number of steps per epoch.
        current_step (int): Current step within the current epoch.
        cumulative_train_loss (float): Cumulative training loss.
    """
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """
        Called at the beginning of training. Initializes the total number of epochs and steps per epoch.

        Args:
            logs (Optional[Dict[str, Any]]): Logs for the training.
        """
        self.epochs = self.params['epochs']
        self.steps_per_epoch = self.params['steps']
        
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """
        Called at the beginning of each epoch. Initializes epoch parameters.

        Args:
            epoch (int): Current epoch number.
            logs (Optional[Dict[str, Any]]): Logs for the epoch.
        """
        self.current_step = 0
        self.cumulative_train_loss = 0.0  # Initialize the cumulative train loss for the epoch
        print(f"\n Epoch {epoch + 1}/{self.epochs}")

     def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        """
        Called at the end of each batch. Updates the progress bar and prints current loss.

        Args:
            batch (int): Current batch number.
            logs (Optional[Dict[str, Any]]): Logs for the batch.
        """
        self.current_step += 1
        self.cumulative_train_loss += logs['train_total_loss']  # Accumulate the batch train loss
        
        avg_train_loss = self.cumulative_train_loss / self.current_step  # Compute the average train loss so far
        
        progbar = "=" * (self.current_step * 50 // self.steps_per_epoch)
        progbar += "-" * (50 - len(progbar))
        
        metrics_str = f" avg_train_total_loss: {avg_train_loss:.4f}"
        
        print(f"\r[{progbar}] {self.current_step}/{self.steps_per_epoch}{metrics_str}", end="")
        
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """
        Called at the end of each epoch. Prints completion message.

        Args:
            epoch (int): Current epoch number.
            logs (Optional[Dict[str, Any]]): Logs for the epoch.
        """
        print(f"\n Epoch {epoch + 1}/{self.epochs} completed")


class WandbLoggingCallback(tf.keras.callbacks.Callback):
    """
    A TensorFlow Keras Callback for logging metrics to Weights and Biases (wandb).

    This callback logs training and validation metrics to wandb at the end of each epoch.
    """
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None):
        """
        Called at the end of each epoch. Logs metrics to wandb.

        Args:
            epoch (int): The current epoch number.
            logs (Optional[Dict[str, float]]): The metric logs for the epoch.
        """
        if logs is not None:
            train_metrics = {k: logs[k] for k in logs if not k.startswith('val_')}
            val_metrics = {k: logs[k] for k in logs if k.startswith('val_')}
            wandb.log({"epoch": epoch, **train_metrics, **val_metrics})

# Usage example with model.fit()
# model.fit(x, y, epochs=10, verbose=0, callbacks=[InPlaceProgressCallback()])

class CosineAnnealingLearningRateScheduler(tf.keras.callbacks.Callback):
    """
    A TensorFlow Keras Callback for a cosine annealing learning rate schedule.

    Attributes:
        total_epochs (int): Total number of epochs for training.
        max_lr (float): Maximum learning rate.
        min_lr (float): Minimum learning rate.
        warmup_epochs (int): Number of warm-up epochs.
    """
    def __init__(self, total_epochs: int, max_lr: float, min_lr: float):
        """
        Initializes the cosine annealing learning rate scheduler.

        Args:
            total_epochs (int): Total number of epochs for training.
            max_lr (float): Maximum learning rate.
            min_lr (float): Minimum learning rate.
        """
        super().__init__()
        self.total_epochs = total_epochs
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_epochs = int(0.05 * total_epochs)  # 10% of total epochs for warm-up

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """
        Called at the beginning of each epoch. Adjusts the learning rate based on the cosine annealing schedule.

        Args:
            epoch (int): The current epoch number.
            logs (Optional[Dict[str, Any]]): Logs for the epoch.
        """
        if epoch < self.warmup_epochs and cfg.optimizer.optimizer_kwargs.warmup:
            lr = (self.max_lr - self.min_lr) / self.warmup_epochs * epoch + self.min_lr
        else:
            lr = self.min_lr + (self.max_lr - self.min_lr) * (
                1 + np.cos(np.pi * (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs))
            ) / 2
        
        # Set the new learning rate to the model
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
