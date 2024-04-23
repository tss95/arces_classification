from tensorflow.keras.callbacks import Callback
import umap
import wandb
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from typing import Dict, Optional

class UMAPCallback(Callback):
    def __init__(self, val_data: np.ndarray, val_labels_onehot: np.ndarray, label_map: Dict, interval: int = 5):
        """
        Initialize the UMAPCallback.

        Args:
            val_data (np.ndarray): Validation data to be visualized using UMAP.
            val_labels_onehot (np.ndarray): One-hot encoded labels for the validation data.
            label_map (dict): Dictionary mapping label indices to human-readable names.
            interval (int, optional): Frequency of epochs at which the callback is executed. Defaults to 5.

        This callback is intended for use with neural network models trained using Keras. At specified intervals,
        it applies UMAP dimensionality reduction to the outputs of the penultimate layer of the model and produces
        a scatter plot of these outputs, color-coded by their labels. This plot is then logged to Wandb for visualization.
        """
        super(UMAPCallback, self).__init__()
        self.val_data = val_data
        self.val_labels_onehot = val_labels_onehot
        self.label_map = label_map
        self.interval = interval
    
    def on_epoch_end(self, epoch: int, logs: Dict = None):
        """
        Perform actions at the end of each epoch.

        Args:
            epoch (int): The current epoch number.
            logs (dict, optional): A dictionary of logs from the training process.

        This method is automatically called at the end of each epoch during training. If the current epoch
        is a multiple of the specified interval, it computes the UMAP embedding for the outputs of the model's
        penultimate layer and creates a scatter plot, which is then logged to Wandb.
        """
        if (epoch % self.interval) == 0:
            # Get the output from the last layer before the output layer
            model = self.model
            layer_output_model = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-2].output)
            layer_output = layer_output_model.predict(self.val_data)
            
            # Perform UMAP dimensionality reduction
            reducer = umap.UMAP()
            embedding = reducer.fit_transform(layer_output)
            
            # Get labels for color mapping
            labels = np.argmax(self.val_labels_onehot, axis=1)
            unique_labels = np.unique(labels)
            
            # Plotting
            plt.figure()
            for i in unique_labels:
                plt.scatter(embedding[labels==i, 0], embedding[labels==i, 1], label=self.label_map[i])
            plt.legend()
            
            # Save the plot to a file and then log it to Wandb
            plot_file_path = f"umap_plot_epoch_{epoch}.png"
            plt.savefig(plot_file_path)
            plt.close()

            # Log the UMAP plot to Wandb
            wandb.log({"umap_epoch_{}".format(epoch): [wandb.Image(plot_file_path, caption=f"UMAP at Epoch {epoch}")]})
