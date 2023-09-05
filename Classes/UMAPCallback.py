from tensorflow.keras.callbacks import Callback
import umap
import wandb
import matplotlib.pyplot as plt

class UMAPCallback(Callback):
    def __init__(self, val_data, val_labels_onehot, label_map, interval=5):
        super(UMAPCallback, self).__init__()
        self.val_data = val_data
        self.val_labels_onehot = val_labels_onehot
        self.label_map = label_map
        self.interval = interval
    
    def on_epoch_end(self, epoch, logs=None):
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
