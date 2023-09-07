import matplotlib.pyplot as plt
from global_config import logger, cfg, model_cfg
import csv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, PrecisionRecallDisplay
import numpy as np
import os
import matplotlib.pyplot as plt
from obspy import Stream, Trace

class Analysis:
    def __init__(self, model, val_data, val_labels_onehot, label_map, date_and_time):
        self.model = model
        self.val_data = val_data
        self.val_labels_onehot = val_labels_onehot
        self.label_map = label_map
        self.date_and_time = date_and_time




    def collect_and_plot_samples(self, generator, metadata, num_samples=3):
        # Initialize a dictionary to hold the samples for each class based on label_map keys
        class_samples = {key: [] for key in self.label_map.keys()}
        used_indices = []
        
        # Iterate through the generator to collect samples
        for index in range(len(generator)):
            batch_data, batch_labels, original_indices = generator.get_item_with_index(index)
            for idx, (data, label_tensor) in enumerate(zip(batch_data, batch_labels)):
                label = np.argmax(label_tensor.numpy())  # Extract the class index
                if len(class_samples[label]) < num_samples:
                    data = data.numpy()
                    # logger.info(f"Found sample candidate of shape: {data.shape}")
                    # logger.info(f"Sample has min and max: ({np.min(data)}, {np.max(data)})")
                    class_samples[label].append(data)  # Assuming data is also a tensor
                    used_indices.append(original_indices[idx])

            # Check if we've collected enough samples for each class
            if all(len(samples) >= num_samples for samples in class_samples.values()):
                break
                
        for label in list(self.label_map.keys()):
            for sample_idx, sample in enumerate(class_samples[label]):
                relevant_metadata = metadata[used_indices[sample_idx]]
                trace_stream = self.get_trace_stream(sample, relevant_metadata)
                
                # Create a new plot for each time series
                fig, ax = plt.subplots(figsize=(15, 5))
                
                trace_stream.plot(handle=True, axes=ax)
                ax.set_title(f'Label: {label}')
                
                # Save plot with an appropriate name
                plot_name = f"collected_sample_{label}_{sample_idx}.png"
                plot_path = os.path.join(cfg.paths.plots_folder, plot_name)
                # logger.info(f"Sample plot {plot_name} generated")
                plt.savefig(plot_path)
                plt.close(fig)
        
        generator.on_epoch_end()

    def get_trace_stream(self, traces, metadata):
        traces = np.array(traces).T
        logger.info(f"Trace input shape {traces.shape}")
        starttime = metadata['trace_stats']['starttime']
        station = metadata['trace_stats']['station']
        channels = metadata['trace_stats']['channels']
        sampl_rate = metadata['trace_stats']['sampling_rate']
        trace_BHE = Trace(data=traces[0], header ={'station' : station,
                                            'channel' : channels[0],
                                            'sampling_rate' : sampl_rate,
                                            'starttime' : starttime})
        trace_BHN = Trace(data=traces[1], header ={'station' : station,
                                                'channel' : channels[1],
                                                'sampling_rate' : sampl_rate,
                                                'starttime' : starttime})
        trace_BHZ = Trace(data=traces[2], header ={'station' : station,
                                                'channel' : channels[2],
                                                'sampling_rate' : sampl_rate,
                                                'starttime' : starttime})
        stream = Stream([trace_BHE, trace_BHN, trace_BHZ])
        return stream


    def main(self):
        self.plot_confusion_matrix()
        self.plot_precision_recall_curve()
        self.incorrect_predictions_overview()
    
    def plot_confusion_matrix(self):
        # Get predictions
        pred_probs = self.model.predict(self.val_data)
        pred_labels = np.argmax(pred_probs, axis=1)
        
        # Get true labels
        true_labels = np.argmax(self.val_labels_onehot, axis=1)
        
        # Compute confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        
        # Plotting
        plt.figure()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.label_map.values())
        disp.plot(cmap=plt.cm.Blues)
        
        # Save the plot
        plt.savefig(f"{cfg.paths.plots_folder}/conf_{cfg.model}_{self.date_and_time}.png")
        plt.close()

    def plot_precision_recall_curve(self):
        # Find the least frequent class
        class_counts = np.sum(self.val_labels_onehot, axis=0)
        least_frequent_class = np.argmin(class_counts)
        
        # Get predictions for least frequent class
        pred_probs = self.model.predict(self.val_data)[:, least_frequent_class]
        true_labels = self.val_labels_onehot[:, least_frequent_class]
        
        # Compute precision-recall curve
        precision, recall, _ = precision_recall_curve(true_labels, pred_probs)
        
        # Plotting
        plt.figure()
        disp = PrecisionRecallDisplay(precision=precision, recall=recall)
        disp.plot()
        
        # Save the plot
        plt.savefig(f"{cfg.paths.plots_folder}/prc_{cfg.model}_{self.date_and_time}.png")
        plt.close()

    def incorrect_predictions_overview(self):
        # Get predictions
        pred_probs = self.model.predict(self.val_data)
        pred_labels = np.argmax(pred_probs, axis=1)
        
        # Get true labels
        true_labels = np.argmax(self.val_labels_onehot, axis=1)
        
        incorrect_indices = np.where(pred_labels != true_labels)[0]
        
        csv_file_path = f"{cfg.paths.predictions_folder}/{cfg.model}_wrong_predictions_{self.date_and_time}.csv"
        
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Index', 'Predicted', 'True'])
            
            for idx in incorrect_indices:
                writer.writerow([idx, self.label_map[pred_labels[idx]], self.label_map[true_labels[idx]]])