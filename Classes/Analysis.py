import matplotlib.pyplot as plt
from global_config import logger, cfg
import csv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, PrecisionRecallDisplay
import numpy as np
import os
import matplotlib.pyplot as plt
from obspy import Stream, Trace
import tensorflow as tf

class Analysis:
    def __init__(self, model, val_gen, label_maps_dict, date_and_time):
        self.model = model
        self.val_gen = val_gen
        self.label_maps = label_maps_dict
        self.date_and_time = date_and_time




    def collect_and_plot_samples(self, generator, metadata, num_samples=3):
        # Initialize a dictionary to hold the samples for each class based on label_map keys
        class_samples = {key: [] for key in self.label_maps.keys()}
        used_indices = []
        # TODO: This function no longer works.
        
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
                
        for label in list(self.label_maps.keys()):
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
        #self.plot_precision_recall_curve()
        self.incorrect_predictions_overview()

    def get_final_labels(self, pred_probs, label_map):
        """
        Given the predicted probabilities from the model, return the final label.
        
        Parameters:
        pred_probs (dict): A dictionary containing the predicted probabilities for both 'detector' and 'classifier'.
        label_map (dict): A dictionary containing mappings from label indices to their string representations.
        
        Returns:
        list: A list of final labels.
        """
        final_labels = []
        pred_probs_detector = tf.sigmoid(tf.cast(pred_probs['detector'], tf.float32)).numpy()       
        pred_labels_detector = self.apply_threshold(pred_probs_detector)

        pred_probs_classifier = tf.sigmoid(tf.cast(pred_probs['classifier'], tf.float32)).numpy()
        pred_labels_classifier = self.apply_threshold(pred_probs_classifier)

        final_labels = self.translate_labels(pred_labels_detector, pred_labels_classifier, label_map)
        return final_labels, {"detector": pred_probs_detector, "classifier": pred_probs_classifier}
            

    def translate_labels(self, labels_detector, labels_classifier, label_map):
        final_labels = []
        for det, cls in zip(labels_detector, labels_classifier):
            det = int(det[0]) if isinstance(det, np.ndarray) else det
            cls = int(cls[0]) if isinstance(cls, np.ndarray) else cls


            if label_map["detector"][det] == "noise":
                final_labels.append("noise")
            else:
                final_labels.append(label_map["classifier"][cls])
            
            logger.info(f"chosen label: {final_labels[-1]}")
        
        return final_labels

    def apply_threshold(self, pred_probs):
        out = []
        for prob in pred_probs:
            if prob <= cfg.data.model_threshold:
                out.append(0)
            else:
                out.append(1)
        return out

    def plot_precision_recall_curve(self):
        final_true_labels, _, final_pred_probs = self.get_y_and_ypred(self.val_gen)
        # Compute precision-recall curve
        precision, recall, _ = precision_recall_curve(final_true_labels, final_pred_probs)

        # Plotting
        plt.figure()
        disp = PrecisionRecallDisplay(precision=precision, recall=recall)
        disp.plot()

        # Save the plot
        plt.savefig(f"{cfg.paths.plots_folder}/prc_{cfg.model}_{self.date_and_time}.png")
        plt.close()
        self.val_gen.on_epoch_end()

    def get_y_and_ypred(self, val_gen):
        final_pred_labels = []
        final_true_labels = []
        final_pred_probs = []
        
        for batch_data, batch_labels in val_gen:
            pred_probs = self.model.predict(batch_data)
            labels, pred_probs = self.get_final_labels(pred_probs, self.label_maps)
            final_pred_labels.extend(labels)
            final_pred_probs.extend(pred_probs)
            final_true_labels.extend(self.translate_labels(batch_labels["detector"].numpy().astype(int), 
                                                           batch_labels["classifier"].numpy().astype(int), self.label_maps))
        return final_true_labels, final_pred_labels, final_pred_probs

    def plot_confusion_matrix(self):
        final_true_labels, final_pred_labels, _ = self.get_y_and_ypred(self.val_gen)
        # Convert lists to numpy arrays for sklearn functions
        final_pred_labels = np.array(final_pred_labels)
        final_true_labels = np.array(final_true_labels)
        logger.info(f"Unique True Labels: {np.unique(final_true_labels)}")
        logger.info(f"Unique Predicted Labels: {np.unique(final_pred_labels)}")
        # Get unique sorted labels
        unique_labels = sorted(np.unique(np.concatenate((final_pred_labels, final_true_labels))))

        # Compute confusion matrix
        cm = confusion_matrix(final_true_labels, final_pred_labels, labels=unique_labels)
        
        # Plotting
        plt.figure()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
        disp.plot(cmap=plt.cm.Blues)
        
        # Save the plot
        plt.savefig(f"{cfg.paths.plots_folder}/conf_{cfg.model}_{self.date_and_time}.png")
        plt.close()
        self.val_gen.on_epoch_end()

    def incorrect_predictions_overview(self):
        incorrect_indices = []
        final_true_labels, final_pred_labels, _ = self.get_y_and_ypred(self.val_gen)

        for i in range(len(final_pred_labels)):
            if final_pred_labels != final_true_labels:
                incorrect_indices.append(i)

        csv_file_path = f"{cfg.paths.predictions_folder}/{cfg.model}_wrong_predictions_{self.date_and_time}.csv"

        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Index', 'Predicted', 'True'])
            
            for idx in incorrect_indices:
                writer.writerow([idx, final_pred_labels[idx], final_true_labels[idx]])
        self.val_gen.on_epoch_end()
