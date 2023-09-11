import matplotlib.pyplot as plt
from global_config import logger, cfg
import csv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, PrecisionRecallDisplay
import numpy as np
import os
import matplotlib.pyplot as plt
from obspy import Stream, Trace

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
        self.plot_precision_recall_curve()
        self.incorrect_predictions_overview()

    def plot_precision_recall_curve(self):
        final_true_labels = []
        final_pred_probs = []

        for batch_data, batch_labels in self.val_gen:
            # Get predictions and true labels
            pred_probs = self.model.predict(batch_data)
            pred_probs_classifier = pred_probs['classifier'][:, 1]  # assuming 1 is "not_noise"

            true_labels = np.argmax(batch_labels['classifier'], axis=1)

            # Logic to append only when the detector is "not_noise"
            mask_not_noise = (np.argmax(batch_labels['detector'], axis=1) == 1)  # assuming 1 is "not_noise"
            true_labels = true_labels[mask_not_noise]
            pred_probs_classifier = pred_probs_classifier[mask_not_noise]

            final_pred_probs.extend(pred_probs_classifier)
            final_true_labels.extend(true_labels)

        # Compute precision-recall curve
        precision, recall, _ = precision_recall_curve(final_true_labels, final_pred_probs)

        # Plotting
        plt.figure()
        disp = PrecisionRecallDisplay(precision=precision, recall=recall)
        disp.plot()

        # Save the plot
        plt.savefig(f"{cfg.paths.plots_folder}/prc_{cfg.model}_{self.date_and_time}.png")
        plt.close()

        
    def plot_confusion_matrix(self):
        final_pred_labels = []
        final_true_labels = []
        
        for batch_data, batch_labels in self.val_gen:
            # Get predictions for both detector and classifier
            pred_probs = self.model.predict(batch_data)
            pred_labels_detector = np.argmax(pred_probs['detector'], axis=1)
            pred_labels_classifier = np.argmax(pred_probs['classifier'], axis=1)

            # Get true labels for both
            true_labels_detector = np.argmax(batch_labels['detector'], axis=1)
            true_labels_classifier = np.argmax(batch_labels['classifier'], axis=1)

            # Logic to combine detector and classifier predictions
            for i in range(len(pred_labels_detector)):
                if pred_labels_detector[i] == self.label_map['detector']['noise']:
                    final_pred_labels.append(self.label_map['final']['noise'])
                else:
                    final_pred_labels.append(self.label_map['final'][self.label_map['classifier'][pred_labels_classifier[i]]])

                if true_labels_detector[i] == self.label_map['detector']['noise']:
                    final_true_labels.append(self.label_map['final']['noise'])
                else:
                    final_true_labels.append(self.label_map['final'][self.label_map['classifier'][true_labels_classifier[i]]])

        # Convert lists to numpy arrays for sklearn functions
        final_pred_labels = np.array(final_pred_labels)
        final_true_labels = np.array(final_true_labels)

        # Compute confusion matrix
        cm = confusion_matrix(final_true_labels, final_pred_labels)
        
        # Plotting
        plt.figure()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.label_map['final'].values())
        disp.plot(cmap=plt.cm.Blues)
        
        # Save the plot
        plt.savefig(f"{cfg.paths.plots_folder}/conf_{cfg.model}_{self.date_and_time}.png")
        plt.close()
        self.val_gen.on_epoch_end()

    def incorrect_predictions_overview(self):
        final_pred_labels = []
        final_true_labels = []
        incorrect_indices = []

        for batch_idx, (batch_data, batch_labels) in enumerate(self.val_generator):
            pred_probs = self.model.predict(batch_data)
            
            pred_labels_detector = np.argmax(pred_probs['detector'], axis=1)
            pred_labels_classifier = np.argmax(pred_probs['classifier'], axis=1)
            
            true_labels_detector = np.argmax(batch_labels['detector'], axis=1)
            true_labels_classifier = np.argmax(batch_labels['classifier'], axis=1)

            batch_size = len(pred_labels_detector)
            
            for i in range(batch_size):
                if pred_labels_detector[i] in self.label_map_detector:
                    final_pred_labels.append(self.label_map_detector[pred_labels_detector[i]])
                else:
                    # Fallback if label is not in map
                    final_pred_labels.append(str(pred_labels_detector[i]))

                if true_labels_detector[i] in self.label_map_detector:
                    final_true_labels.append(self.label_map_detector[true_labels_detector[i]])
                else:
                    # Fallback if label is not in map
                    final_true_labels.append(str(true_labels_detector[i]))
                
                # Check for incorrect prediction
                if final_pred_labels[-1] != final_true_labels[-1]:
                    incorrect_indices.append(batch_idx * batch_size + i)

        csv_file_path = f"{cfg.paths.predictions_folder}/{cfg.model}_wrong_predictions_{self.date_and_time}.csv"

        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Index', 'Predicted', 'True'])
            
            for idx in incorrect_indices:
                writer.writerow([idx, final_pred_labels[idx], final_true_labels[idx]])
        self.val_generator.on_epoch_end()
