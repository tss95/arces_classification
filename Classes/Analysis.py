import matplotlib.pyplot as plt
from global_config import logger, cfg
import csv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, PrecisionRecallDisplay
import numpy as np
import os
import matplotlib.pyplot as plt
from obspy import Stream, Trace
import tensorflow as tf
from Classes.Utils import get_final_labels, translate_labels, apply_threshold, get_y_and_ypred, get_index_of_wrong_predictions
from Classes.Augment import augment_pipeline
from collections import Counter
import math
import geopandas as gpd

class Analysis:
    def __init__(self, model, val_gen, label_maps_dict, date_and_time):
        self.model = model
        self.val_gen = val_gen
        self.label_maps = label_maps_dict
        self.date_and_time = date_and_time

    def collect_and_plot_samples(self, generator, metadata, num_samples=3):
        # Initialize a dictionary to hold the samples for each class based on label_map keys
        real_labels = []
        eq_samples, ex_samples, no_samples = [], [], []
        
        # Iterate through the generator to collect samples
        for index in range(len(generator)):
            if len(eq_samples) >= num_samples and len(ex_samples) >= num_samples and len(no_samples) >= num_samples:
                break
            print(f"Iterating at index: {index} / {num_samples - 1}")
            batch_data, batch_labels, original_indices = generator.get_item_with_index(index)
            #batch_data, batch_labels = augment_pipeline(batch_data, batch_labels)
            string_labels = translate_labels(list(batch_labels["detector"]), list(batch_labels["classifier"]), self.label_maps)
            for idx, label in enumerate(string_labels):
                if len(eq_samples) < num_samples:
                    if label == "earthquake":
                        eq_samples.append((batch_data[idx], original_indices[idx]))
                        real_labels.append(label)
                if len(ex_samples) < num_samples:
                    if label == "explosion":
                        ex_samples.append((batch_data[idx], original_indices[idx]))
                        real_labels.append(label)
                if len(no_samples) < num_samples:
                    if label == "noise":
                        no_samples.append((batch_data[idx], original_indices[idx]))
                        real_labels.append(label)

        # Create a dictionary to map samples
        class_samples = {
            "earthquake": eq_samples,
            "explosion": ex_samples,
            "noise": no_samples
        }
        
        # Plotting
        for label, samples in class_samples.items():
            for sample_idx, (sample, original_idx) in enumerate(samples):
                relevant_metadata = metadata[original_idx]
                trace_stream = self.get_trace_stream(sample, relevant_metadata)
                
                # Create a new plot for each time series
                fig, ax = plt.subplots(figsize=(15, 5))
                if trace_stream:
                    trace_stream.plot(handle=True, axes=ax)  # Assuming trace_stream has a plot method
                ax.set_title(f'Label: {label}')
                
                # Save plot with an appropriate name
                plot_name = f"collected_sample_{label}_{sample_idx}.png"
                plot_path = os.path.join(cfg.paths.plots_folder, plot_name)
                print(f"Sample plot {plot_name} generated")
                plt.savefig(plot_path)
                plt.close(fig)
        
        # Assuming the generator has an on_epoch_end method
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
    
    def explore_induced_events(self, generator, metadata, unswapped_labels):
        final_true_labels, final_pred_labels, final_pred_probs = get_y_and_ypred(self.model, generator, self.label_maps)
        final_true_labels = np.array(final_true_labels)
        final_pred_labels = np.array(final_pred_labels)
        n_samples = len(final_true_labels)
        unswapped_labels = np.array(unswapped_labels)[:n_samples]
        relevant_idx = np.where(unswapped_labels == 'induced or triggered event')[0]

        print("Size of unswapped_labels: ", len(unswapped_labels))
        print("Size of final_pred_labels: ", len(final_pred_labels))
        print("Max index in relevant_idx: ", np.max(relevant_idx))

        predictions_on_induced_events = final_pred_labels[relevant_idx]
        final_pred_probs_on_induced_events = {"classifier": np.array(final_pred_probs["classifier"])[relevant_idx],
                                              "detector":   np.array(final_pred_probs["detector"])[relevant_idx]}
        true_labels_on_induced_events = final_true_labels[relevant_idx]

        # Get unique sorted labels

        self.plot_prob_distributions_two_stages({"detector": final_pred_probs_on_induced_events["detector"],
                                    "classifier": final_pred_probs_on_induced_events["classifier"]},
                                   predictions_on_induced_events,
                                   ['earthquake', 'explosion', 'noise'],
                                   ['detector', 'classifier'],
                                   "Induced Earthquakes")

    def explore_regular_events(self, generator, metadata, target_label):
        final_true_labels, final_pred_labels, final_pred_probs = get_y_and_ypred(self.model, generator, self.label_maps)
        final_true_labels = np.array(final_true_labels)
        final_pred_labels = np.array(final_pred_labels)
        relevant_idx = np.where(final_true_labels == target_label)[0]

        predictions_on_target_events = final_pred_labels[relevant_idx]
        final_pred_probs_on_target_events = {"classifier": np.array(final_pred_probs["classifier"])[relevant_idx],
                                              "detector":   np.array(final_pred_probs["detector"])[relevant_idx]}
        true_labels_on_target_events = final_true_labels[relevant_idx] # Doesnt seem like this is needed in this function

        self.plot_prob_distributions_two_stages({"detector": final_pred_probs_on_target_events["detector"],
                                    "classifier": final_pred_probs_on_target_events["classifier"]},
                                    predictions_on_target_events,
                                    ['earthquake', 'explosion', 'noise'],
                                    ['detector', 'classifier'],
                                    f'{str(target_label).title()} events')



    def plot_prob_distributions_two_stages(self, pred_probs, final_pred_labels, label_names, stage_names, title):
        fig, axs = plt.subplots(len(stage_names) + 1, 1, figsize=(10, 12))
        
        for i, stage in enumerate(stage_names):
            # Isolate the probabilities for the given stage
            stage_probs = pred_probs[stage]
            
            # Plot the distribution
            axs[i].hist(stage_probs, bins=np.linspace(0, 1, 50), alpha=0.5)
            axs[i].set_title(f"Probability Distribution at {stage} Stage")
            axs[i].set_xlabel("Probability")
            axs[i].set_ylabel("Frequency")
        
        # Add table with final prediction counts
        columns = ['Label', 'Count']
        # Count occurrences of each label
        label_counts = Counter(final_pred_labels)

        # Create cell_data
        cell_data = [[label, label_counts.get(label, 0)] for label in label_names]

        
        axs[-1].axis('tight')
        axs[-1].axis('off')
        axs[-1].table(cellText=cell_data, colLabels=columns, cellLoc='center', loc='center')
        
        plt.suptitle(f"Probability Distributions of {title} for Model {cfg.model} on {self.date_and_time}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save the plot
        plt.savefig(f"{cfg.paths.plots_folder}/{title}_prob_distributions_{cfg.model}_{self.date_and_time}.png")
        plt.close()
        

    def plot_mistakes_by_distance(self, val_meta, bin_step=50):
        # Get the indices of wrong predictions using your function
        true_labels, predicted_labels, _ = get_y_and_ypred(self.model, self.val_gen, self.label_maps)
        meta = {}
        for i in range(len(true_labels)):
            meta[i] = val_meta[i]
        # Filter out noise by index events:
        noise_idx = np.where(np.array(true_labels) == 'noise')[0]
        not_noise_idx = np.where(np.array(true_labels) != 'noise')[0]
        true_labels = np.delete(true_labels, noise_idx)
        predicted_labels = np.delete(predicted_labels, noise_idx)
        m = {}
        for idx in not_noise_idx:
            m[idx] = meta[idx]
        val_meta = m
        wrong_indices = np.array(get_index_of_wrong_predictions(true_labels, predicted_labels))
        meta_indexes = sorted(val_meta.keys())
        relevant_meta = {}
        for i, idx in enumerate(wrong_indices):
            relevant_meta[i] = val_meta[meta_indexes[idx]]
        true_labels = np.array(true_labels)[wrong_indices]
        predicted_labels = np.array(predicted_labels)[wrong_indices]
        distances = [meta['dist_to_arces'] for meta in relevant_meta]

        # Dynamically create distance bins based on data and desired bin step (50 or 100 km)
        min_distance = math.floor(min(distances) / bin_step) * bin_step
        max_distance = math.ceil(max(distances) / bin_step) * bin_step
        distance_bins = np.arange(min_distance, max_distance + bin_step, bin_step)

        # Count events in each distance bin
        total_counts, _ = np.histogram(distances, bins=distance_bins)

        # Count mistakes in each distance bin
        mistake_distances = np.array(distances)[wrong_indices]
        mistake_counts, _ = np.histogram(mistake_distances, bins=distance_bins)

        # Calculate mistake rates
        with np.errstate(divide='ignore', invalid='ignore'):
            mistake_rates = mistake_counts / total_counts
            mistake_rates[np.isnan(mistake_rates)] = 0

        # Plot mistake rates
        plt.figure()
        plt.bar(distance_bins[:-1], mistake_rates, width=bin_step-1, align='edge', tick_label=[str(bin) for bin in distance_bins[:-1]])
        plt.xlabel("Distance bin (km)")
        plt.ylabel("Mistake rate")
        plt.title(f"Mistake rate by distance to event (Bin step: {bin_step} km)")
        plt.savefig(f"{cfg.paths.plots_folder}/mistakes_by_distance_{cfg.model}_{self.date_and_time}.png")
        plt.close()
        self.val_gen.on_epoch_end()

    def plot_events_on_map(self, val_meta, bin_step=50):
       # Get the indices of wrong predictions using your function
        true_labels, predicted_labels, _ = get_y_and_ypred(self.model, self.val_gen, self.label_maps)
        meta = {}
        for i in range(len(true_labels)):
            meta[i] = val_meta[i]
        # Filter out noise by index events:
        noise_idx = np.where(np.array(true_labels) == 'noise')[0]
        not_noise_idx = np.where(np.array(true_labels) != 'noise')[0]
        true_labels = np.delete(true_labels, noise_idx)
        predicted_labels = np.delete(predicted_labels, noise_idx)
        m = {}
        for idx in not_noise_idx:
            m[idx] = meta[idx]
        val_meta = m
        
        # Load map (replace with your specific shapefile for Scandinavia)
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        ax = world[world.name.isin(['Norway', 'Sweden', 'Finland', 'Iceland'])].plot()

        # Separate latitudes and longitudes
        longitudes = [meta['origins'][0]['longitude'] for meta in relevant_meta]
        latitudes = [meta['origins'][0]['latitude'] for meta in relevant_meta]

        # Filter the event types and locations for wrong predictions
        wrong_lats = [latitudes[i] for i in wrong_indices]
        wrong_longs = [longitudes[i] for i in wrong_indices]
        wrong_events = [true_labels[i] for i in wrong_indices]
        
        for lat, long, event in zip(wrong_lats, wrong_longs, wrong_events):
            if event == 'explosion':
                plt.scatter(long, lat, c='red', label='Wrong Explosion', marker='o')
            elif event == 'earthquake':
                plt.scatter(long, lat, c='green', label='Wrong Earthquake', marker='^')

        plt.legend()
        plt.savefig(f"{cfg.paths.plots_folder}/map_errors_{cfg.model}_{self.date_and_time}.png")
        plt.close()
        self.val_gen.on_epoch_end()


    def plot_precision_recall_curve(self):
        final_true_labels, _, final_pred_probs = get_y_and_ypred(self.model, self.val_gen, self.label_maps)
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

    def plot_confusion_matrix(self):
        final_true_labels, final_pred_labels, _ = get_y_and_ypred(self.model, self.val_gen, self.label_maps)
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
        final_true_labels, final_pred_labels, _ = get_y_and_ypred(self.model, self.val_gen, self.label_maps)

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
