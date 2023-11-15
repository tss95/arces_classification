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
import contextily as ctx

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

    def convert_to_detector_labels(self, true_labels, predicted_labels):
        conv_true, conv_pred = [], []
        for i in range(len(true_labels)):
            conv_true.append(self.cast_to_detector(true_labels[i]))
            conv_pred.append(self.cast_to_detector(predicted_labels[i]))
        return conv_true, conv_pred

    def cast_to_detector(self, label):
        if label != "noise":
            return "event"
        return "noise"

        
    def plot_d_mistakes_by_distance(self, val_meta, bin_step=100):
        # Rather than error rate by the right class, looking at the error rate of each stage would be better.
        # For example, a distant event is more important to detect corretly than classify corretly.
        true_labels, predicted_labels, _ = get_y_and_ypred(self.model, self.val_gen, self.label_maps)
        true_labels, predicted_labels = self.convert_to_detector_labels(true_labels, predicted_labels)
        # Match the length of meta to the length of true_labels and predicted_labels. This will align indexes
        meta = {}
        for i in range(len(true_labels)):
            meta[i] = val_meta[i]
        # Get noise indexes
        noise_idx = np.where(np.array(true_labels) == 'noise')[0]
        not_noise_idx = np.where(np.array(true_labels) != 'noise')[0]
        # Filter out noise by index events:
        true_labels = np.delete(true_labels, noise_idx)
        predicted_labels = np.delete(predicted_labels, noise_idx)
        m = {}
        for idx in not_noise_idx:
            m[idx] = meta[idx]
        not_noise_meta = m
        # Get the distances of all non noise events
        distances_all = []
        for idx in list(not_noise_meta.keys()):
            distances_all.append(not_noise_meta[idx]['dist_to_arces'])
        # Get the indexes of wrong predictions
        wrong_indices = np.array(get_index_of_wrong_predictions(true_labels, predicted_labels))
        # Get the meta indexes of the wrong predictions
        not_noise_meta_indexes = sorted(not_noise_meta.keys())
        # Get the meta of the wrong predictions
        not_noise_error_metas = {}
        for i, idx in enumerate(wrong_indices):
            not_noise_error_metas[i] = not_noise_meta[not_noise_meta_indexes[idx]]
        # Get the true and predicted labels of the wrong predictions
        true_labels = np.array(true_labels)[wrong_indices]
        predicted_labels = np.array(predicted_labels)[wrong_indices]

        distances = [] # Distances of non noise events incorrectly predicted
        for i in list(not_noise_error_metas.keys()):
            distances.append(not_noise_error_metas[i]['dist_to_arces'])


        # Dynamically create distance bins based on data and desired bin step (e.g., 100 km)
        min_distance = math.floor(min(distances) / bin_step) * bin_step
        max_distance = math.ceil(max(distances) / bin_step) * bin_step
        distance_bins = np.arange(min_distance, max_distance + bin_step, bin_step)

        # Count mistakes in each distance bin
        mistake_counts, _ = np.histogram(distances, bins=distance_bins)

        # Count total events in each distance bin
        total_counts, _ = np.histogram(distances_all, bins=distance_bins)


        # Calculate mistake rates
        with np.errstate(divide='ignore', invalid='ignore'):
            mistake_rates = np.true_divide(mistake_counts, total_counts)
            mistake_rates = np.nan_to_num(mistake_rates)  # Convert NaNs to zeros

        # Plot mistake rates
        plt.figure(figsize=(12, 7))  # Increased figure size for readability
        tick_labels = [f"{int(bin_)}" for bin_ in distance_bins[:-1]]
        plt.bar(distance_bins[:-1] + bin_step/2, mistake_rates, width=bin_step * 0.9, align='center')
        plt.xticks(ticks=distance_bins[:-1] + bin_step/2, labels=tick_labels, rotation=45, ha='right', fontsize=8)  # Reduced font size
        plt.xlabel("Distance bin (km)")
        plt.ylabel("Mistake rate")
        plt.title(f"Detector mistake rate by distance to event (Bin step: {bin_step} km)")
        plt.grid(True)
        plt.tight_layout()   # Adjust the plot to ensure everything fits without overlapping
        plt.savefig(f"{cfg.paths.plots_folder}/d_mistakes_by_distance_{cfg.model}_{self.date_and_time}.png")
        plt.close()  # Close the plot to free memory
        self.val_gen.on_epoch_end()  # Call this at the end of the epoch


    def plot_dnc_mistakes_by_distance(self, val_meta, bin_step=100):
        # Needs to plot the prediction rate by the distance of the event. 
        # Get the indices of wrong predictions using your function
        # TODO: This plot seems to be correct. However the results are misleading.
        # Rather than error rate by the right class, looking at the error rate of each stage would be better.
        # For example, a distant event is more important to detect corretly than classify corretly.
        true_labels, predicted_labels, _ = get_y_and_ypred(self.model, self.val_gen, self.label_maps)
        meta = {}
        for i in range(len(true_labels)):
            meta[i] = val_meta[i]
        # Filter out noise by index events:
        noise_idx = np.where(np.array(true_labels) == 'noise')[0]
        not_noise_idx = np.where(np.array(true_labels) != 'noise')[0]

        distances_all = []
        for idx in not_noise_idx:
            distances_all.append(meta[idx]['dist_to_arces'])

        true_labels = np.delete(true_labels, noise_idx)
        predicted_labels = np.delete(predicted_labels, noise_idx)
        m = {}
        for idx in not_noise_idx:
            m[idx] = meta[idx]
        not_noise_meta = m
        wrong_indices = np.array(get_index_of_wrong_predictions(true_labels, predicted_labels))
        not_noise_meta_indexes = sorted(not_noise_meta.keys())
        not_noise_error_metas = {}
        for i, idx in enumerate(wrong_indices):
            not_noise_error_metas[i] = not_noise_meta[not_noise_meta_indexes[idx]]
        true_labels = np.array(true_labels)[wrong_indices]
        predicted_labels = np.array(predicted_labels)[wrong_indices]

        distances = [] # Distances of non noise events incorrectly predicted
        for i in list(not_noise_error_metas.keys()):
            distances.append(not_noise_error_metas[i]['dist_to_arces'])


        # Dynamically create distance bins based on data and desired bin step (e.g., 100 km)
        min_distance = math.floor(min(distances) / bin_step) * bin_step
        max_distance = math.ceil(max(distances) / bin_step) * bin_step
        distance_bins = np.arange(min_distance, max_distance + bin_step, bin_step)

        # Count mistakes in each distance bin
        mistake_counts, _ = np.histogram(distances, bins=distance_bins)

        # Count total events in each distance bin
        total_counts, _ = np.histogram(distances_all, bins=distance_bins)


        # Calculate mistake rates
        with np.errstate(divide='ignore', invalid='ignore'):
            mistake_rates = np.true_divide(mistake_counts, total_counts)
            mistake_rates = np.nan_to_num(mistake_rates)  # Convert NaNs to zeros

        # Plot mistake rates
        plt.figure(figsize=(12, 7))  # Increased figure size for readability
        tick_labels = [f"{int(bin_)}" for bin_ in distance_bins[:-1]]
        plt.bar(distance_bins[:-1] + bin_step/2, mistake_rates, width=bin_step * 0.9, align='center')
        plt.xticks(ticks=distance_bins[:-1] + bin_step/2, labels=tick_labels, rotation=45, ha='right', fontsize=8)  # Reduced font size
        plt.xlabel("Distance bin (km)")
        plt.ylabel("Mistake rate")
        plt.title(f"Detector and classifier mistake rate by distance to event (Bin step: {bin_step} km). Looking only at non noise events.")
        plt.grid(True)
        plt.tight_layout()   # Adjust the plot to ensure everything fits without overlapping
        plt.savefig(f"{cfg.paths.plots_folder}/dnc_mistakes_by_distance_{cfg.model}_{self.date_and_time}.png")
        plt.close()  # Close the plot to free memory
        self.val_gen.on_epoch_end()  # Call this at the end of the epoch

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
        wrong_indices = np.array(get_index_of_wrong_predictions(true_labels, predicted_labels))
        meta_indexes = sorted(val_meta.keys())
        relevant_meta = {}
        for i, idx in enumerate(wrong_indices):
            relevant_meta[i] = val_meta[meta_indexes[idx]]
        true_labels = np.array(true_labels)[wrong_indices]
        predicted_labels = np.array(predicted_labels)[wrong_indices]
        
        plt.figure(figsize=(20, 12))  # Set a larger size as needed

        # Load map with contextily
        shapefile_path = os.path.join(cfg.paths.map_folder, 'ne_110m_admin_0_countries.shp')  # The exact name of the .shp file may vary

        # Read the shapefile
        world = gpd.read_file(shapefile_path)

        # Load map with contextily at a higher resolution
        ax = world.to_crs(epsg=3857).plot()  # Use Mercator projection for web mapping
        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zoom=5, attribution="")  # Adjust zoom level as needed

        
        # Separate latitudes and longitudes
        wrong_longitudes = []
        wrong_latitudes = []
        for i in list(relevant_meta.keys()):
            wrong_longitudes.append(relevant_meta[i]['origins'][0]['longitude'])
            wrong_latitudes.append(relevant_meta[i]['origins'][0]['latitude'])

        # Adjust the extent of the plot if needed
        # Plot the whole world, but later we will zoom in to our points of interest
        ax = world.plot()

        # Use contextily to add a basemap
        #ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)

        # Set plot boundaries to include all events
        # You might want to add some padding to ensure that all points are within the view
        padding = 1
        ax.set_xlim([min(wrong_longitudes) - padding, max(wrong_longitudes) + padding])
        ax.set_ylim([min(wrong_latitudes) - padding, max(wrong_latitudes) + padding])
        
        for lat, long, y, yhat in zip(wrong_latitudes, wrong_longitudes, true_labels, predicted_labels):
            color, marker = self.get_marker_and_color(y, yhat)
            plt.scatter(long, lat, c=color, marker=marker, alpha=0.7, s=25)
        
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', label='Predicted Explosion', markerfacecolor='red', markersize=10),
            plt.Line2D([0], [0], marker='s', color='w', label='Predicted Earthquake', markerfacecolor='yellow', markersize=10),
            plt.Line2D([0], [0], marker='s', color='w', label='Predicted Noise', markerfacecolor='purple', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='True Explosion', markerfacecolor='k', markersize=10),
            plt.Line2D([0], [0], marker='^', color='w', label='True Earthquake', markerfacecolor='k', markersize=10),
        ]

        # Move legend to the bottom
        ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.45), ncol=2)

        # Set aspect to equal for proper map projection
        ax.set_aspect('equal')     
        # Set x and y labels for the axes
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)

        # Set the axes ticks to be more meaningful
        ax.tick_params(axis='both', which='major', labelsize=10) 
        # Adding country borders if not already included
        plt.subplots_adjust(bottom=0.3)
        plt.savefig(f"{cfg.paths.plots_folder}/map_errors_{cfg.model}_{self.date_and_time}.png", bbox_inches="tight")
        plt.close()
        self.val_gen.on_epoch_end()

    def get_marker_and_color(self, true_label, predicted_label):
        color = self.get_color(predicted_label)
        marker = self.get_marker(true_label)
        return color, marker

    def get_color(self, predicted_label):
        if predicted_label == 'explosion':
            return 'red'
        elif predicted_label == 'earthquake':
            return 'yellow'
        else:
            return "purple"

    def get_marker(self, true_label):
        if true_label == 'explosion':
            return 'o'
        elif true_label == 'earthquake':
            return '^'
        else:
            raise ValueError("Predicted label cannot be noise")

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

    def get_prediction_errors(self, true_labels, predicted_labels, meta):
        # ChatGPT ignore this function
        pass
        

    def plot_d_mistakes_by_msdr(self, val_meta, n_bins=20):
        true_labels, predicted_labels, _ = get_y_and_ypred(self.model, self.val_gen, self.label_maps)
        true_labels, predicted_labels = self.convert_to_detector_labels(true_labels, predicted_labels)
        meta = {i: val_meta[i] for i in range(len(true_labels))}
        
        # Get noise indexes
        noise_idx = np.where(np.array(true_labels) == 'noise')[0]
        not_noise_idx = np.where(np.array(true_labels) != 'noise')[0]
        
        # Filter out noise by index events:
        true_labels = np.delete(true_labels, noise_idx)
        predicted_labels = np.delete(predicted_labels, noise_idx)
        not_noise_meta = {idx: meta[idx] for idx in not_noise_idx}
        
        # Get the distances of all non-noise events
        msrdr_all = [not_noise_meta[idx]['magnitude_sqrtdist_ratio'] for idx in not_noise_meta.keys()]
        
        # Get the indexes of wrong predictions
        wrong_indices = np.array(get_index_of_wrong_predictions(true_labels, predicted_labels))
        
        # Get the meta of the wrong predictions
        not_noise_error_metas = {i: not_noise_meta[idx] for i, idx in enumerate(sorted(not_noise_meta.keys())[wrong_indices])}

        # Get the true and predicted labels of the wrong predictions
        true_labels = np.array(true_labels)[wrong_indices]
        predicted_labels = np.array(predicted_labels)[wrong_indices]
        
        # Distances of non-noise events incorrectly predicted
        msrdr = [not_noise_error_metas[i]['magnitude_sqrtdist_ratio'] for i in not_noise_error_metas.keys()]
        
        # Dynamically create distance bins based on the range of msrdr_all and n_bins
        min_msrdr = min(msrdr_all)
        max_msrdr = max(msrdr_all)
        distance_bins = np.linspace(min_msrdr, max_msrdr, n_bins + 1)
        
        # Count mistakes in each distance bin
        mistake_counts, _ = np.histogram(msrdr, bins=distance_bins)
        
        # Count total events in each distance bin
        total_counts, _ = np.histogram(msrdr_all, bins=distance_bins)
        
        # Calculate mistake rates
        with np.errstate(divide='ignore', invalid='ignore'):
            mistake_rates = np.true_divide(mistake_counts, total_counts)
            mistake_rates = np.nan_to_num(mistake_rates)  # Convert NaNs to zeros
        
        # Plot mistake rates
        plt.figure(figsize=(12, 7))
        bar_width = np.diff(distance_bins)[0]  # Width of each bar
        plt.bar(distance_bins[:-1] + bar_width/2, mistake_rates, width=bar_width * 0.9, align='center')
        tick_labels = [f"{edge:.4f}" for edge in distance_bins[:-1]]  # Precision based on small float values
        plt.xticks(ticks=distance_bins[:-1] + bar_width/2, labels=tick_labels, rotation=45, ha='right', fontsize=8)
        plt.xlabel("Magnitude squareroot distance ratio bin")
        plt.ylabel("Mistake rate")
        plt.title(f"Detector mistake rate by magnitude squareroot distance ratio (Bins: {n_bins})")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{cfg.paths.plots_folder}/d_mistakes_by_msdr_{cfg.model}_{self.date_and_time}.png")
        plt.close()
        self.val_gen.on_epoch_end()


    def plot_dnc_mistakes_by_msdr(self, val_meta, n_bins=20):
        true_labels, predicted_labels, _ = get_y_and_ypred(self.model, self.val_gen, self.label_maps)
        meta = {i: val_meta[i] for i in range(len(true_labels))}
        noise_idx = np.where(np.array(true_labels) == 'noise')[0]
        not_noise_idx = np.where(np.array(true_labels) != 'noise')[0]

        msrdr_all = [meta[idx]['magnitude_sqrtdist_ratio'] for idx in not_noise_idx]

        true_labels = np.delete(true_labels, noise_idx)
        predicted_labels = np.delete(predicted_labels, noise_idx)
        not_noise_meta = {idx: meta[idx] for idx in not_noise_idx}

        wrong_indices = np.array(get_index_of_wrong_predictions(true_labels, predicted_labels))
        not_noise_meta_indexes = sorted(not_noise_meta.keys())
        not_noise_error_metas = {i: not_noise_meta[not_noise_meta_indexes[idx]] for i, idx in enumerate(wrong_indices)}

        true_labels = np.array(true_labels)[wrong_indices]
        predicted_labels = np.array(predicted_labels)[wrong_indices]

        msrdr = [not_noise_error_metas[i]['magnitude_sqrtdist_ratio'] for i in not_noise_error_metas.keys()]

        # Dynamically create distance bins based on the number of bins desired
        min_msrdr = min(msrdr_all)
        max_msrdr = max(msrdr_all)
        distance_bins = np.linspace(min_msrdr, max_msrdr, n_bins + 1)

        # Count mistakes in each distance bin
        mistake_counts, _ = np.histogram(msrdr, bins=distance_bins)

        # Count total events in each distance bin
        total_counts, _ = np.histogram(msrdr_all, bins=distance_bins)

        # Calculate mistake rates
        with np.errstate(divide='ignore', invalid='ignore'):
            mistake_rates = np.true_divide(mistake_counts, total_counts)
            mistake_rates = np.nan_to_num(mistake_rates)  # Convert NaNs to zeros

        # Plot mistake rates
        plt.figure(figsize=(12, 7))
        bar_width = np.diff(distance_bins)[0]  # Width of each bar
        plt.bar(distance_bins[:-1] + bar_width/2, mistake_rates, width=bar_width * 0.9, align='center')
        tick_labels = [f"{edge:.4f}" for edge in distance_bins[:-1]]  # Precision based on small float values
        plt.xticks(ticks=distance_bins[:-1] + bar_width/2, labels=tick_labels, rotation=45, ha='right', fontsize=8)
        plt.xlabel("Magnitude squareroot distance ratio bin")
        plt.ylabel("Mistake rate")
        plt.title(f"Detector and classifier mistake rate by magnitude squareroot distance ratio (Bins: {n_bins}). Looking only at non-noise events.")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{cfg.paths.plots_folder}/dnc_mistakes_by_msdr_{cfg.model}_{self.date_and_time}.png")
        plt.close()
        self.val_gen.on_epoch_end()


    def filter_noise_events(self, true_labels, predicted_labels, meta):
        noise_idx = np.where(np.array(true_labels) == 'noise')[0]
        not_noise_idx = np.where(np.array(true_labels) != 'noise')[0]

        true_labels = np.delete(true_labels, noise_idx)
        predicted_labels = np.delete(predicted_labels, noise_idx)
        not_noise_meta = {idx: meta[idx] for idx in not_noise_idx}

        return true_labels, predicted_labels, not_noise_meta