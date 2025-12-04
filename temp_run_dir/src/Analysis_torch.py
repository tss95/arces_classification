import matplotlib.pyplot as plt
import numpy as np
import torch
import datetime
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
import os
import h5py


class Analysis:
    
    def __init__(self, model, dataloader, label_dict, classifier_label_map, detector_label_map, events, cfg):
        self.cfg = cfg
        self.events = events
        self.label_map = label_dict
        self.classifier_label_map = classifier_label_map
        self.detector_label_map = detector_label_map
        self.reversed_classifier_label_map = {v: k for k, v in classifier_label_map.items()}
        self.reversed_detector_label_map = {v: k for k, v in detector_label_map.items()}
        self.prob_dist = None
        self.one_hot_y_true  = None
        self.pred_dict = None
        self.events_dict = self.convert_events_to_dict(events)
        self.events_dict = self.populate_windows(self.events_dict)
        self.y_true, self.y_pred, self.ids, self.processed_Xs = self.get_y_and_y_pred(model, dataloader)
        self.y_true_string = self.get_string_labels_from_true(self.y_true)
        self.y_pred_string = self.get_string_labels_from_prob(self.y_pred)
        self.class_names = np.unique(self.y_true_string)
        self.snrs_by_id = self.get_snrs_using_ids(self.ids, cfg)
        
    
    """
    def get_n_best_and_worst(self, n, head):
        # Here I first need to mask out any noise events by finding their indices
        if head == "classifier":
            non_noise_indices = np.array([i for i, label in enumerate(self.y_true_string) if label != "noise"])
            non_noise_indices = non_noise_indices.flatten()
            # Then i need to remove these indexes from consideration in the classication set
            pred = self.y_pred[head][non_noise_indices].flatten()
            true = np.array(self.y_true[head])[non_noise_indices]
        else:
            pred = self.y_pred[head]
            true = self.y_true[head]
        diff = np.abs(true - pred)
        sorted_indices = np.argsort(diff)
        sorted_indices = sorted_indices.flatten()
        best_indices = sorted_indices[:n]
        worst_indices = sorted_indices[-n:]
        best_ids = [self.ids[i] for i in best_indices]
        worst_ids = [self.ids[i] for i in worst_indices]
        # I would also like to return the prediction probabilities for these events
        best_probs = [pred[i] for i in best_indices]
        worst_probs = [pred[i] for i in worst_indices]
        return best_ids, worst_ids, best_probs, worst_probs
        """
        
    def get_n_best_and_worst(self, n, label):
        # Ensure prob_dist and one_hot_y_true are initialized
        self.prob_dist = self.convert_output_probabilties_to_unconditional(self.y_pred) if self.prob_dist is None else self.prob_dist
        self.one_hot_y_true = self.convert_to_one_hot(self.y_true) if self.one_hot_y_true is None else self.one_hot_y_true
        # Ensure pred_dict is initialized
        self.create_pred_dict(self.prob_dist, self.one_hot_y_true, self.ids) if self.pred_dict is None else self.pred_dict
        
        # Determine the index for the label of interest
        label_idx = {"noise": 0, "earthquake": 1, "explosion": 2}[label]
        
        # Find indices where the true label matches the label of interest
        relevant_indices = np.where(self.one_hot_y_true[:, label_idx] == 1)[0]
        
        # Calculate absolute differences for the label of interest, but only for relevant indices
        diff = np.abs(self.one_hot_y_true[relevant_indices, label_idx] - self.prob_dist[relevant_indices, label_idx])
        
        # Sort the differences and get indices for n best and worst predictions among the relevant ones
        sorted_indices_relevant = np.argsort(diff)
        best_indices_relevant = relevant_indices[sorted_indices_relevant[:n]]
        worst_indices_relevant = relevant_indices[sorted_indices_relevant[-n:]]
        
        # Extract IDs for the best and worst predictions
        best_ids = [self.ids[i] for i in best_indices_relevant]
        worst_ids = [self.ids[i] for i in worst_indices_relevant]
        
        # Also extract the prediction probabilities for these events
        best_probs = [self.prob_dist[i] for i in best_indices_relevant]
        worst_probs = [self.prob_dist[i] for i in worst_indices_relevant]
        
        return best_ids, worst_ids, best_probs, worst_probs
            
    def create_pred_dict(self, prob_dist, one_hot_y_true, ids):
        self.pred_dict = {}
        for i in range(len(ids)):
            self.pred_dict[self.ids[i]] = {'y_true': one_hot_y_true[i], 'y_pred': prob_dist[i]}
    
    def convert_to_one_hot(self, y_true):
        one_hot = []
        for i in range(len(y_true["detector"])):
            detector = y_true["detector"][i]
            classifier = y_true["classifier"][i]
            if detector == 1:  
                if classifier != 1:
                    one_hot.append([0, 1, 0])
                else:
                    one_hot.append([0, 0, 1])
            else:
                one_hot.append([1, 0, 0])
        return np.array(one_hot)
    
    def convert_output_probabilties_to_unconditional(self, y_prob):
        output_prob = []
        for i in range(len(y_prob["detector"])):
            p_noise = 1 - y_prob["detector"][i][0]
            p_not_noise = y_prob["detector"][i][0]
            p_earthquake_given_not_noise = 1 - y_prob["classifier"][i][0]
            p_explosion_given_not_noise = y_prob["classifier"][i][0]
            p_earthquake = p_not_noise * p_earthquake_given_not_noise
            p_explosion = p_not_noise * p_explosion_given_not_noise
            output_prob.append([p_noise, p_earthquake, p_explosion])
        return np.array(output_prob)
        
        
        
    def convert_events_to_dict(self, events):
        event_dict = {}
        
        for event in events:
            chunk_index = event[0]
            in_chunk_index = event[1]
            id = event[2]
            label = event[3]
            start_index = event[4]
            end_index = event[5]
            
            event_dict[id] = {
                "chunk_index": chunk_index,
                "in_chunk_index": in_chunk_index,
                "label": label,
                "start_index": start_index,
                "end_index": end_index
            }
        return event_dict
    
    def populate_windows(self, events_dict):
        full_path = f"{self.cfg.data_paths.loaded_path}/val_{'debug' if self.cfg.data.debug else 'full'}_data.h5"
        with h5py.File(full_path, 'r') as file:
            windows = np.array(file['windows'])
            ids = np.array(file['event_ids'])
            for i, id in enumerate(ids):
                id = id.decode('utf-8')
                events_dict[id]["window"] = windows[i]
        return events_dict
    
    def get_y_and_y_pred(self, model, dataloader):
        detector_true, classifier_true = [], []
        detector_logit, classifier_logit = [], []
        ids = []
        processed_Xs = {}
        for batch in dataloader:
            X, labels, id = batch
            y_pred = model(X)
            detector_true.extend(labels["detector"].cpu().numpy().flatten())
            detector_logit.extend(y_pred["detector"].detach().numpy())
            classifier_true.extend(labels["classifier"].cpu().numpy().flatten())
            classifier_logit.extend(y_pred["classifier"].detach().numpy())
            ids.extend(id)
            for x, i in zip(X, id):
                processed_Xs[i] = x
        # Sigmoid on each of the outputs
        detector_prob = torch.sigmoid(torch.tensor(detector_logit)).numpy()
        classifier_prob = torch.sigmoid(torch.tensor(classifier_logit)).numpy()
        pred = {"detector": detector_prob, "classifier": classifier_prob}
        true = {"detector": detector_true, "classifier": classifier_true}
        return true, pred, ids, processed_Xs
    
    
    def get_string_labels_from_prob(self, prob_output):
        detector_output = prob_output["detector"]
        classifier_output = prob_output["classifier"]
        detector_output = self.threshold_output(detector_output, self.cfg.data.model_threshold)
        classifier_output = self.threshold_output(classifier_output, self.cfg.data.model_threshold)
        int_output = {"detector": detector_output, "classifier": classifier_output}
        return self.translate_model_labels_to_string(int_output)
    
    def get_string_labels_from_true(self, true_output):
        return self.translate_model_labels_to_string(true_output)
        
        
    def threshold_output(self, output, threshold):
        return (output > threshold).astype(int)
        
        
    def translate_model_labels_to_string(self, int_output):
        detector_label = int_output["detector"]
        classifier_label = int_output["classifier"]              
        string_label = []
        for d, c in zip(detector_label, classifier_label):
            detector_label = self.reversed_detector_label_map[int(d)]
            classifier_label = self.reversed_classifier_label_map[int(c)]
            if detector_label == "noise":
                string_label.append(detector_label)
            else:
                string_label.append(classifier_label)
        return string_label
    
    
    def plot_waveform(self, axes, waveform, sample_rate, window, start_index):
        num_channels = waveform.shape[0]
        time = np.arange(waveform.shape[1]) / sample_rate
        # Calculate the absolute start time of the event
        if start_index != None:
            event_start_sec = window[0] + (start_index / sample_rate)
        else:
            event_start_sec = window[0]
        event_start_time = datetime.datetime.fromtimestamp(event_start_sec).strftime('%Y-%m-%d %H:%M:%S')

        for i, ax in enumerate(axes):
            ax.plot(time, waveform[i])
            ax.set_xlim(0, waveform.shape[1] / sample_rate)
            ax.set_xticklabels([])  # Remove x-axis tick labels

        return event_start_time


    def plot_n_best_and_worst_waveforms(self, best_ids, worst_ids, best_probs, worst_probs, label):
        items = [('Best', best_ids, best_probs), ('Worst', worst_ids, worst_probs)]
        num_channels = 3  # Number of channels per waveform
        num_subplots = (len(best_ids) + len(worst_ids)) * num_channels + 2  # +2 for separation titles
        
        fig, axs = plt.subplots(num_subplots, 1, figsize=(10, 3 * num_subplots), squeeze=False)
        axs = axs.ravel()

        index = 0
        for category, ids, probs in items:
            # Add a separation title for each category
            axs[index].axis('off')  # Turn off the axis
            axs[index].text(0.5, 0.5, f"{category} Predictions", ha='center', va='center', fontsize=20, weight='bold')
            index += 1
            
            for id_, prob in zip(ids, probs):
                waveform = self.processed_Xs[id_]
                window = self.events_dict[id_]["window"]
                sample_rate = self.cfg.data.sample_rate
                start_index = self.events_dict[id_]["start_index"]
                
                axes_for_waveform = axs[index:index+num_channels]
                event_start_time = self.plot_waveform(axes_for_waveform, waveform, sample_rate, window, start_index)
                true_label = label
                pred_label = self.pred_dict[id_]["y_pred"]
                if np.argmax(pred_label) == 0:
                    pred_label = "noise"
                    pred_prob = prob[0] 
                elif np.argmax(pred_label) == 1:
                    pred_label = "earthquake"
                    pred_prob = prob[1]
                else:
                    pred_label = "explosion"
                    pred_prob = prob[2]
                if true_label == "noise":
                    true_prob = prob[0]
                elif true_label == "earthquake":
                    true_prob = prob[1]
                else:
                    true_prob = prob[2]
                
                pred_prob = str(np.round(pred_prob, 3)*100)
                true_prob = str(np.round(true_prob, 3)*100)
                snr = self.snrs_by_id[id_]
                if snr == None:
                    snr = "N/A"
                title_text = f"ID: {id_}, Start: {event_start_time}\nTrue: {true_label}, Predicted: {pred_label}\n {true_label}_prob: {true_prob}%, {pred_label}_prob: {pred_prob}%, \n SNR: {snr}"
                axes_for_waveform[0].set_title(title_text, fontsize=15, pad=20)
                
                index += num_channels

        plt.tight_layout()
        plt.savefig(f"{self.cfg.project_paths.output_folders.plots_folder}/{label}_best_worst_predictions.png")


    
    def plot_best_and_worst(self, n):
        best_ids, worst_ids, best_probs, worst_probs = self.get_n_best_and_worst(n, "noise")
        self.plot_n_best_and_worst_waveforms(best_ids, worst_ids, best_probs, worst_probs, "noise")
        best_ids, worst_ids, best_probs, worst_probs = self.get_n_best_and_worst(n, "earthquake")
        self.plot_n_best_and_worst_waveforms(best_ids, worst_ids, best_probs, worst_probs, "earthquake")
        best_ids, worst_ids, best_probs, worst_probs = self.get_n_best_and_worst(n, "explosion")
        self.plot_n_best_and_worst_waveforms(best_ids, worst_ids, best_probs, worst_probs, "explosion")
        
        
        
    def plot_confusion_matrix(self, y_true, y_pred):
        # Compute confusion matrix
        unique_labels = sorted(np.unique(np.concatenate((y_pred, y_true))))
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
        
        # Calculate proportions for each cell in the confusion matrix
        proportions = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Use a colormap with good contrast for text readability
        cax = ax.matshow(proportions, cmap=plt.cm.Reds)
        
        # Add colorbar
        fig.colorbar(cax, label='Proportion of True Labels (%)')
        
        # Annotate cells with count and proportion, adjust text color based on cell color
        for (i, j), val in np.ndenumerate(cm):
            proportion = proportions[i, j]
            text_color = 'white' if proportion > 0.5 else 'black'
            ax.text(j, i, f"{val}\n({proportion:.2%})", ha='center', va='center', color=text_color, fontsize=8)
        
        # Set axis labels with a buffer space to ensure visibility
        ax.set_xticks(np.arange(len(unique_labels)))
        ax.set_xticklabels(unique_labels, rotation=45, ha='right')
        ax.set_yticks(np.arange(len(unique_labels)))
        ax.set_yticklabels(unique_labels)
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        ax.set_title('Confusion Matrix')

        # Adjust layout to make room for xtick labels
        plt.tight_layout(pad=1.0)

        # Save the plot to a specified path
        plt.savefig(f"{self.cfg.project_paths.output_folders.plots_folder}/confusion_matrix.png")
        plt.close()
        
    def analysis_package(self, n):
        self.plot_confusion_matrix(self.y_true_string, self.y_pred_string)
        self.plot_snr_distributions("detector")
        self.plot_snr_distributions("classifier")
        self.plot_best_and_worst(n)
        self.plot_snr_incorrect_predictions("detector")
        self.plot_snr_incorrect_predictions("classifier")

    def get_snrs(self, non_noise_indexes, cfg):
        loaded_path = cfg.data_paths.loaded_path
        file_name = f"val_{'full' if not self.cfg.data.debug else 'debug'}_data.h5"
        full_path = os.path.join(loaded_path, file_name)
        with h5py.File(full_path, 'r') as file:
            snrs = np.array(file['snr'])[non_noise_indexes]
        return snrs
    
    def get_snrs_using_ids(self, ids, cfg):
        loaded_path = cfg.data_paths.loaded_path
        file_name = f"val_{'full' if not self.cfg.data.debug else 'debug'}_data.h5"
        full_path = os.path.join(loaded_path, file_name)
        snrs_dict = {}
        with h5py.File(full_path, 'r') as file:
            snrs = np.array(file['snr'])
            in_file_ids = np.array(file['event_ids'])
            for in_file_id, snr in zip(in_file_ids, snrs):
                in_file_id = in_file_id.decode('utf-8')
                if in_file_id in ids:
                    snrs_dict[in_file_id] = snr
        return snrs_dict
            
            

            
    def plot_snr_distributions(self, detector_or_classifier):
        non_noise_indexes = np.array([i for i, label in enumerate(self.y_true_string) if label != "noise"]).flatten()
        y_true = np.array(self.y_true[detector_or_classifier])[non_noise_indexes]
        y_pred = (np.array(self.y_pred[detector_or_classifier])[non_noise_indexes]).flatten()
        snrs = self.get_snrs(non_noise_indexes, self.cfg)

        # Define logarithmic bins
        min_snr = np.min(snrs)
        max_snr = np.max(snrs)
        bins = np.logspace(np.log10(max(min_snr, 1)), np.log10(max_snr), num=30)  # Adjust num for finer/coarser bins

        # Plot the distribution of all SNRs
        plt.figure(figsize=(12, 8))
        y_pred = self.threshold_output(y_pred, self.cfg.data.model_threshold)
        # Identify correctly predicted events and plot their distribution
        correct_predictions = y_true == y_pred
        correct_snrs = snrs[correct_predictions]
        plt.hist(snrs, bins=bins, color='red', alpha=0.5, label='All SNRs', zorder=1, edgecolor = "black")
        plt.hist(correct_snrs, bins=bins, color='green', alpha=1, label='Correct Predictions', zorder=2, edgecolor="black")        
        # Logarithmic scale and labels
        plt.xscale('log')
        plt.xlabel('SNR (log scale)')
        plt.ylabel('Count')
        plt.title(f'Distribution of SNRs and Correct Predictions - {detector_or_classifier.capitalize()}')
        plt.legend()
        plt.savefig(f"{self.cfg.project_paths.output_folders.plots_folder}snr_distribution_{detector_or_classifier}.png")
        
    def plot_snr_incorrect_predictions(self, detector_or_classifier):
        non_noise_indexes = np.array([i for i, label in enumerate(self.y_true_string) if label != "noise"]).flatten()
        y_true = np.array(self.y_true[detector_or_classifier])[non_noise_indexes]
        y_pred = (np.array(self.y_pred[detector_or_classifier])[non_noise_indexes]).flatten()
        snrs = self.get_snrs(non_noise_indexes, self.cfg)

        # Apply threshold to predictions if needed
        y_pred = self.threshold_output(y_pred, self.cfg.data.model_threshold)

        # Define logarithmic bins
        min_snr = np.min(snrs)
        max_snr = np.max(snrs)
        bins = np.logspace(np.log10(max(min_snr, 1)), np.log10(max_snr), num=30)

        # Determine incorrect predictions
        incorrect_predictions = y_true != y_pred
        incorrect_snrs = snrs[incorrect_predictions]  # Extract SNRs for incorrect predictions

        # Plot the distribution of incorrect SNRs
        plt.figure(figsize=(12, 8))
        plt.hist(incorrect_snrs, bins=bins, color='blue', alpha=0.7, edgecolor='black', label='Incorrect Predictions', zorder=2)
        
        # Logarithmic scale and labels
        plt.xscale('log')
        plt.xlabel('SNR (log scale)')
        plt.ylabel('Count')
        plt.title(f'Distribution of SNRs for Incorrect Predictions - {detector_or_classifier.capitalize()}')
        plt.legend()
        plt.savefig(f"{self.cfg.project_paths.output_folders.plots_folder}snr_incorrect_{detector_or_classifier}.png")
        
        

            

        
        
        
        