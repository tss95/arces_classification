import matplotlib.pyplot as plt
import numpy as np
import torch
import datetime
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd


class Analysis:
    
    def __init__(self, model, dataloader, label_dict, classifier_label_map, detector_label_map, events, cfg):
        self.cfg = cfg
        self.events = events
        self.label_map = label_dict
        self.classifier_label_map = classifier_label_map
        self.detector_label_map = detector_label_map
        self.y_true, self.y_pred, self.ids, self.processed_Xs = self.get_y_and_y_pred(model, dataloader)
        self.y_true_string = self.get_string_labels_from_true(self.y_true)
        self.y_pred_string = self.get_string_labels_from_prob(self.y_pred)
        self.class_names = np.unique(self.y_true_string)
        
    def get_n_best_and_worst_detector(self, n):
        diff = np.abs(self.y_true["detector"] - self.y_pred["detector"])
        sorted_indices = np.argsort(diff)
        best_indices = sorted_indices[:n]
        worst_indices = sorted_indices[-n:]
        best_ids = [self.ids[i] for i in best_indices]
        worst_ids = [self.ids[i] for i in worst_indices]
        # I would also like to return the prediction probabilities for these events
        best_probs = [self.y_pred["detector"][i] for i in best_indices]
        worst_probs = [self.y_pred["detector"][i] for i in worst_indices]
        return best_ids, worst_ids, best_probs, worst_probs
        
    
    def get_n_best_and_worst_classifier(self, n):
        # Here I first need to mask out any noise events by finding their indices
        non_noise_indices = np.array([i for i, label in enumerate(self.y_true_string) if label != "noise"])
        # Then i need to remove these indexes from consideration in the classication set
        non_noise_classification_pred = self.y_pred["classifier"][non_noise_indices]
        non_noise_classification_true = self.y_true["classifier"][non_noise_indices]
        diff = np.abs(non_noise_classification_true - non_noise_classification_pred)
        sorted_indices = np.argsort(diff)
        best_indices = sorted_indices[:n]
        worst_indices = sorted_indices[-n:]
        best_ids = [self.ids[i] for i in best_indices]
        worst_ids = [self.ids[i] for i in worst_indices]
        # I would also like to return the prediction probabilities for these events
        best_probs = [non_noise_classification_pred[i] for i in best_indices]
        worst_probs = [non_noise_classification_pred[i] for i in worst_indices]
        return best_ids, worst_ids, best_probs, worst_probs
        
        
    
    def get_y_and_y_pred(self, model, dataloader):
        detector_true, classifier_true = [], []
        detector_logit, classifier_logit = [], []
        ids = []
        processed_Xs = {}
        for batch in dataloader:
            X, labels, id = batch
            y_pred = model(X)
            detector_true.extend(labels["detector"].cpu().numpy())
            detector_logit.extend(y_pred[0].cpu().numpy())
            classifier_true.extend(labels["classifier"].cpu().numpy())
            classifier_logit.extend(y_pred[1].cpu().numpy())
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
        detector_output = self.threshold_output(detector_output, self.cfg.detector.threshold)
        classifier_output = self.threshold_output(classifier_output, self.cfg.classifier.threshold)
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
            detector_label = self.detector_label_map[d]
            classifier_label = self.classifier_label_map[c]
            if detector_label == "noise":
                string_label.append(detector_label)
            else:
                string_label.append(classifier_label)
        return string_label
    
    
    def plot_waveform(self, ax, waveform, window, sample_rate, start_index, num_ticks=4):
        num_channels = waveform.shape[0]
        time = np.arange(waveform.shape[1]) / sample_rate
        time += window[0]  # adjust time to start at the start of the window

        for i in range(num_channels):
            ax[i].plot(time, waveform[i])
            ax[i].set_xlim(window)

            if i == num_channels - 1:
                ax[i].set_xlabel('Time (s)')
                ticks = np.linspace(window[0], window[1], num_ticks)
                ax[i].set_xticks(ticks)
                ax[i].set_xticklabels([datetime.datetime.fromtimestamp(tick).strftime('%Y-%m-%d %H:%M:%S') for tick in ticks])
            else:
                ax[i].set_xticklabels([])

        # Convert the start time to a datetime object and format it as a string
        start_time = datetime.datetime.fromtimestamp(window[0] + start_index / sample_rate)
        start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
        return ax, start_time_str
        
    def plot_n_best_and_worst_waveforms(self, best_ids, worst_ids, best_probs, worst_probs, label):
        # Combining IDs and probabilities for easier iteration and plotting
        items = [('Best', best_ids, best_probs), ('Worst', worst_ids, worst_probs)]
        num_subplots = len(best_ids) + len(worst_ids)
        
        # Adjusting rows and columns for the grid
        num_rows = (num_subplots + 3) // 4  # Ensures at least 4 plots per row, with adjustment for odd numbers
        num_cols = 4  # Assuming you want 4 columns max per row

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 5 * num_rows), squeeze=False)
        axs = axs.flatten()  # Flatten the axis array for easier indexing
        
        # Iterating over both best and worst
        index = 0  # Index for axs
        for category, ids, probs in items:
            for i, id_ in enumerate(ids):
                # Adjust this part according to how you want to use probabilities or any other parameters for plotting
                waveform = self.processed_Xs[id_]  # Placeholder for how you retrieve your waveform data
                window = self.events[id_]["window"]  # Placeholder for your window parameter
                sample_rate = self.cfg.data.sample_rate  # Placeholder for your sample_rate parameter
                start_index = self.events[id_]["start_index"]  # Placeholder for your start_index parameter

                # Plotting on the correct subplot axis
                ax = axs[index] if index < len(axs) else plt.subplot(num_rows, num_cols, index + 1)
                ax, start_time_str = self.plot_waveform(ax, waveform, window, sample_rate, start_index)
                title = f"Event start: {start_time_str}, event id {id_} \n True label {self.y_true_string[id_]}, Predicted label {self.y_pred_string[id_]}, \n Probs: {best_probs[i]}"
                ax.set_title(title)  # Example title to include probability
                index += 1

        # Adjust layout
        plt.tight_layout()
        plt.savefig(f"{self.cfg.project_paths.plots_folder}best_worst_prediction_for_{label}.png")
    
    def plot_best_and_worst(self, n):
        best_ids, worst_ids, best_probs, worst_probs = self.get_n_best_and_worst_classifier(n)
        self.plot_n_best_and_worst_waveforms(best_ids, worst_ids, best_probs, worst_probs, "classifier")
        best_ids, worst_ids, best_probs, worst_probs = self.get_n_best_and_worst_detector(n)
        self.plot_n_best_and_worst_waveforms(best_ids, worst_ids, best_probs, worst_probs, "detector")
        
        
    def plot_confusion_matrix(self, y_true, y_pred):
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Create a DataFrame for easier plotting
        cm_df = pd.DataFrame(cm, index=self.class_names, columns=self.class_names)

        # Create a heatmap
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(f"{self.cfg.project_paths.plots_folder}confusion_matrix.png")

        
        
        
        