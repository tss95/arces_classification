import matplotlib.pyplot as plt
from global_config import logger, cfg, model_cfg
import csv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, PrecisionRecallDisplay
import numpy as np

class Analysis:
    def __init__(self, model, val_data, val_labels_onehot, label_map, date_and_time):
        self.model = model
        self.val_data = val_data
        self.val_labels_onehot = val_labels_onehot
        self.label_map = label_map
        self.date_and_time = date_and_time

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