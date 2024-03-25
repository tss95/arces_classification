import torch
from torch.utils.data import Dataset

class BeamDataset(Dataset):
    def __init__(self, events_list, label_dict, data_dict):
        self.events_list = events_list
        self.data_dict = data_dict
        self.label_dict = label_dict
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def __len__(self):
        return len(self.events_list)

    def __getitem__(self, idx):
        year, event_id, label, _, _ = self.events_list[idx]
        sample = self.data_dict[year]["events"][event_id]

        # Scalar labels for the detector
        # "noise" is represented by 0 and "not_noise" by 1
        detector_label = 0 if label == "noise" else 1
        # Scalar labels for the classifier
        # "earthquake" is represented by 1 and "explosion" by 0
        # This assumes that "earthquake" is the positive class we are more interested in
        classifier_label = 0 if label == "earthquake" else 1
        processed_labels = {'detector': detector_label, 'classifier': classifier_label}
        return sample, processed_labels, event_id