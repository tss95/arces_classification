import torch
from torch.utils.data import Dataset

class BeamDataset(Dataset):
    def __init__(self, data_set, label_dict, transforms = None):
        self.data_set = data_set
        self.label_dict = label_dict
        self.transforms = transforms
        self.event_ids = list(data_set.keys())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def __len__(self):
        return len(self.event_ids)

    def __getitem__(self, idx):
        event_id = self.event_ids[idx]
        sample, label, event_id, start_index, end_index = self.data_set[event_id]['X'], self.data_set[event_id]['Y'], event_id, self.data_set[event_id]['start_index'], self.data_set[event_id]['end_index']
        if self.transforms:
            for transform in self.transforms:
                sample = transform(sample, start_index, end_index)
        detector_label = torch.tensor(0 if label == "noise" else 1, dtype=torch.float32)
        classifier_label = torch.tensor(0 if label == "earthquake" else 1, dtype=torch.float32)
        processed_labels = {'detector': detector_label, 'classifier': classifier_label}
        return sample, processed_labels, event_id