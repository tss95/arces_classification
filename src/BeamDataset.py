import torch
from torch.utils.data import Dataset
import h5py

class BeamDataset(Dataset):
    def __init__(self, data_set, label_dict, transforms = None):
        self.data_set = data_set
        self.label_dict = label_dict
        self.transforms = transforms
        self.event_ids = list(data_set.keys())
        
    def __len__(self):
        return len(self.event_ids)

    def __getitem__(self, idx):
        event_id = self.event_ids[idx]
        sample, label, event_id, start_index, end_index = self.data_set[event_id]['X'], self.data_set[event_id]['Y'], event_id, self.data_set[event_id]['start_index'], self.data_set[event_id]['end_index']
        if self.transforms:
            for transform in self.transforms:
                sample = transform(sample, start_index, end_index)
        detector_label = torch.tensor([0 if label == "noise" else 1], dtype=torch.float32)
        classifier_label = torch.tensor([0 if label == "earthquake" else 1], dtype=torch.float32)
        processed_labels = {'detector': detector_label, 'classifier': classifier_label}
        return sample, processed_labels, event_id
    
    
class BeamDatasetHDF5(Dataset):
    
    def __init__(self, hdf5_path, index_list, chunk_size = 128, transforms=None):
        self.transforms = transforms
        # Load the index list
        self.index_list = index_list
        
        # Open the HDF5 file once and keep it open
        self.hdf5_path = hdf5_path
        self.chunk_size = chunk_size
    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        # Use the chunk index and in-chunk index to access the sample
        chunk_index, in_chunk_index, event_id, label_str, start_index, end_index = self.index_list[idx]
        # Assuming chunk size and in-chunk index are used to directly access data
        # If not directly applicable, adjust according to your data organization in HDF5
        with h5py.File(self.hdf5_path, 'r') as hdf5_file:
            # Calculate global index
            global_index = (chunk_index * self.chunk_size) + in_chunk_index
            sample = hdf5_file['data'][global_index]
            sample = torch.tensor(sample, dtype=torch.float32)
        if self.transforms:
            for transform in self.transforms:
                sample = transform(sample, start_index, end_index)
        detector_label = torch.tensor([0 if label_str == "noise" else 1], dtype=torch.float32)
        classifier_label = torch.tensor([0 if label_str == "earthquake" else 1], dtype=torch.float32)
      
        processed_labels = {'detector': detector_label, 'classifier': classifier_label}
        return sample, processed_labels, event_id
