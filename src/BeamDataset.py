import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from src.Augment import augment_torch
import torch

augment = augment_torch()

class BeamDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None, scaler = None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.scaler:
            sample = self.scaler.transform(sample)
        if self.transform:
            sample = self.transform(sample)

        return sample, label



