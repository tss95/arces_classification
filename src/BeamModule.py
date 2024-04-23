from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, DistributedSampler
#from torch.utils.data.distributed import DistributedSampler
from src.BeamDataset import BeamDatasetHDF5
import torch
from functools import partial
import pytorch_lightning as pl
import os
import pickle

# Moved outside and added an extra `transforms` parameter
def batch_collate_fn(batch, transforms=None):
    batched_data = [item[0] for item in batch]
    batched_labels = {'detector': [], 'classifier': []}
    batched_ids = [item[2] for item in batch]
    
    for _, labels, _ in batch:
        batched_labels['detector'].append(labels['detector'])
        batched_labels['classifier'].append(labels['classifier'])

    batched_data = torch.stack(batched_data)
    batched_labels['detector'] = torch.stack(batched_labels['detector'])
    batched_labels['classifier'] = torch.stack(batched_labels['classifier'])

    if transforms:
        for transform in transforms:
            batched_data = transform(batched_data)

    return batched_data, batched_labels, batched_ids

def get_collate_fn_with_transforms(transforms=None):
    # Using `partial` to bind the transforms to the collate function
    return partial(batch_collate_fn, transforms=transforms)

def is_distributed_strategy(trainer):
    # This function aims to check if the trainer is configured for distributed training.
    # Adjust the logic based on available attributes in your PyTorch Lightning version.
    return trainer.num_devices > 1 or trainer.num_nodes > 1 or isinstance(trainer.strategy, pl.strategies.DDPStrategy)


class BeamModule(LightningDataModule):
    def __init__(self, transforms_by_sample, transforms_by_set, cfg):
        super().__init__()
        self.transforms_by_sample = transforms_by_sample
        self.transforms_by_set = transforms_by_set
        self.cfg = cfg

    def setup(self, stage=None):
        # Similar setup as before for train, val, and test datasets
        path = self.cfg.data_paths.loaded_path
        self.datasets = {}
        self.full_list = {}
        for key in ["train", "val"]:
            data_name = f"{key}_{'full' if not self.cfg.data.debug else 'debug'}_data.h5"
            data_list = f"{key}_{'full' if not self.cfg.data.debug else 'debug'}_index_list.pkl"
            full_data = os.path.join(path, data_name)
            full_list = os.path.join(path, data_list)
            with open(full_list, 'rb') as file:
                self.full_list[key] = pickle.load(file)
            self.datasets[key] = BeamDatasetHDF5(full_data, self.full_list[key], transforms=self.transforms_by_sample)
    
    def train_dataloader(self):
        # Check if using DDP or DDPSpawn (or any other distributed strategy PyTorch Lightning supports)
        if self.trainer and is_distributed_strategy(self.trainer):
            sampler = DistributedSampler(self.datasets["train"], shuffle=True)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        return DataLoader(self.datasets["train"],
                          batch_size=self.cfg.optimizer.batch_size,
                          shuffle=shuffle,
                          sampler=sampler,
                          num_workers=self.cfg.num_workers,
                          collate_fn=get_collate_fn_with_transforms(transforms=self.transforms_by_set["train"]),
                          persistent_workers=True,
                          pin_memory=True
                        )
                          
    def val_dataloader(self):
        return DataLoader(self.datasets["val"], 
                          batch_size=self.cfg.optimizer.batch_size, 
                          shuffle=False,  # Typically, you don't shuffle validation data
                          num_workers=self.cfg.num_workers,
                          collate_fn=get_collate_fn_with_transforms(transforms=self.transforms_by_set["val"]),
                          persistent_workers=True,
                          pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.datasets["test"], 
                          batch_size=self.cfg.optimizer.batch_size, 
                          shuffle=False,  # Test data is also typically not shuffled
                          num_workers=self.cfg.num_workers,
                          collate_fn=get_collate_fn_with_transforms(transforms=self.transforms_by_set["test"]),
                          persistent_workers=True,
                          pin_memory=True)