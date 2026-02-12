from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, DistributedSampler, WeightedRandomSampler
#from torch.utils.data.distributed import DistributedSampler
from src.BeamDataset import BeamDatasetHDF5
import torch
from functools import partial
import pytorch_lightning as pl
import os
import pickle
import logging
from collections import Counter

logger = logging.getLogger("ARCES")

# Moved outside and added an extra `transforms` parameter
def batch_collate_fn(batch, transforms=None):
    batched_data = [item[0] for item in batch]
    label_keys = list(batch[0][1].keys()) if batch else []
    batched_labels = {key: [] for key in label_keys}
    batched_ids = [item[2] for item in batch]
    
    for _, labels, _ in batch:
        for key in label_keys:
            batched_labels[key].append(labels[key])

    batched_data = torch.stack(batched_data)
    for key in label_keys:
        batched_labels[key] = torch.stack(batched_labels[key])

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
        self.set_transforms_on_device = bool(
            getattr(getattr(cfg, "data", object()), "set_transforms_on_device", False)
        )

    def _sample_transforms_for_split(self, split):
        if isinstance(self.transforms_by_sample, dict):
            return self.transforms_by_sample.get(split, [])
        return self.transforms_by_sample

    def setup(self, stage=None):
        # Similar setup as before for train, val, and test datasets
        path = self.cfg.data_paths.loaded_path
        self.datasets = {}
        self.full_list = {}
        splits = ["train", "val"]
        if getattr(self.cfg.data, "load_testset", False):
            splits.append("test")
        for key in splits:
            data_name = f"{key}_{'full' if not self.cfg.data.debug else 'debug'}_data.h5"
            data_list = f"{key}_{'full' if not self.cfg.data.debug else 'debug'}_index_list.pkl"
            full_data = os.path.join(path, data_name)
            full_list = os.path.join(path, data_list)
            if not os.path.exists(full_data) or not os.path.exists(full_list):
                if key == "test":
                    logger.warning(
                        "Requested test split but missing files: %s / %s. Skipping test dataset.",
                        full_data,
                        full_list,
                    )
                    continue
                raise FileNotFoundError(
                    f"Missing required dataset files for split '{key}': {full_data}, {full_list}"
                )
            with open(full_list, 'rb') as file:
                self.full_list[key] = pickle.load(file)
            self.datasets[key] = BeamDatasetHDF5(
                full_data,
                self.full_list[key],
                transforms=self._sample_transforms_for_split(key),
            )

    def _train_sampler_mode(self):
        return str(getattr(self.cfg.data, "train_sampler_mode", "uniform")).strip().lower()

    def _build_weighted_train_sampler(self):
        records = self.full_list.get("train", [])
        if not records:
            return None
        label_weights_cfg = getattr(self.cfg.data, "train_label_sampling_weights", None)
        if label_weights_cfg is None:
            return None

        label_weights = {}
        if hasattr(label_weights_cfg, "items"):
            iterator = label_weights_cfg.items()
        else:
            iterator = vars(label_weights_cfg).items()
        for key, value in iterator:
            if str(key).startswith("_"):
                continue
            try:
                label_weights[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        if not label_weights:
            return None
        labels = [str(rec[3]) for rec in records]
        sample_weights = [float(label_weights.get(label, 1.0)) for label in labels]
        if not sample_weights or max(sample_weights) <= 0:
            return None

        replacement = bool(getattr(self.cfg.data, "train_sampler_replacement", True))
        factor = float(getattr(self.cfg.data, "train_sampler_num_samples_factor", 1.0))
        num_samples = max(1, int(round(len(sample_weights) * factor)))
        generator = torch.Generator()
        generator.manual_seed(int(getattr(self.cfg, "seed", 42)))

        counts = Counter(labels)
        total_weight = sum(sample_weights)
        expected = {}
        for label, count in counts.items():
            label_weight = float(label_weights.get(label, 1.0))
            expected[label] = (count * label_weight) / total_weight if total_weight > 0 else 0.0
        logger.info(
            "Using weighted train sampler with label_weights=%s expected_mix=%s num_samples=%d replacement=%s",
            label_weights,
            {k: round(v, 4) for k, v in expected.items()},
            num_samples,
            replacement,
        )

        return WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=num_samples,
            replacement=replacement,
            generator=generator,
        )
    
    def train_dataloader(self):
        # Check if using DDP or DDPSpawn (or any other distributed strategy PyTorch Lightning supports)
        weighted_mode = self._train_sampler_mode() == "weighted_by_label"
        if self.trainer and is_distributed_strategy(self.trainer):
            if weighted_mode:
                logger.warning(
                    "train_sampler_mode=weighted_by_label is not enabled under distributed strategy. "
                    "Falling back to DistributedSampler(shuffle=True)."
                )
            sampler = DistributedSampler(self.datasets["train"], shuffle=True)
            shuffle = False
        else:
            if weighted_mode:
                sampler = self._build_weighted_train_sampler()
                if sampler is None:
                    logger.warning(
                        "Requested weighted_by_label sampler but could not build it. "
                        "Falling back to uniform shuffling."
                    )
                    shuffle = True
                else:
                    shuffle = False
            else:
                sampler = None
                shuffle = True

        return DataLoader(self.datasets["train"],
                          batch_size=self.cfg.optimizer.batch_size,
                          shuffle=shuffle,
                          sampler=sampler,
                          num_workers=self.cfg.num_workers,
                          collate_fn=get_collate_fn_with_transforms(
                              transforms=None if self.set_transforms_on_device else self.transforms_by_set["train"]
                          ),
                          persistent_workers=bool(self.cfg.num_workers and self.cfg.num_workers > 0),
                          pin_memory=True
                        )
                          
    def val_dataloader(self):
        return DataLoader(self.datasets["val"], 
                          batch_size=self.cfg.optimizer.batch_size, 
                          shuffle=False,  # Typically, you don't shuffle validation data
                          num_workers=self.cfg.num_workers,
                          collate_fn=get_collate_fn_with_transforms(
                              transforms=None if self.set_transforms_on_device else self.transforms_by_set["val"]
                          ),
                          persistent_workers=bool(self.cfg.num_workers and self.cfg.num_workers > 0),
                          pin_memory=True)
    
    def test_dataloader(self):
        if "test" not in self.datasets:
            raise RuntimeError("Test dataloader requested but test split is not loaded.")
        return DataLoader(self.datasets["test"], 
                          batch_size=self.cfg.optimizer.batch_size, 
                          shuffle=False,  # Test data is also typically not shuffled
                          num_workers=self.cfg.num_workers,
                          collate_fn=get_collate_fn_with_transforms(
                              transforms=None if self.set_transforms_on_device else self.transforms_by_set["test"]
                          ),
                          persistent_workers=bool(self.cfg.num_workers and self.cfg.num_workers > 0),
                          pin_memory=True)

    def _active_split(self):
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return None
        if getattr(trainer, "training", False):
            return "train"
        if getattr(trainer, "testing", False):
            return "test"
        if getattr(trainer, "validating", False) or getattr(trainer, "sanity_checking", False):
            return "val"
        return None

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if not self.set_transforms_on_device:
            return batch
        split = self._active_split()
        if split is None:
            return batch
        transforms = self.transforms_by_set.get(split, [])
        if not transforms:
            return batch
        x, y, ids = batch
        for transform in transforms:
            x = transform(x)
        return x, y, ids
