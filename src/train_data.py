from typing import Dict, List, Tuple

from src.BeamModule import BeamModule
from src.Scaler_torch import Scaler
from src.Transforms import LiveStyleCenterCropTransform, RandomCropTransform, ScalingTransform
from src.Utils_torch import load_preprocessed_data_dict, setup_transforms


def build_sample_transforms(cfg) -> Dict[str, List]:
    train_sample_transform = RandomCropTransform(cfg)
    val_sample_mode = str(getattr(cfg.data, "validation_sample_mode", "random_crop")).lower()
    if val_sample_mode == "live_center":
        val_sample_transform = LiveStyleCenterCropTransform(cfg)
    elif val_sample_mode == "random_crop":
        val_sample_transform = RandomCropTransform(cfg)
    else:
        raise ValueError(
            f"Unsupported data.validation_sample_mode='{val_sample_mode}'. "
            "Use one of: ['random_crop', 'live_center']."
        )

    transforms_by_sample = {
        "train": [train_sample_transform],
        "val": [val_sample_transform],
    }
    if bool(getattr(cfg.data, "load_testset", False)):
        transforms_by_sample["test"] = [val_sample_transform]
    return transforms_by_sample


def build_data_module_and_scaler(cfg):
    """
    Build datamodule/scaler bundle for training.

    Returns:
        tuple: (key_dicts, data_module, scaler, transforms_by_sample, transforms_by_set)
    """
    key_dicts = load_preprocessed_data_dict(cfg)

    transforms_by_sample = build_sample_transforms(cfg)
    transforms_by_set = setup_transforms(cfg, add_scaling=False)
    scaler = Scaler(cfg)

    data_module = BeamModule(transforms_by_sample, transforms_by_set, cfg)
    data_module.setup()

    if scaler.requires_fit:
        fit_transforms = (
            transforms_by_set["train"]
            if getattr(cfg.data, "set_transforms_on_device", False)
            else None
        )
        scaler.fit_loader(data_module.train_dataloader(), batch_transforms=fit_transforms)

    scaling_transform = ScalingTransform(scaler)
    for split in transforms_by_set:
        transforms_by_set[split].append(scaling_transform)

    return key_dicts, data_module, scaler, transforms_by_sample, transforms_by_set
