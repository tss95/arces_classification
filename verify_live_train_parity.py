import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency in some envs
    torch = None

try:
    import h5py
except ImportError:  # pragma: no cover - optional dependency in some envs
    h5py = None


def _ensure_env() -> None:
    """Set PROJECT_DIR/DATA_DIR defaults so config loading works in isolation."""
    repo_root = Path(__file__).resolve().parent
    if "PROJECT_DIR" not in os.environ:
        os.environ["PROJECT_DIR"] = str(repo_root)
    if "DATA_DIR" not in os.environ:
        # Falls back to the repo root; paths in config remain relative to DATA_DIR.
        os.environ["DATA_DIR"] = str(repo_root)


_ensure_env()

from global_config import cfg, logger  # noqa: E402

try:
    from src.Scaler_torch import Scaler  # noqa: E402
except ImportError as exc:  # pragma: no cover - optional dependency in some envs
    Scaler = None
    _scaler_import_error = exc

try:
    from src.Utils_torch import setup_transforms  # noqa: E402
except ImportError as exc:  # pragma: no cover - optional dependency in some envs
    setup_transforms = None
    _transforms_import_error = exc


def _describe_training(cfg) -> Dict[str, object]:
    window_timesteps = int(cfg.augment.random_crop_kwargs.timesteps)
    sample_rate = int(cfg.data.sample_rate)
    window_seconds = window_timesteps / sample_rate
    if setup_transforms is None:
        logger.info("setup_transforms unavailable (%s); skipping transform introspection.", _transforms_import_error)
        by_set = {}
    else:
        transforms = setup_transforms(cfg, add_scaling=False)
        by_set = {key: [type(t).__name__ for t in val] for key, val in transforms.items()}
    sample_transforms = ["RandomCropTransform"]
    scaling_cfg = {
        "scaler_type": cfg.scaling.scaler_type,
        "global_or_local": cfg.scaling.global_or_local,
        "per_channel": bool(cfg.scaling.per_channel),
    }
    return {
        "window_timesteps": window_timesteps,
        "window_seconds": window_seconds,
        "sample_rate": sample_rate,
        "transforms_by_set": by_set,
        "sample_transforms": sample_transforms,
        "scaling": scaling_cfg,
    }


def _describe_live(cfg) -> Dict[str, object]:
    window_timesteps = int(cfg.live.length * cfg.live.sample_rate)
    sample_rate = int(cfg.live.sample_rate)
    window_seconds = cfg.live.length
    filters = {
        "detrend": bool(cfg.filters.detrend),
        "taper": bool(cfg.filters.taper),
        "mode": cfg.filters.highpass_or_bandpass,
        "band": (
            cfg.filters.band_kwargs.min,
            cfg.filters.band_kwargs.max,
        ),
        "highpass_freq": cfg.filters.high_kwargs.high_freq,
    }
    scaling_cfg = {
        "scaler_type": cfg.scaling.scaler_type,
        "global_or_local": cfg.scaling.global_or_local,
        "per_channel": bool(cfg.scaling.per_channel),
    }
    return {
        "window_timesteps": window_timesteps,
        "window_seconds": window_seconds,
        "sample_rate": sample_rate,
        "filters": filters,
        "scaling": scaling_cfg,
    }


def _load_sample_shape(cfg) -> Optional[Tuple[int, ...]]:
    if h5py is None:
        logger.info("h5py not available; skipping HDF5 sample shape check.")
        return None
    split = "debug" if cfg.data.debug else "full"
    candidate = (
        Path(cfg.data_paths.loaded_path)
        / f"train_{split}_data.h5"
    )
    if not candidate.exists():
        logger.info("No training HDF5 found at %s; skipping sample shape check.", candidate)
        return None
    with h5py.File(candidate, "r") as handle:
        data_shape = handle["data"].shape  # (N, C, T)
    return tuple(data_shape[1:])  # (C, T)


def _check_checkpoint_scaler(cfg, scaler: Scaler) -> Optional[str]:
    if torch is None:
        logger.info("PyTorch not available; skipping checkpoint scaler check.")
        return None
    ckpt = Path(cfg.pretrained_model_name)
    if not ckpt.exists():
        logger.info("No checkpoint found at %s; skipping scaler state check.", ckpt)
        return None
    state = torch.load(ckpt, map_location="cpu")
    if scaler.requires_fit and "scaler_state" not in state:
        return (
            f"Checkpoint {ckpt} is missing scaler_state while scaler "
            f"requires fitting ({scaler.global_or_local}, per_channel={scaler.per_channel})."
        )
    return None


def _compare(training: Dict[str, object], live: Dict[str, object], sample_shape: Optional[Tuple[int, ...]]) -> List[str]:
    issues: List[str] = []
    if training["sample_rate"] != live["sample_rate"]:
        issues.append(f"Sample rate mismatch (train={training['sample_rate']}, live={live['sample_rate']}).")
    if training["window_timesteps"] != live["window_timesteps"]:
        issues.append(
            f"Window length mismatch (train={training['window_timesteps']} timesteps, "
            f"live={live['window_timesteps']} timesteps)."
        )
    if training["scaling"] != live["scaling"]:
        issues.append(f"Scaling config differs (train={training['scaling']}, live={live['scaling']}).")
    if sample_shape and sample_shape[-1] != training["window_timesteps"]:
        issues.append(
            f"Training HDF5 window length is {sample_shape[-1]} timesteps but "
            f"crop/window config expects {training['window_timesteps']}."
        )
    return issues


def main() -> int:
    training = _describe_training(cfg)
    live = _describe_live(cfg)
    scaler = Scaler(cfg) if Scaler is not None else None
    sample_shape = _load_sample_shape(cfg)
    checkpoint_issue = _check_checkpoint_scaler(cfg, scaler) if scaler is not None else None

    issues = _compare(training, live, sample_shape)
    if checkpoint_issue:
        issues.append(checkpoint_issue)

    logger.info("=== Training pipeline ===")
    logger.info(
        "sample_rate=%s, window=%s timesteps (%.2fs), sample_transforms=%s, batch_transforms=%s",
        training["sample_rate"],
        training["window_timesteps"],
        training["window_seconds"],
        training["sample_transforms"],
        training["transforms_by_set"],
    )
    logger.info("=== Live pipeline ===")
    logger.info(
        "sample_rate=%s, window=%s timesteps (%.2fs), filters=%s",
        live["sample_rate"],
        live["window_timesteps"],
        live["window_seconds"],
        live["filters"],
    )
    if sample_shape:
        logger.info("Training HDF5 sample shape (C, T): %s", sample_shape)
    if issues:
        logger.error("❌ Live/train parity check failed:")
        for issue in issues:
            logger.error(" - %s", issue)
        return 1

    logger.info("✅ Live/train parity check passed (core dimensions and scaling match).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
