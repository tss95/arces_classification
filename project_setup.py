import logging.config
import colorlog
from omegaconf import OmegaConf
from types import SimpleNamespace
from typing import Any, Dict, Union, Tuple
import os
from config.logging_config import LOGGING_CONFIG


def dict_to_namespace(d: Dict[str, Any]) -> SimpleNamespace:
    """
    Converts a dictionary to a SimpleNamespace object for easier attribute access.

    Args:
        d (Dict[str, Any]): Dictionary to convert.

    Returns:
        SimpleNamespace: The converted dictionary as a namespace.
    """
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_namespace(v)
    return SimpleNamespace(**d)


def validate_window_cfg(cfg: SimpleNamespace) -> None:
    """Validate that canonical window settings are consistent across config sections."""
    data = getattr(cfg, "data", None)
    if data is None:
        return
    window_seconds = getattr(data, "window_seconds", None)
    if window_seconds is None:
        return
    sample_rate = getattr(data, "sample_rate", None)
    if sample_rate is None:
        raise ValueError("cfg.data.sample_rate is required when cfg.data.window_seconds is set.")
    expected_timesteps = int(round(float(window_seconds) * float(sample_rate)))

    default_length_seconds = getattr(data, "default_length_seconds", None)
    if default_length_seconds is not None and float(default_length_seconds) != float(window_seconds):
        raise ValueError(
            f"cfg.data.default_length_seconds ({default_length_seconds}) must match "
            f"cfg.data.window_seconds ({window_seconds})."
        )

    augment = getattr(cfg, "augment", None)
    if augment is not None and getattr(augment, "random_crop_kwargs", None) is not None:
        timesteps = augment.random_crop_kwargs.timesteps
        if int(timesteps) != expected_timesteps:
            raise ValueError(
                f"cfg.augment.random_crop_kwargs.timesteps ({timesteps}) must equal "
                f"cfg.data.window_seconds * cfg.data.sample_rate ({expected_timesteps})."
            )

    live = getattr(cfg, "live", None)
    if live is not None:
        live_length = getattr(live, "length", None)
        live_sample_rate = getattr(live, "sample_rate", None)
        if live_length is not None and float(live_length) != float(window_seconds):
            raise ValueError(
                f"cfg.live.length ({live_length}) must match cfg.data.window_seconds ({window_seconds})."
            )
        if live_sample_rate is not None and float(live_sample_rate) != float(sample_rate):
            raise ValueError(
                f"cfg.live.sample_rate ({live_sample_rate}) must match cfg.data.sample_rate ({sample_rate})."
            )

def add_data_paths(d):
    data_dir = os.getenv('DATA_DIR')
    if data_dir is None:
        raise EnvironmentError("The ROOT_DIR environment variable is not set. Please set this variable to the path of your root directory. Instructions can be found in the README.md file.")
    if 'data_paths' in d:
        for k, v in d['data_paths'].items():
            if isinstance(v, str):
                d['data_paths'][k] = os.path.join(data_dir,  v.lstrip('/'))
            elif isinstance(v, dict):
                add_data_paths(v)
    return d

def get_config_dir() -> str:
    project_dir = os.getenv('PROJECT_DIR')
    if project_dir is None:
        raise EnvironmentError("The PROJECT_DIR environment variable is not set. Please set this variable to the path of your root directory. Instructions can be found in the README.md file.")
    return os.path.join(project_dir, 'config')



def setup_config_and_logging():
    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger('ARCES')

    config_dir = get_config_dir()
    args = OmegaConf.load(f'{config_dir}/data_config.yaml')
    args_dict = OmegaConf.to_container(args, resolve=True)
    args_dict = add_data_paths(args_dict)  # Ensure root paths are added

    args = OmegaConf.create(args_dict)
    OmegaConf.set_struct(args, False)
    cfg = dict_to_namespace(args)

    model_args = OmegaConf.load(f"{config_dir}/models/{cfg.model_name}.yaml")
    model_args_dict = OmegaConf.to_container(model_args, resolve=True)
    model_args = OmegaConf.create(model_args_dict)
    OmegaConf.set_struct(model_args, False)
    model_cfg = dict_to_namespace(model_args)



    if cfg.data.debug:
        logger.warning("Debug mode is enabled. Verbose logging is enabled.")
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    validate_window_cfg(cfg)

    return logger, cfg, model_cfg
