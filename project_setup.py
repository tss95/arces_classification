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


def parse_optional_bool_env(name: str):
    raw = os.getenv(name)
    if raw is None:
        return None
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid value for {name}: {raw}. Expected true/false.")


def resolve_model_config_path(config_dir: str, cfg: SimpleNamespace) -> str:
    """
    Resolve the model config path, optionally overridden by MODEL_CONFIG env var.

    Accepted MODEL_CONFIG values:
    - "alexnet" or "alexnet.yaml" (resolved under config/models)
    - Absolute path to a yaml file
    """
    model_override = os.getenv("MODEL_CONFIG")
    if not model_override:
        return os.path.join(config_dir, "models", f"{cfg.model_name}.yaml")

    candidate = model_override.strip()
    if not candidate.endswith(".yaml"):
        candidate = f"{candidate}.yaml"
    if not os.path.isabs(candidate):
        candidate = os.path.join(config_dir, "models", candidate)
    if not os.path.exists(candidate):
        raise FileNotFoundError(f"MODEL_CONFIG override not found: {candidate}")

    cfg.model_name = os.path.splitext(os.path.basename(candidate))[0]
    return candidate



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

    predict_override = parse_optional_bool_env("PREDICT_MODE")
    if predict_override is not None:
        cfg.predict = bool(predict_override)

    model_config_path = resolve_model_config_path(config_dir, cfg)
    model_args = OmegaConf.load(model_config_path)
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
