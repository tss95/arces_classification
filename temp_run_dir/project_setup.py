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

    return logger, cfg, model_cfg