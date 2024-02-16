import socket
import logging.config
import colorlog
from omegaconf import OmegaConf
from types import SimpleNamespace
from typing import Any, Dict, Union, Tuple
import os
import datetime
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

def add_root_paths(d):
    root_dir = os.getenv('ROOT_DIR')
    if root_dir is None:
        raise EnvironmentError("The ROOT_DIR environment variable is not set. Please set this variable to the path of your root directory. Instructions can be found in the README.md file.")
    if 'paths' in d:
        for k, v in d['paths'].items():
            if isinstance(v, str):
                d['paths'][k] = os.path.join(root_dir, v.lstrip('/'))
            elif isinstance(v, dict):
                update_paths(v)
    return d

def get_config_dir() -> str:
    """
    Determines the configuration base directory based on whether the code is running inside a Docker container.

    Returns:
        str: The path to the configuration directory.

    Raises:
        EnvironmentError: If the ROOT_DIR environment variable is not set.
    """
    root_dir = os.getenv('ROOT_DIR')
    if root_dir is None:
        raise EnvironmentError("The ROOT_DIR environment variable is not set. Please set this variable to the path of your root directory. Instructions can be found in the README.md file.")
    return os.path.join(root_dir, 'config')

def setup_config_and_logging():
    """
    Sets up configuration and logging for the application.

    Returns:
        Tuple[Logger, SimpleNamespace, SimpleNamespace, str]: Tuple containing the logger, 
        general configuration, model-specific configuration, and run ID.
    """
    run_id = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger('ARCES')
    #logger.setLevel(logging.DEBUG)

    config_dir = get_config_dir()
    args = OmegaConf.load(f'{config_dir}/data_config.yaml')
    args_dict = OmegaConf.to_container(args, resolve=True)

    # Add the roots defined in the ROOT_DIR environment variable to the paths.
    args_dict = add_root_paths(args_dict)

    args = OmegaConf.create(args_dict)
    OmegaConf.set_struct(args, False)
    cfg = dict_to_namespace(args)

    model_args = OmegaConf.load(f"{config_dir}/models/{cfg.model}.yaml")
    model_args_dict = OmegaConf.to_container(model_args, resolve=True)
    model_args = OmegaConf.create(model_args_dict)
    OmegaConf.set_struct(model_args, False)
    model_cfg = dict_to_namespace(model_args)

        # Update plots folder to include model-specific subdirectory
    model_plot_folder = os.path.join(cfg.paths.plots_folder, cfg.model, run_id)
    os.makedirs(model_plot_folder, exist_ok=True)
    cfg.paths.plots_folder = model_plot_folder

    # Create live_test_path if it does not exist
    live_test_path = os.path.join(config_dir, cfg.paths.live_test_path)
    os.makedirs(live_test_path, exist_ok=True)

    if cfg.predict:
        logger.warning("Script is now running in predict mode. Train data will not be loaded.")
        cfg.data.what_to_load=["val"]


    if cfg.data.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    
    return logger, cfg, model_cfg, run_id
