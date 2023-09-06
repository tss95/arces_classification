import socket
import logging.config
import colorlog
from omegaconf import OmegaConf
from types import SimpleNamespace

def dict_to_namespace(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_namespace(v)
    return SimpleNamespace(**d)

def update_paths(d):
    for k, v in d.items():
        if isinstance(v, str) and v.startswith('/staff/tord/Workspace/arces_classification/'):
            d[k] = v.replace('/staff/tord/Workspace/arces_classification/', '/tf/')
        elif isinstance(v, dict):
            update_paths(v)

def get_config_dir():
    hostname = socket.gethostname()
    if hostname == 'saturn.norsar.no':
        return "/staff/tord/Workspace/arces_classification/config"
    else:
        return "/tf/config"

def setup_config_and_logging():
    # Your LOGGING_CONFIG presumably comes from another file. Import it here.
    from config.logging_config import LOGGING_CONFIG

    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger('ARCES')
    logger.setLevel(logging.DEBUG)

    config_dir = get_config_dir()
    args = OmegaConf.load(f'{config_dir}/data_config.yaml')
    args_dict = OmegaConf.to_container(args, resolve=True)

    # Update paths if not running on 'norsar.saturn.no'
    if socket.gethostname() != 'saturn.norsar.no':
        update_paths(args_dict)

    args = OmegaConf.create(args_dict)
    OmegaConf.set_struct(args, False)
    cfg = dict_to_namespace(args)

    model_args = OmegaConf.load(f"{config_dir}/models/{cfg.model}.yaml")
    model_args_dict = OmegaConf.to_container(model_args, resolve=True)
    model_args = OmegaConf.create(model_args_dict)
    OmegaConf.set_struct(model_args, False)
    model_cfg = dict_to_namespace(model_args)
    
    return logger, cfg, model_cfg
