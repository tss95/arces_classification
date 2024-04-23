from global_config import logger, cfg
import numpy as np
from src.Utils_torch import *
import torch
import pickle
import torch.multiprocessing as mp


def save_data_with_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
        
if __name__ == "__main__":
    if cfg.data.debug:
        cfg.optimizer.max_epochs = 1
    multi_gpu = False
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            cfg.optimizer.batch_size = int(cfg.optimizer.batch_size / num_gpus)
            logger.info(f"Detected {num_gpus} GPUs, setting batch size to {cfg.optimizer.batch_size}")
            multi_gpu = True
    cfg.data.preloaded = False
    run_id = datetime.datetime.now().strftime("%y%m%d_%H%M%S") if not cfg.run_id else cfg.run_id
    mp.set_start_method('spawn', force=True)
    os.environ['WANDB_START_METHOD'] = 'thread'
    logger.info(f"Run ID: {run_id}, debug mode: {cfg.data.debug}, num_epochs: {cfg.optimizer.max_epochs}, multi_gpu: {multi_gpu}")
    cfg = prepare_folders_paths_cfg(run_id, cfg, make_folders=True)
    train_events, val_events, test_events, all_events, label_dict, class_weights, classifier_label_map, detector_label_map = preprocessing_pipeline(cfg)
    logger.info("Data loaded")
    
    pickle_filename = "preprocessed_data_full.pkl" if not cfg.data.debug else f"preprocessed_data_debug.pkl"
    
    save_data_with_pickle({
    "train_events": train_events,
    "val_events": val_events,
    "test_events": test_events,
    "all_events": all_events,
    "label_dict": label_dict,
    "class_weights": class_weights,
    "classifier_label_map": classifier_label_map,
    "detector_label_map": detector_label_map
    }, os.path.join(cfg.data_paths.loaded_path, pickle_filename))
    
