from global_config import logger, cfg
import numpy as np
from src.Utils_torch import *
import torch
import h5py
import torch.multiprocessing as mp
import pickle

def create_and_populate_hdf5(events, dataset_name, cfg, chunk_size=128):
    n_samples = len(events.keys())
    sample_shape = events[list(events.keys())[0]]['X'].numpy().shape  # Adjusted for the correct key
    data_dtype = events[list(events.keys())[0]]['X'].numpy().dtype
    index_list = []  # To store [chunk_index, in-chunk_index, event_id, label, start_index, end_index]
    logger.info(f"Creating {dataset_name} dataset with {n_samples} samples, sample shape: {sample_shape}, data type: {data_dtype}")
    label_dict = {"noise": 0, "earthquake": 1, "explosion":2} 
    hdf5_path = f"{cfg.data_paths.loaded_path}/{dataset_name}_{'debug' if cfg.data.debug else 'full'}_data.h5"
    with h5py.File(hdf5_path, 'w') as f:
        dset_data = f.create_dataset('data', shape=(n_samples,) + sample_shape, dtype=data_dtype, chunks=(chunk_size,) + sample_shape)
        dset_labels = f.create_dataset('labels', shape=(n_samples,), dtype='i', chunks=(chunk_size,))
        dset_event_ids = f.create_dataset('event_ids', shape=(n_samples,), dtype=h5py.special_dtype(vlen=str), chunks=(chunk_size,))
        dset_years = f.create_dataset('years', shape=(n_samples,), dtype='int', chunks=(chunk_size,))
        dset_snr = f.create_dataset('snr', shape=(n_samples,), dtype='float', chunks=(chunk_size,))
        dset_index_start = f.create_dataset('index_start', shape=(n_samples,), dtype='int', chunks=(chunk_size,))
        dset_index_end = f.create_dataset('index_end', shape=(n_samples,), dtype='int', chunks=(chunk_size,))
        dset_windows = f.create_dataset('windows', shape=(n_samples, 2), dtype='float', chunks=(chunk_size, 2))
        
        # Populate datasets
        for i, (event_id, event_data) in enumerate(events.items()):
            chunk_index = i // chunk_size
            in_chunk_index = i % chunk_size
            dset_data[i] = event_data['X']
            dset_labels[i] = label_dict[event_data['Y']]  # Assuming you have a mapping in label_dict
            dset_event_ids[i] = event_id
            dset_years[i] = event_data['year']
            dset_snr[i] = event_data.get('snr', np.nan)  # Default to np.nan if 'snr' key is missing
            dset_index_start[i] = event_data['start_index'] if event_data['start_index'] is not None else -1
            dset_index_end[i] = event_data['end_index'] if event_data['end_index'] is not None else -1
            dset_windows[i] = event_data.get('window', (np.nan, np.nan))  # Assuming 'window' is a tuple; default to (np.nan, np.nan) if missing
            # Generate index list entry
            label = event_data['Y']  # Assuming you have a mapping in label_dict
            start_index = event_data['start_index']
            end_index = event_data['end_index']
            index_list.append([chunk_index, in_chunk_index, event_id, label, start_index, end_index])
    logger.info(f"Dataset {dataset_name} created and populated at {hdf5_path}")
    # Pickle the index list
    index_list_path = f"{cfg.data_paths.loaded_path}/{dataset_name}_{'debug' if cfg.data.debug else 'full'}_index_list.pkl"
    with open(index_list_path, 'wb') as file:
        pickle.dump(index_list, file)

        
if __name__ == "__main__":
    if cfg.data.debug:
        cfg.optimizer.max_epochs = 1
    multi_gpu = False
    #cfg.data.preloaded = False
    run_id = datetime.datetime.now().strftime("%y%m%d_%H%M%S") if not cfg.run_id else cfg.run_id
    mp.set_start_method('spawn', force=True)
    os.environ['WANDB_START_METHOD'] = 'thread'
    logger.info(f"Run ID: {run_id}, debug mode: {cfg.data.debug}, num_epochs: {cfg.optimizer.max_epochs}, multi_gpu: {multi_gpu}")
    cfg = prepare_folders_paths_cfg(run_id, cfg, make_folders=True)
    train_events, val_events, test_events, all_events, label_dict, class_weights, classifier_label_map, detector_label_map = preprocessing_pipeline(cfg)
    logger.info("Data loaded")
    dicts = {"label_dict": label_dict, "classifier_label_map": classifier_label_map, "detector_label_map": detector_label_map, "class_weights": class_weights}
    with open(f"{cfg.data_paths.loaded_path}/key_dicts.pkl", 'wb') as file:
        pickle.dump(dicts, file)
    
    
    chunk_size = 128
    sample_shape = train_events[list(train_events.keys())[0]]['X'].shape
    data_dtype =   train_events[list(train_events.keys())[0]]['X'].dtype
    logger.info(f"Sample shape: {sample_shape}, data type: {data_dtype}")


    # Assuming your preprocessing pipeline has already been run and populated train_events, val_events, and test_events
    datasets = {
        "train": train_events,
        "val": val_events
    }

    for name, events in datasets.items():
        create_and_populate_hdf5(events, name, cfg, chunk_size = chunk_size)
        