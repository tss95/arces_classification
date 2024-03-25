from global_config import logger
import os
#from src.Utils import *
import pandas as pd
import numpy as np
import datetime
from h5py import File
from src.Transforms import BandpassFilterTransform, AddNoiseTransform, AddGapTransform, TaperTransform, ZeroChannelTransform

def setup_transforms(cfg):
    transforms = {"train": [], "val": [], "test": []}
    aug = cfg.augment
    for transforms_key in transforms.keys():
        if aug.bandpass:
            transforms[transforms_key].append(BandpassFilterTransform(aug.bandpass_kwargs.optional_min,
                                                                     aug.bandpass_kwargs.optional_max,
                                                                     cfg.data.sample_rate,
                                                                     aug.bandpass_kwargs.prob if transforms_key == "train" else 0,
                                                                     aug.bandpass_kwargs.default_min,
                                                                     aug.bandpass_kwargs.default_max))

        if aug.zero_channel:
            if transforms_key == "train":
                transforms[transforms_key].append(ZeroChannelTransform(aug.zero_channel_kwargs.prob))
        if aug.add_gap:
            if transforms_key == "train":
                transforms[transforms_key].append(AddGapTransform(aug.add_gap_kwargs.prob,
                                                                  aug.add_gap_kwargs.max_gap))
        if aug.add_noise:
            if transforms_key == "train":
                transforms[transforms_key].append(AddNoiseTransform(aug.add_noise_kwargs.prob,
                                                                   cfg.scaling.per_channel))
        if aug.taper:
            transforms[transforms_key].append(TaperTransform(aug.taper_kwargs.prob,
                                                            aug.taper_kwargs.alpha))
    return transforms

def prepare_folders_paths_cfg(run_id, cfg, make_folders=True):
    """
    Prepare folders and paths for the seismic data classification project.

    This function creates the necessary folders for storing logs, models, and plots. It also sets up the output project paths.

    Args:
    cfg: The global configuration object containing all necessary paths and settings.

    """
    project_path = os.environ.get('PROJECT_DIR')
    output_path = os.path.join(project_path, cfg.project_paths.output_folder, cfg.model_name, run_id)
    cfg.project_paths.output_folder = output_path
    if make_folders:
        os.makedirs(output_path, exist_ok=True)
    for key in cfg.project_paths.output_folders:
        if cfg.project_paths.output_folders[key].endswith('/'):
            cfg.project_paths.output_folders[key] = os.path.join(output_path, cfg.project_paths.output_folders[key])
            if make_folders:
                os.makedirs(cfg.project_paths.output_folders[key], exist_ok=True)
    for key in cfg.project_paths:
        if key not in ["output_folders", "output_folder"]:
            cfg.project_paths[key] = os.path.join(output_path, cfg.project_paths[key])
            if make_folders:
                os.makedirs(cfg.project_paths[key], exist_ok=True)
    return cfg
            

def get_snr(metadata_df, event_id):
    snr = metadata_df[metadata_df['event_id'] == event_id]['snr']
    if snr.empty:
        return None
    return snr.max()

def convert_to_unix_time(time_str):
    return np.datetime64(time_str).astype('datetime64[s]').astype('int')

def convert_to_index(arrival_time_unix, window_start_unix, sampling_rate):
    return int((arrival_time_unix - window_start_unix) / sampling_rate)

def calculate_arrival_indices(arrival_times_str, window, sampling_rate):
    window_start_unix = convert_to_unix_time(window[0])
    return [convert_to_index(convert_to_unix_time(t), window_start_unix, sampling_rate) for t in arrival_times_str]

def get_arrival_times(metadata_df, arrival_ids):
    return metadata_df[metadata_df['arrival_id'].isin(arrival_ids)]['time'].values

def process_event(metadata_df, window, arrival_ids, sampling_rate):
    arrival_times_str = get_arrival_times(metadata_df, arrival_ids)
    arrival_indices = calculate_arrival_indices(arrival_times_str, window, sampling_rate)
    # Add logic for P arrival, S-wave calculation, and coda factor here
    # end_index = ...
    return arrival_indices, end_index

def event_mapper(noise_beams, labels_event_types, metadata, cfg):
    events = []
    for label_path in labels_event_types:
        year = int(label_path.split("/")[-1].split("_")[1])
        metadata_df = pd.read_csv(get_file_by_year(metadata, year))
        with File(label_path, 'r') as f:
            labels = f['labels'][:]
            ids = f['event_id'][:]
            windows = f['window'][:]
            arrivals = f['arrivals'][:]
        
        for event_index, (label, event_id, window, arrival_id_bytes) in enumerate(zip(labels, ids, windows, arrivals)):
            event_id = event_id.decode('utf-8')
            label = label.decode('utf-8')
            if cfg.data.include_induced and "induced" in label:
                label = "earthquake"
            elif not cfg.data.include_induced and "induced" in label:
                continue
            snr = get_snr(metadata_df, event_id)
            arrival_ids = [int(id_str) for id_str in arrival_id_bytes.decode('utf-8').split(',')]
            arrival_indices, end_index = process_event(metadata_df, window, arrival_ids, cfg.live.sampling_rate)
            events.append((year, event_id, label, snr, event_index, arrival_indices, end_index))
        
        with File(get_file_by_year(noise_beams, year), 'r') as f:
            noise = f['station'][:]
            for noise_index in range(len(noise)):
                events.append((year, "not_applicable", "noise", None, noise_index, None, []))  # No SNR data for noise events
    return events

def get_file_by_year(files, year):
    for file in files:
        if str(year) in file.split("/")[-1]:
            #print(file)
            return file



def train_val_test_split(all_events, cfg):
    val_years = cfg.data.val_years
    test_years = cfg.data.test_years
    train_events, val_events, test_events = [], [], []
    for event in all_events:
        if event[0] in val_years:
            val_events.append(event)
        elif event[0] in test_years:
            test_events.append(event)
        else:
            train_events.append(event)
    return train_events, val_events, test_events

def get_file_names(cfg):
    beams_noise, beams, labels_event_type, metadata = [], [], [], []
    file_endings = ["_beams.hdf5", "_beams_noise.hdf5", "_labels_event_type.hdf5"]
    file_endings_metadata = ["snrupdate.csv"]
    for file in os.listdir(cfg.data_paths.data_path):
        path = cfg.data_paths.data_path + file
        if file.endswith(tuple(file_endings)):
            if file.endswith("_beams.hdf5"):
                beams.append(path)
            elif file.endswith("_beams_noise.hdf5"):
                beams_noise.append(path)
            elif file.endswith("_labels_event_type.hdf5"):
                labels_event_type.append(path)
    for file in os.listdir(cfg.data_paths.metadata_folder):
        path = cfg.data_paths.metadata_folder + file
        if file.endswith(tuple(file_endings_metadata)):
            metadata.append(path)
    return beams_noise, beams, labels_event_type, metadata

def calculate_class_weights(train_events):
    # Initialize counts
    detector_counts = {'noise': 0, 'not_noise': 0}
    classifier_counts = {'earthquake': 0, 'explosion': 0}

    # Count occurrences
    for _, _, label, _, _ in train_events:
        if label == 'noise':
            detector_counts['noise'] += 1
        else:
            detector_counts['not_noise'] += 1
        
        if label == 'earthquake':
            classifier_counts['earthquake'] += 1
        else:
            classifier_counts['explosion'] += 1

    # Calculate total samples for each task
    total_detector = sum(detector_counts.values())
    total_classifier = sum(classifier_counts.values())

    # Calculate class weights for each task
    detector_weights = {label: total_detector / count for label, count in detector_counts.items()}
    classifier_weights = {label: total_classifier / count for label, count in classifier_counts.items()}

    return {'detector': detector_weights, 'classifier': classifier_weights}, {'detector': detector_counts, 'classifier': classifier_counts}


def preprocessing_pipeline(cfg):
    beams_noise, beams, labels_event_type, metadata = get_file_names(cfg)
    all_events = event_mapper(beams_noise, labels_event_type, metadata, cfg)
    train_events, val_events, test_events = train_val_test_split(all_events, cfg)
    all_labels = list(set([event[2] for event in all_events]))
    label_dict = {label: i for i, label in enumerate(all_labels)}
    class_weights, class_counts = calculate_class_weights(train_events)
    logger.info(f"Class counts: {class_counts}")
    logger.info("Loading data")
    train_dict = load_data_dict(train_events, beams, beams_noise)
    val_dict = load_data_dict(val_events, beams, beams_noise)
    test_dict = load_data_dict(test_events, beams, beams_noise)
    detector_label_map = {"noise" : 0,"event" : 1}
    classifier_label_map = {"earthquake" : 0, "exlposion" : 1}
    return train_events, val_events, test_events, all_events, label_dict, class_weights, train_dict, val_dict, test_dict, classifier_label_map, detector_label_map

def load_data_dict(events, beams, beams_noise):
    # Creates a 
    data_dict = {}
    events_by_year = {}
    for event in events:
        year = event[0]
        if year not in data_dict.keys():
            data_dict[year] = {"events": {}, 
                               "noise": {}}
            events_by_year[year] = []
        events_by_year[year].append(event)
        
    for year in data_dict.keys():
        beam_file = get_file_by_year(beams, year)
        noise_file = get_file_by_year(beams_noise, year)
        with File(beam_file, 'r') as f:
            X = f['X'][:]
            X = np.transpose(X, (0, 2, 1))
            event_id = f['event_id'][:]
            for event_id, beam in zip(event_id, X):
                data_dict[year]["events"][event_id.decode('utf-8')] = beam
        with File(noise_file, 'r') as f:
            noise = f['X'][:]
            for i, n in enumerate(noise):
                data_dict[year]["noise"][i] = n
    return data_dict 
        
        