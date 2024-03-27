from global_config import logger
import os
#from src.Utils import *
import pandas as pd
import numpy as np
import datetime
from h5py import File
from torch import from_numpy
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
            transforms[transforms_key].append(TaperTransform(aug.taper_kwargs.alpha))
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

def get_pick_station(station):
    if station == "ARCES" or station == "ARA0" or station == "ARE0" or station == "ARA1":
        pick_station =  ["ARCES", "ARA0", "ARE0", "ARA1"]
    return pick_station

def get_snr_and_distance(metadata_df, event_id, station):
    pick_station = get_pick_station(station)
    relevant_rows = metadata_df[metadata_df['event_id'] == event_id]
    relevant_rows = relevant_rows[relevant_rows['station'].isin(pick_station)]
    snr = relevant_rows['snr']
    distance = relevant_rows['distance']
    if snr.empty:
        return None, distance.max()
    
    return snr.max(), distance.max()

def convert_to_unix_time(time_str):
    return np.datetime64(time_str).astype('datetime64[s]').astype('int')

def convert_to_index(arrival_time_unix, window_start_unix, sampling_rate):
    return int((arrival_time_unix - window_start_unix) *     sampling_rate)

def calculate_arrival_indices(arrival_dict, window, sampling_rate):
    window_start_unix = window[0]
    # Update arrival_dict with indices
    for arrival_id, details in arrival_dict.items():
        arrival_time_str = details['time']
        arrival_time_unix = convert_to_unix_time(arrival_time_str)
        index = convert_to_index(arrival_time_unix, window_start_unix, sampling_rate)
        # Add index to the details
        arrival_dict[arrival_id]['index'] = index
    return arrival_dict

def get_arrival_dict(metadata_df, arrival_ids, station):
    pick_station = get_pick_station(station)
    event_id = "_".join(arrival_ids[0].split("_")[0:3])
    arrival_ids_short = [int(arrival_id.split("_")[-1]) for arrival_id in arrival_ids]
    filtered_df = metadata_df[metadata_df['event_id'] == event_id]
    filtered_df = filtered_df[filtered_df['station'].isin(pick_station)]
    
    arrival_dict = {
        f"{event_id}_{row['arrival_id']}": {
            "label": row['label'],
            "time": row['time']
        }
        for _, row in filtered_df.iterrows()
    }
    return arrival_dict

def get_relevant_p_and_s(arrival_dict):
    # If there are one or more P arrivals i choose the earliest one. If there are none, I return None
    # If there are one or more S arrivals i choose the latest one. If there are none, I return None
    # Both P and S could have different labels depending on the phase. I will thus be searching the first letter of the label
    p_arrivals = []
    s_arrivals = []
    for arrival_id, details in arrival_dict.items():
        p_arrivals.append(arrival_dict[arrival_id]["index"]) if details["label"][0] == "P" else s_arrivals.append(arrival_dict[arrival_id]["index"])
    selected_P = min(p_arrivals) if p_arrivals else None
    selected_S = max(s_arrivals) if s_arrivals else None
    return selected_P, selected_S

def process_event_temporal_location(metadata_df, window, arrival_ids, distance, station, cfg):
    sampling_rate = cfg.data.sample_rate
    coda_factor = cfg.data.coda_factor
    arrival_dict = get_arrival_dict(metadata_df, arrival_ids, station)
    arrival_dict = calculate_arrival_indices(arrival_dict, window, sampling_rate)
    p, s = get_relevant_p_and_s(arrival_dict) # TODO Handle multiple events in a single waveform
    # Convert to seconds:
    tp = p / sampling_rate if p else None
    ts = s / sampling_rate if s else None
    # If multiple p or s picks, take the first p and last s
    # Distance in kilometers
    # TODO: Calculate all of this with timesteps
    # Calculate estimated p or s if needed:
    if tp is None or ts is None:
        if not ts:
            ts = (distance/9) + tp
        if not tp:
            tp = (distance/9) - ts
    end_of_event_estimate_index = (tp + coda_factor * (ts-tp)) * sampling_rate
    start_of_event_estimate_index = tp * sampling_rate
    return start_of_event_estimate_index, end_of_event_estimate_index

def get_arrival_ids(metadata_df, event_id, station):
    pick_station = get_pick_station(station)
    arrival_rows = metadata_df[metadata_df['event_id'] == event_id]
    arrival_rows = arrival_rows[arrival_rows['station'].isin(pick_station)]
    arrival_ids = arrival_rows['arrival_id']
    arrival_ids = [event_id + "_" + str(arrival_id) for arrival_id in arrival_ids]
    return arrival_ids

def event_mapper(noise_beams, labels_event_types, metadata, cfg):
    events = {}
    event_durations = []
    for label_path in labels_event_types:
        year = int(label_path.split("/")[-1].split("_")[1])
        metadata_df = pd.read_csv(get_file_by_year(metadata, year))
        with File(label_path, 'r') as f:
            labels = f['labels'][:]
            ids = f['event_id'][:]
            windows = f['window'][:]
            stations = f['station'][:]

        
        for event_index, (label, event_id, window, station) in enumerate(zip(labels, ids, windows, stations)):
            event_id = event_id.decode('utf-8')
            label = label.decode('utf-8')
            station = station.decode('utf-8')
            if isinstance(event_id, list):
                raise ValueError("Event ID is a list", event_id)
            if cfg.data.include_induced and "induced" in label:
                label = "earthquake"
            elif not cfg.data.include_induced and "induced" in label:
                continue
            arrival_ids = get_arrival_ids(metadata_df, event_id, station)
            snr, distance = get_snr_and_distance(metadata_df, event_id, station)
            start_index, end_index = process_event_temporal_location(metadata_df, window, arrival_ids, distance, station, cfg)
            events[event_id] = {"X": None, "Y": label, "snr": snr, "year": year, "start_index": start_index, "end_index": end_index, "window": window}
            event_durations.append((end_index - start_index)/cfg.data.sample_rate)
        
        with File(get_file_by_year(noise_beams, year), 'r') as f:
            noise_windows = f['window'][:]
            for noise_index in range(len(noise_windows)):
                events[str(year) + "_" + str(noise_index)] = {"X": None, "Y": "noise", "snr": None, "year": year, "start_index": None, "end_index": None, "window": noise_windows[noise_index]}
    logger.info(f"Average event duration: {np.mean(event_durations)}, median event duration: {np.median(event_durations)}, min event duration: {np.min(event_durations)}, max event duration: {np.max(event_durations)}, max event duration index: {np.argmax(event_durations)}")
    return events

def get_file_by_year(files, year):
    for file in files:
        if str(year) in file.split("/")[-1]:
            #print(file)
            return file



def train_val_test_split(all_events, cfg):
    val_years = cfg.data.val_years
    test_years = cfg.data.test_years if cfg.data.load_testset else []
    train_events, val_events, test_events = {}, {}, {}
    for event_id in all_events.keys():
        if all_events[event_id]["year"] in val_years:
            val_events[event_id] = all_events[event_id]
        elif all_events[event_id]["year"] in test_years:
            test_events[event_id] = all_events[event_id]
        else:
            train_events[event_id] = all_events[event_id]
    return train_events, val_events, test_events

def make_dataset(events):
    data, labels, event_ids = [], [], []
    for year in events.keys():
        for datatype in events[year].keys():
            for event_id in events[year][datatype].keys():
                data.append(events[year][datatype][event_id]['X']) 
                labels.append(events[year][datatype][event_id]['Y'])
                event_ids.append(event_id)
    data = np.array(data)            
    return (data, labels, event_ids)

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
    if cfg.data.debug:
        years_to_load = cfg.data.val_years + [2010]
        if cfg.data.load_testset:
            years_to_load.extend(cfg.data.test_years)
        b = [get_file_by_year(beams, year) for year in years_to_load]
        l = [get_file_by_year(labels_event_type, year) for year in years_to_load]
        bn = [get_file_by_year(beams_noise, year) for year in years_to_load]
        beams = b
        labels_event_type = l
        beams_noise = bn
    return beams_noise, beams, labels_event_type, metadata

def calculate_class_weights(train_events):
    # Initialize counts
    detector_counts = {'noise': 0, 'not_noise': 0}
    classifier_counts = {'earthquake': 0, 'explosion': 0}

    # Count occurrences
    for event_id in train_events.keys():
        label = train_events[event_id]["Y"]
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
    all_labels = []
    for event_id in all_events.keys():
        all_labels = all_events[event_id]["Y"]
    all_labels = list(set(all_labels))
    label_dict = {label: i for i, label in enumerate(all_labels)}
    class_weights, class_counts = calculate_class_weights(train_events)
    logger.info(f"Class counts: {class_counts}")
    logger.info("Loading data")
    train_events = load_data_set(train_events, beams, beams_noise)
    val_events = load_data_set(val_events, beams, beams_noise)
    test_events = load_data_set(test_events, beams, beams_noise)
    detector_label_map = {"noise" : 0, "event" : 1}
    classifier_label_map = {"earthquake" : 0, "exlposion" : 1}
    return train_events, val_events, test_events, all_events, label_dict, class_weights, classifier_label_map, detector_label_map

def load_data_set(events, beams, beams_noise):
    years = []
    for event_id in events:
        year = events[event_id]["year"]
        if year not in years:
            years.append(year)
    
    for year in years:
        beam_file = get_file_by_year(beams, year)
        noise_file = get_file_by_year(beams_noise, year)
        with File(beam_file, 'r') as f:
            X = f['X'][:]
            X = np.transpose(X, (0, 2, 1))
            event_id = f['event_id'][:]
            for event_id, beam in zip(event_id, X):
                event_id = event_id.decode('utf-8')
                events[event_id]['X'] = from_numpy(beam)
        with File(noise_file, 'r') as f:
            X = f['X'][:]
            X = np.transpose(X, (0, 2, 1))
            for i, n in enumerate(X):
                events[str(year) + "_" + str(i)]['X'] = from_numpy(n)
                
    return events
    """
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
                event_id = event_id.decode('utf-8')
                data_dict[year]["events"][event_id] = {}
                data_dict[year]["events"][event_id]['X'] = beam
                data_dict[year]["events"][event_id]['y'] = event[2]
                data_dict[year]["events"][event_id

        with File(noise_file, 'r') as f:
            noise = f['X'][:]
            for i, n in enumerate(noise):
                data_dict[year]["noise"][year + "_" + i]['X'] = n
                data_dict[year]["noise"][year + "_" + i]['y'] = "noise"
    dataset = make_dataset(data_dict)
    return dataset """
        
        