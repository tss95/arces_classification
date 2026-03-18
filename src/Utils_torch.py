from collections import Counter
import datetime
import os
import pickle
import re
import socket
from types import SimpleNamespace

import numpy as np
import pandas as pd
from h5py import File
from torch import from_numpy

from global_config import logger
from src.Transforms import (
    AddGapTransform,
    AddNoiseTransform,
    BandpassFilterTransform,
    MinMaxPerChannelTransform,
    OnlineFilteringTransform,
    ScalingTransform,
    TaperTransform,
    ZeroChannelTransform,
)


EVENTCLASS_FILE_RE = re.compile(
    r"^eventclass(?:_(?P<variant>nofilt))?_(?P<year>\d{4})_(?P<kind>beams|beams_noise|labels_event_type)\.hdf5$"
)
SUPPORTED_EVENTCLASS_VARIANTS = {"filtered", "nofilt"}
SUPPORTED_EVENT_LABELS = {"earthquake", "explosion", "noise"}


def parse_eventclass_filename(filename):
    match = EVENTCLASS_FILE_RE.match(filename)
    if not match:
        return None
    variant = "nofilt" if match.group("variant") == "nofilt" else "filtered"
    year = int(match.group("year"))
    kind = match.group("kind")
    return variant, year, kind


def get_source_variant(cfg):
    variant = str(getattr(cfg.data, "source_variant", "filtered")).strip().lower()
    if variant not in SUPPORTED_EVENTCLASS_VARIANTS:
        raise ValueError(
            f"Unsupported cfg.data.source_variant='{variant}'. "
            f"Expected one of: {sorted(SUPPORTED_EVENTCLASS_VARIANTS)}"
        )
    return variant


def resolve_metadata_file_for_year(year, metadata_folder, source_variant):
    if source_variant == "nofilt":
        candidates = [
            f"eventclass_nofilt_{year}_arrivals_update.csv",
            f"metadata_{year}_arrivals_snrupdate.csv",
            f"eventclass_{year}_arrivals_snrupdate.csv",
            f"metadata_{year}_arrivals.csv",
        ]
    else:
        candidates = [
            f"metadata_{year}_arrivals_snrupdate.csv",
            f"eventclass_{year}_arrivals_snrupdate.csv",
            f"eventclass_{year}_arrivals_update.csv",
            f"metadata_{year}_arrivals.csv",
        ]

    for name in candidates:
        path = os.path.join(metadata_folder, name)
        if os.path.exists(path):
            return path

    fallback_pattern = re.compile(rf".*_{year}_.*arrivals.*\.csv$")
    fallback = sorted(
        os.path.join(metadata_folder, name)
        for name in os.listdir(metadata_folder)
        if fallback_pattern.match(name)
    )
    return fallback[0] if fallback else None


def normalize_event_label(raw_label, cfg):
    label = str(raw_label).strip().lower()
    labels_to_drop = {
        str(v).strip().lower() for v in getattr(cfg.data, "drop_event_labels", [])
    }
    if label in labels_to_drop:
        return None
    if "induced" in label:
        return "earthquake" if cfg.data.include_induced else None
    if label not in SUPPORTED_EVENT_LABELS:
        return None
    return label


def load_preprocessed_data(cfg):
    filename = (
        "preprocessed_data_full.pkl"
        if not cfg.data.debug
        else "preprocessed_data_debug.pkl"
    )
    filename = os.path.join(cfg.data_paths.loaded_path, filename)
    with open(filename, "rb") as file:
        data = pickle.load(file)

    train_events = data.get("train_events", None)
    val_events = data.get("val_events", None)
    test_events = data.get("test_events", None)
    all_events = data.get("all_events", None)
    label_dict = data.get("label_dict", None)
    class_weights = data.get("class_weights", None)
    classifier_label_map = data.get("classifier_label_map", None)
    detector_label_map = data.get("detector_label_map", None)

    logger.info(f"Loaded preprocessed data from {filename}")

    return (
        train_events,
        val_events,
        test_events,
        all_events,
        label_dict,
        class_weights,
        classifier_label_map,
        detector_label_map,
    )


def load_preprocessed_data_dict(cfg):
    filename = os.path.join(cfg.data_paths.loaded_path, "key_dicts.pkl")
    with open(filename, "rb") as file:
        data = pickle.load(file)
    return data


def setup_transforms(cfg, scaler_transform=None, add_scaling=True):
    transforms = {"train": [], "val": [], "test": []}
    aug = cfg.augment
    use_online_filtering = bool(getattr(cfg.filters, "use_filters", False))
    for transforms_key in transforms.keys():
        if use_online_filtering:
            # Crop is applied per-sample in BeamDatasetHDF5 before these batch transforms.
            transforms[transforms_key].append(OnlineFilteringTransform(cfg, split=transforms_key))
        else:
            # Legacy augmentation-only path when filters are explicitly disabled.
            if aug.bandpass:
                transforms[transforms_key].append(
                    BandpassFilterTransform(
                        aug.bandpass_kwargs.optional_min,
                        aug.bandpass_kwargs.optional_max,
                        cfg.data.sample_rate,
                        aug.bandpass_kwargs.prob if transforms_key == "train" else 0,
                        aug.bandpass_kwargs.default_min,
                        aug.bandpass_kwargs.default_max,
                    )
                )
            if aug.taper:
                transforms[transforms_key].append(TaperTransform(aug.taper_kwargs.alpha))

        if aug.zero_channel and transforms_key == "train":
            transforms[transforms_key].append(
                ZeroChannelTransform(aug.zero_channel_kwargs.prob)
            )
        if aug.add_gap and transforms_key == "train":
            transforms[transforms_key].append(
                AddGapTransform(aug.add_gap_kwargs.prob, aug.add_gap_kwargs.max_size)
            )
        if aug.add_noise and transforms_key == "train":
            transforms[transforms_key].append(
                AddNoiseTransform(aug.add_noise_kwargs.prob, cfg.scaling.per_channel)
            )
        if add_scaling:
            transform_to_use = scaler_transform or MinMaxPerChannelTransform(cfg)
            transforms[transforms_key].append(transform_to_use)
    return transforms


def prepare_folders_paths_cfg(
    run_id, cfg: SimpleNamespace, make_folders=True
) -> SimpleNamespace:
    """
    Prepare folders and paths for the seismic data classification project.

    This function creates the necessary folders for storing logs, models, and plots.
    It also sets up the output project paths.

    Args:
    cfg: The global configuration object containing all necessary paths and settings.

    """
    project_path = os.environ.get("PROJECT_DIR")
    cfg.pretrained_model_name = os.path.join(project_path, cfg.pretrained_model_name)
    output_path = os.path.join(project_path, cfg.project_paths.output_folder, cfg.model_name, run_id)
    cfg.project_paths.output_folder = output_path
    if make_folders:
        os.makedirs(output_path, exist_ok=True)
    for key in cfg.project_paths.output_folders:
        if cfg.project_paths.output_folders[key].endswith("/"):
            cfg.project_paths.output_folders[key] = os.path.join(
                output_path, cfg.project_paths.output_folders[key]
            )
            if make_folders:
                os.makedirs(cfg.project_paths.output_folders[key], exist_ok=True)
    for key in cfg.project_paths:
        if key not in ["output_folders", "output_folder"]:
            cfg.project_paths[key] = os.path.join(output_path, cfg.project_paths[key])
            if make_folders:
                os.makedirs(cfg.project_paths[key], exist_ok=True)

    return cfg


def get_pick_station(station):
    arces_group = {"ARCES", "ARA0", "ARE0", "ARA1"}
    if station in arces_group:
        return ["ARCES", "ARA0", "ARE0", "ARA1"]
    return [station]


def get_snr_and_distance(metadata_df, event_id, station):
    pick_station = get_pick_station(station)
    relevant_rows = metadata_df[metadata_df["event_id"] == event_id]
    relevant_rows = relevant_rows[relevant_rows["station"].isin(pick_station)]
    if relevant_rows.empty:
        return None, np.nan

    snr = relevant_rows["snr"].dropna()
    distance = relevant_rows["distance"].dropna()
    max_snr = float(snr.max()) if not snr.empty else None
    max_distance = float(distance.max()) if not distance.empty else np.nan
    return max_snr, max_distance


def convert_to_unix_time(time_str):
    return np.datetime64(time_str).astype("datetime64[s]").astype("int")


def convert_to_index(arrival_time_unix, window_start_unix, sampling_rate):
    return int((arrival_time_unix - window_start_unix) * sampling_rate)


def calculate_arrival_indices(arrival_dict, window, sampling_rate):
    if not arrival_dict:
        return arrival_dict
    window_start_unix = window[0]
    for arrival_id, details in arrival_dict.items():
        arrival_time_unix = convert_to_unix_time(details["time"])
        index = convert_to_index(arrival_time_unix, window_start_unix, sampling_rate)
        arrival_dict[arrival_id]["index"] = index
    return arrival_dict


def get_arrival_dict(metadata_df, arrival_ids, station):
    if not arrival_ids:
        return {}

    pick_station = get_pick_station(station)
    event_id = "_".join(arrival_ids[0].split("_")[0:3])
    filtered_df = metadata_df[metadata_df["event_id"] == event_id]
    filtered_df = filtered_df[filtered_df["station"].isin(pick_station)]
    if filtered_df.empty:
        return {}

    arrival_dict = {
        f"{event_id}_{row['arrival_id']}": {"label": row["label"], "time": row["time"]}
        for _, row in filtered_df.iterrows()
    }
    return arrival_dict


def get_relevant_p_and_s(arrival_dict):
    # If there are one or more P arrivals choose the earliest one.
    # If there are one or more S arrivals choose the latest one.
    p_arrivals = []
    s_arrivals = []
    for details in arrival_dict.values():
        phase_label = str(details.get("label", "")).strip().upper()
        if "index" not in details:
            continue
        if phase_label.startswith("P"):
            p_arrivals.append(details["index"])
        else:
            s_arrivals.append(details["index"])
    selected_p = min(p_arrivals) if p_arrivals else None
    selected_s = max(s_arrivals) if s_arrivals else None
    return selected_p, selected_s


def process_event_temporal_location(metadata_df, window, arrival_ids, distance, station, cfg):
    sampling_rate = cfg.data.sample_rate
    coda_factor = cfg.data.coda_factor
    arrival_dict = get_arrival_dict(metadata_df, arrival_ids, station)
    if not arrival_dict:
        return None, None

    arrival_dict = calculate_arrival_indices(arrival_dict, window, sampling_rate)
    p, s = get_relevant_p_and_s(arrival_dict)

    tp = p / sampling_rate if p is not None else None
    ts = s / sampling_rate if s is not None else None
    distance_term = (
        (float(distance) / 9.0) if distance is not None and np.isfinite(distance) else None
    )

    if tp is None and ts is None:
        return None, None
    if tp is None:
        if distance_term is None:
            return None, None
        tp = ts - distance_term
    if ts is None:
        if distance_term is None:
            return None, None
        ts = tp + distance_term
    if ts < tp:
        tp, ts = ts, tp

    end_of_event_estimate_index = (tp + coda_factor * (ts - tp)) * sampling_rate
    start_of_event_estimate_index = tp * sampling_rate
    return float(start_of_event_estimate_index), float(end_of_event_estimate_index)


def get_arrival_ids(metadata_df, event_id, station):
    pick_station = get_pick_station(station)
    arrival_rows = metadata_df[metadata_df["event_id"] == event_id]
    arrival_rows = arrival_rows[arrival_rows["station"].isin(pick_station)]
    arrival_ids = arrival_rows["arrival_id"]
    return [f"{event_id}_{arrival_id}" for arrival_id in arrival_ids]


def event_mapper(noise_beams_by_year, labels_event_types_by_year, metadata_by_year, cfg):
    events = {}
    event_durations = []
    dropped_label_counts = Counter()
    duplicate_event_ids = 0
    skipped_invalid_window_events = 0

    for year in sorted(labels_event_types_by_year.keys()):
        label_path = labels_event_types_by_year[year]
        metadata_path = metadata_by_year.get(year)
        if metadata_path is None:
            raise FileNotFoundError(f"Missing metadata file for year {year}")

        metadata_df = pd.read_csv(metadata_path)
        metadata_by_event_id = {
            event_id: group
            for event_id, group in metadata_df.groupby("event_id", sort=False)
        }
        with File(label_path, "r") as f:
            labels = f["labels"][:]
            ids = f["event_id"][:]
            windows = f["window"][:]
            stations = f["station"][:]

        for raw_label, raw_event_id, window, raw_station in zip(labels, ids, windows, stations):
            event_id = raw_event_id.decode("utf-8")
            station = raw_station.decode("utf-8")
            label = normalize_event_label(raw_label.decode("utf-8"), cfg)
            if label is None:
                dropped_label_counts[raw_label.decode("utf-8")] += 1
                continue

            event_metadata = metadata_by_event_id.get(event_id)
            if event_metadata is None:
                snr, distance = None, np.nan
                arrival_ids = []
                metadata_slice = metadata_df.iloc[0:0]
            else:
                pick_station = get_pick_station(station)
                metadata_slice = event_metadata[event_metadata["station"].isin(pick_station)]
                if metadata_slice.empty:
                    metadata_slice = event_metadata
                snr_values = metadata_slice["snr"].dropna() if "snr" in metadata_slice else pd.Series(dtype=float)
                distance_values = (
                    metadata_slice["distance"].dropna()
                    if "distance" in metadata_slice
                    else pd.Series(dtype=float)
                )
                snr = float(snr_values.max()) if not snr_values.empty else None
                distance = (
                    float(distance_values.max()) if not distance_values.empty else np.nan
                )
                if "arrival_id" in metadata_slice:
                    arrival_ids = [
                        f"{event_id}_{arrival_id}"
                        for arrival_id in metadata_slice["arrival_id"]
                    ]
                else:
                    arrival_ids = []
            start_index, end_index = process_event_temporal_location(
                metadata_slice, window, arrival_ids, distance, station, cfg
            )
            if (
                start_index is None
                or end_index is None
                or not np.isfinite(start_index)
                or not np.isfinite(end_index)
                or end_index <= start_index
            ):
                skipped_invalid_window_events += 1
                continue

            if event_id in events:
                duplicate_event_ids += 1
            events[event_id] = {
                "X": None,
                "Y": label,
                "snr": snr,
                "year": year,
                "start_index": start_index,
                "end_index": end_index,
                "window": window,
            }
            if (
                start_index is not None
                and end_index is not None
                and np.isfinite(start_index)
                and np.isfinite(end_index)
            ):
                event_durations.append((end_index - start_index) / cfg.data.sample_rate)

        with File(noise_beams_by_year[year], "r") as f:
            noise_windows = f["window"][:]
            for noise_index in range(len(noise_windows)):
                events[f"{year}_{noise_index}"] = {
                    "X": None,
                    "Y": "noise",
                    "snr": None,
                    "year": year,
                    "start_index": None,
                    "end_index": None,
                    "window": noise_windows[noise_index],
                }

    if event_durations:
        logger.info(
            "Average event duration: %s, median event duration: %s, min event duration: %s, max event duration: %s",
            np.mean(event_durations),
            np.median(event_durations),
            np.min(event_durations),
            np.max(event_durations),
        )
    else:
        logger.warning("No event duration estimates were computed from metadata.")
    if dropped_label_counts:
        logger.info("Dropped events by raw label: %s", dict(dropped_label_counts))
    if duplicate_event_ids:
        logger.warning(
            "Encountered %s duplicate event ids while mapping labels; latest row wins.",
            duplicate_event_ids,
        )
    if skipped_invalid_window_events:
        logger.warning(
            "Skipped %s events due to missing/invalid phase window estimates.",
            skipped_invalid_window_events,
        )
    return events


def get_file_by_year(files, year):
    if isinstance(files, dict):
        return files.get(year)
    for file in files:
        parsed = parse_eventclass_filename(os.path.basename(file))
        if parsed and parsed[1] == year:
            return file
    return None


def train_val_test_split(all_events, cfg):
    val_years = set(cfg.data.val_years)
    test_years = set(cfg.data.test_years) if cfg.data.load_testset else set()
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
                data.append(events[year][datatype][event_id]["X"])
                labels.append(events[year][datatype][event_id]["Y"])
                event_ids.append(event_id)
    data = np.array(data)
    return data, labels, event_ids


def get_file_names(cfg):
    data_path = cfg.data_paths.data_path
    metadata_folder = cfg.data_paths.metadata_folder
    source_variant = get_source_variant(cfg)
    min_year = int(getattr(cfg.data, "min_year", 0))
    max_year = int(getattr(cfg.data, "max_year", 9999))

    beams_by_year, beams_noise_by_year, labels_by_year = {}, {}, {}
    logger.info(
        "Data path: %s | source_variant: %s | year_range: [%s, %s]",
        data_path,
        source_variant,
        min_year,
        max_year,
    )
    for file in os.listdir(data_path):
        parsed = parse_eventclass_filename(file)
        if not parsed:
            continue
        file_variant, year, kind = parsed
        if file_variant != source_variant:
            continue
        if year < min_year or year > max_year:
            continue

        path = os.path.join(data_path, file)
        if kind == "beams":
            beams_by_year[year] = path
        elif kind == "beams_noise":
            beams_noise_by_year[year] = path
        elif kind == "labels_event_type":
            labels_by_year[year] = path

    available_years = sorted(
        set(beams_by_year) & set(beams_noise_by_year) & set(labels_by_year)
    )
    if not available_years:
        raise FileNotFoundError(
            f"No eventclass files found for source_variant='{source_variant}' in {data_path}"
        )

    metadata_by_year = {}
    missing_metadata_years = []
    for year in available_years:
        metadata_path = resolve_metadata_file_for_year(
            year, metadata_folder, source_variant
        )
        if metadata_path is None:
            missing_metadata_years.append(year)
        else:
            metadata_by_year[year] = metadata_path
    if missing_metadata_years:
        raise FileNotFoundError(
            f"Missing metadata CSV for years: {missing_metadata_years}. "
            f"Checked in {metadata_folder}"
        )

    if cfg.data.debug:
        years_to_load = set(cfg.data.val_years + [2000, 2001])
        if cfg.data.load_testset:
            years_to_load.update(cfg.data.test_years)
        years_to_load = sorted(year for year in years_to_load if year in available_years)
        if not years_to_load:
            raise ValueError(
                "Debug mode selected no overlapping years between requested and available datasets."
            )
    else:
        years_to_load = available_years

    beams_by_year = {year: beams_by_year[year] for year in years_to_load}
    beams_noise_by_year = {year: beams_noise_by_year[year] for year in years_to_load}
    labels_by_year = {year: labels_by_year[year] for year in years_to_load}
    metadata_by_year = {year: metadata_by_year[year] for year in years_to_load}
    logger.info("Using years: %s", years_to_load)
    return beams_noise_by_year, beams_by_year, labels_by_year, metadata_by_year


def calculate_class_weights(train_events):
    detector_counts = {"noise": 0, "not_noise": 0}
    classifier_counts = {"earthquake": 0, "explosion": 0}
    single_counts = {"noise": 0, "earthquake": 0, "explosion": 0}
    unique_labels = []

    for event_id in train_events.keys():
        label = train_events[event_id]["Y"]
        if label in single_counts:
            single_counts[label] += 1
        if label == "noise":
            detector_counts["noise"] += 1
        else:
            detector_counts["not_noise"] += 1
            if label == "explosion":
                classifier_counts["explosion"] += 1
            elif label == "earthquake":
                classifier_counts["earthquake"] += 1
        if label not in unique_labels:
            unique_labels.append(label)
    logger.info(f"Unique labels: {unique_labels}")

    total_detector = sum(detector_counts.values())
    total_classifier = sum(classifier_counts.values())
    total_single = sum(single_counts.values())

    detector_weights = {
        label: (total_detector / count if count > 0 else 0.0)
        for label, count in detector_counts.items()
    }
    classifier_weights = {
        label: (total_classifier / count if count > 0 else 0.0)
        for label, count in classifier_counts.items()
    }
    single_weights = {
        label: (total_single / count if count > 0 else 0.0)
        for label, count in single_counts.items()
    }
    logger.info(f"Detector weights: {detector_weights}")
    logger.info(f"Classifier weights: {classifier_weights}")
    logger.info(f"Single weights: {single_weights}")
    logger.info("==================================================================")

    return (
        {"detector": detector_weights, "classifier": classifier_weights, "single": single_weights},
        {"detector": detector_counts, "classifier": classifier_counts, "single": single_counts},
    )


def preprocessing_pipeline(cfg):
    if cfg.data.preloaded:
        return load_preprocessed_data(cfg)
    beams_noise, beams, labels_event_type, metadata = get_file_names(cfg)
    all_events = event_mapper(beams_noise, labels_event_type, metadata, cfg)
    train_events, val_events, test_events = train_val_test_split(all_events, cfg)
    all_labels = sorted({event["Y"] for event in all_events.values()})
    label_dict = {label: i for i, label in enumerate(all_labels)}
    class_weights, class_counts = calculate_class_weights(train_events)
    logger.info(f"Class counts: {class_counts}")
    logger.info("Loading data")
    train_events = load_data_set(train_events, beams, beams_noise)
    val_events = load_data_set(val_events, beams, beams_noise)
    test_events = load_data_set(test_events, beams, beams_noise)
    detector_label_map = {"noise": 0, "event": 1}
    classifier_label_map = {"earthquake": 0, "explosion": 1}
    single_label_map = {"noise": 0, "earthquake": 1, "explosion": 2}
    return (
        train_events,
        val_events,
        test_events,
        all_events,
        label_dict,
        class_weights,
        classifier_label_map,
        detector_label_map,
        single_label_map,
    )


def load_data_set(events, beams_by_year, beams_noise_by_year):
    years = sorted({events[event_id]["year"] for event_id in events})
    for year in years:
        beam_file = beams_by_year.get(year)
        noise_file = beams_noise_by_year.get(year)
        if beam_file is None or noise_file is None:
            raise FileNotFoundError(
                f"Missing beam/noise files for year {year}. beam={beam_file}, noise={noise_file}"
            )

        with File(beam_file, "r") as f:
            x_data = np.transpose(f["X"][:], (0, 2, 1))
            event_ids = f["event_id"][:]
            for raw_event_id, beam in zip(event_ids, x_data):
                event_id = raw_event_id.decode("utf-8")
                if event_id in events:
                    events[event_id]["X"] = from_numpy(beam)

        with File(noise_file, "r") as f:
            x_data = np.transpose(f["X"][:], (0, 2, 1))
            for i, noise_sample in enumerate(x_data):
                noise_id = f"{year}_{i}"
                if noise_id in events:
                    events[noise_id]["X"] = from_numpy(noise_sample)

    missing = [event_id for event_id, data in events.items() if data["X"] is None]
    if missing:
        raise ValueError(
            f"{len(missing)} events are missing waveform data after load. First few: {missing[:5]}"
        )
    return events
