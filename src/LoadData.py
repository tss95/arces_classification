# Your existing imports and setup here
from global_config import logger, cfg
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import os
import pickle
from obspy import Trace, Stream
from typing import Tuple, Dict, List, Optional

class LoadData:
    
    def __init__(self):
        """
        Initialize the LoadData class.

        This class is responsible for loading, filtering, and managing seismic data based on the configuration 
        specified in the global configuration (`cfg`). It handles different data sets (training, validation, test) 
        and applies necessary preprocessing steps such as filtering.

        Attributes:
            train_data (Optional[Tuple[np.ndarray, np.ndarray, Dict]]): Loaded and processed training data.
            val_data (Optional[Tuple[np.ndarray, np.ndarray, Dict]]): Loaded and processed validation data.
            test_data (Optional[Tuple[np.ndarray, np.ndarray, Dict]]): Loaded and processed test data.
        """
        
        if cfg.data.what_to_load == ["train", "val"]:
            self.train_data, self.val_data, _ = self.load_datasets()
            self.train_data = self.filter_data(self.train_data)
            self.val_data = self.filter_data(self.val_data)
        if cfg.data.what_to_load == ["train", "test"]:
            self.train_data, _, self.test_data= self.load_datasets()
            self.train_data = self.filter_data(self.train_data)
            self.test_data = self.filter_data(self.test_data)
        if cfg.data.what_to_load == ["train", "val", "test"]:
            self.train_data, self.val_data, self.test_data = self.load_datasets()
            self.train_data = self.filter_data(self.train_data)
            self.val_data = self.filter_data(self.val_data)
            self.test_data = self.filter_data(self.test_data)
        if cfg.data.what_to_load == ["val"]:
            self.train_data, self.val_data, self.test_data = self.load_datasets()
            self.val_data = self.filter_data(self.val_data)

        logger.info("Loaded data and applied filter. Ready for scaling.")
        #if self.train_data is not None:
        #    print("Train data first entry", self.train_data[0][0], self.train_data[1][0], self.train_data[2][0]) 

    
    def get_train_dataset(self) -> Optional[Tuple[np.ndarray, np.ndarray, Dict]]:
        """
        Retrieve the processed training dataset.

        Returns:
            Optional[Tuple[np.ndarray, np.ndarray, Dict]]: The training dataset containing traces, labels, and metadata.
        """
        return self.train_data
    
    def get_val_dataset(self) -> Optional[Tuple[np.ndarray, np.ndarray, Dict]]:
        """
        Retrieve the processed validation dataset.

        Returns:
            Optional[Tuple[np.ndarray, np.ndarray, Dict]]: The validation dataset containing traces, labels, and metadata.
        """
        return self.val_data
    
    def get_test_dataset(self) -> Optional[Tuple[np.ndarray, np.ndarray, Dict]]:
        """
        Retrieve the processed test dataset.

        Returns:
            Optional[Tuple[np.ndarray, np.ndarray, Dict]]: The test dataset containing traces, labels, and metadata.
        """
        return self.test_data
        
    def filter_data(self, dataset: Tuple[np.ndarray, np.ndarray, Dict]) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Apply filtering to the provided dataset based on the configuration settings.

        Args:
            dataset (Tuple[np.ndarray, np.ndarray, Dict]): The dataset to be filtered, consisting of traces, labels, and metadata.

        Returns:
            Tuple[np.ndarray, np.ndarray, Dict]: The filtered dataset.
        """
        if cfg.data.load_filtered:
            logger.warning("Pre filtered data is loaded. If this is an error -> check your data config file.")
            #r=np.random.randint(0, len(dataset[1]))
            #assert dataset[2][r]["is_filtered"] == True, f"Sample test failed. Data is not actually prefiltered."
            return dataset
        traces, labels, metadata = dataset
        for idx, trace in enumerate(tqdm(traces, desc="Filtering data")):
            metadata[idx]['is_filtered'] = False
            if cfg.filters.use_filters:
                traces[idx] = self.apply_filter(trace, metadata[idx])
                metadata[idx]['is_filtered'] = True
            metadata[idx]['is_scaled'] = False
            metadata[idx]['dataset_idx_verification'] = idx
            
        return (traces, labels, metadata)  # Optional, as the original dataset is modified
            
            
        
    def load_datasets(self) -> Tuple[Optional[Tuple[np.ndarray, np.ndarray, Dict]], 
                                     Optional[Tuple[np.ndarray, np.ndarray, Dict]], 
                                     Optional[Tuple[np.ndarray, np.ndarray, Dict]]]:
        """
        Load the datasets as specified in the configuration. This method can load train, validation, and test datasets.

        Returns:
            Tuple containing train, validation, and test datasets, respectively.
            Each element in the tuple is a dataset represented as (traces, labels, metadata), or None if not loaded.
        """
        what_to_load = cfg.data.what_to_load
        train_data, val_data, test_data = None, None, None
        if "train" in what_to_load:
            logger.info("Loading training dataset.")
            train_data = self.load_data("train")
            logger.info("Finished loading training dataset.")
        if "val" in what_to_load:
            logger.info("Loading validation dataset.")
            val_data = self.load_data("val")
            logger.info("Finished loading validation dataset.")
        if "test" in what_to_load:
            logger.info("Loading test dataset.")
            test_data = self.load_data("test")
            logger.info("Finished loading test dataset.")
        return train_data, val_data, test_data
    
    def load_data(self, data_description: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Load a specific dataset (train, val, or test) based on the provided description.

        Args:
            data_description (str): A string indicating which dataset to load ('train', 'val', or 'test').

        Returns:
            Tuple[np.ndarray, np.ndarray, Dict]: The loaded dataset consisting of traces, labels, and metadata.
        """
        traces, label, metadata = None, None, None
        trace_path = os.path.join(cfg.paths.loaded_path, f'{data_description}_traces{"_filtered" if cfg.data.load_filtered else ""}.npy')
        logger.warning("Loading: %s", str(trace_path))
        traces = np.load(trace_path, allow_pickle=True)
        label = np.load(os.path.join(cfg.paths.loaded_path, f'{data_description}_labels.npy'), allow_pickle=True)
        with open(os.path.join(cfg.paths.loaded_path, f'{data_description}_metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        metadata = self.change_key_to_index(metadata)
        if not cfg.data.include_induced:
            traces, label, metadata = self.remove_induced_events(traces, label, metadata)
        return traces, label, metadata
    
    def remove_induced_events(self, traces: np.ndarray, label: np.ndarray, metadata: Dict) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Remove induced or triggered events from the dataset.

        Args:
            traces (np.ndarray): Array of seismic traces.
            label (np.ndarray): Array of labels corresponding to the traces.
            metadata (Dict): Metadata associated with the traces.

        Returns:
            Tuple[np.ndarray, np.ndarray, Dict]: The dataset with induced events removed.
        """
        relevant_idx = np.where(np.array(metadata['labels']) != 'induced or triggered event')[0]
        metadata = {k: v for k, v in metadata.items() if k in relevant_idx}
        return traces[relevant_idx], label[relevant_idx], metadata

    def change_key_to_index(self, input_dict: Dict) -> Dict:
        """
        Convert the keys of the provided dictionary to sequential indices.

        Args:
            input_dict (Dict): The original dictionary with arbitrary keys.

        Returns:
            Dict: A dictionary with keys converted to sequential indices.
        """
        output_dict = {}
        for idx, key in enumerate(list(input_dict.keys())):
            output_dict[idx] = input_dict[key]
            output_dict[idx]['path'] = key
        return output_dict

    
    def filter_all_data(self, dataset: List[Tuple[np.ndarray, np.ndarray, Dict]]) -> List[Tuple[np.ndarray, np.ndarray, Dict]]:
        """
        Apply filtering to all entries in a list of datasets.

        Args:
            dataset (List[Tuple[np.ndarray, np.ndarray, Dict]]): A list of datasets to be filtered.

        Returns:
            List[Tuple[np.ndarray, np.ndarray, Dict]]: The list of filtered datasets.
        """
        if cfg.data.load_filtered:
            logger.warning("Data assumed to be prefiltered, skipping filtering process.")
            return dataset
        for idx, entry in enumerate(dataset):
            data, _, meta = entry
            dataset[idx][0] = self.apply_filter(data, meta)
        return dataset
        
        
    def apply_filter(self, trace: np.ndarray, meta: Dict) -> np.ndarray:
        """
        Apply the configured filter to a single trace.

        Args:
            trace (np.ndarray): The seismic trace data to be filtered.
            meta (Dict): Metadata associated with the trace.

        Returns:
            np.ndarray: The filtered trace.
        """
        filter_name = cfg.filters.highpass_or_bandpass
        station = meta['trace_stats']['station']
        channels = meta['trace_stats']['channels']
        sampl_rate = meta['trace_stats']['sampling_rate']
        starttime = meta['trace_stats']['starttime']
        trace_BHE = Trace(data=trace[0], header ={'station' : station,
                                                  'channel' : channels[0],
                                                  'sampling_rate' : sampl_rate,
                                                  'starttime' : starttime})
        trace_BHN = Trace(data=trace[1], header ={'station' : station,
                                                  'channel' : channels[1],
                                                  'sampling_rate' : sampl_rate,
                                                  'starttime' : starttime})
        trace_BHZ = Trace(data=trace[2], header ={'station' : station,
                                                  'channel' : channels[2],
                                                  'sampling_rate' : sampl_rate,
                                                  'starttime' : starttime})
        stream = Stream([trace_BHE, trace_BHN, trace_BHZ])
        if cfg.filters.detrend:
            stream.detrend('demean')
        if filter_name == "highpass":
            stream.filter('highpass', freq = cfg.filters.high_kwargs.high_freq)
        if filter_name == "bandpass":
            stream.filter('bandpass', freqmin=cfg.filters.band_kwargs.min, freqmax=cfg.filters.band_kwargs.max)
        return np.array(stream)

    def save_filtered_data(self, train_data: Tuple[np.ndarray, np.ndarray, Dict], 
                           val_data: Tuple[np.ndarray, np.ndarray, Dict], 
                           test_data: Tuple[np.ndarray, np.ndarray, Dict]):
        """
        Save the filtered train, validation, and test datasets to disk.

        Args:
            train_data (Tuple[np.ndarray, np.ndarray, Dict]): The training dataset.
            val_data (Tuple[np.ndarray, np.ndarray, Dict]): The validation dataset.
            test_data (Tuple[np.ndarray, np.ndarray, Dict]): The test dataset.
        """
        for dataset, name in [(train_data, 'train'), (val_data, 'val'), (test_data, 'test')]:
            #traces, labels, metadata = load_data(dataset["filename"], name)
            traces, _, _ = dataset
            np.save(os.path.join(cfg.paths.loaded_path, f'{name}_traces_filtered.npy'), traces)
            #np.save(os.path.join(output_folder, f'{name}_labels_filtered.npy'), labels)
            #with open(os.path.join(output_folder, f'{name}_metadata_filtered.pkl'), 'wb') as f:
            #    pickle.dump(metadata, f)
    

    
if __name__ == '__main__':
    print("Running main")
    #loadData = LoadData()
    #train_data = loadData.get_train_dataset()
    #val_data= loadData.get_val_dataset()
    #test_data = loadData.get_test_dataset()
    #loadData.save_filtered_data(train_data, val_data, test_data)



    
