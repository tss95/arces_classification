# Your existing imports and setup here
from global_config import logger, cfg
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import os
import pickle
from obspy import Trace, Stream

class LoadData:
    
    def __init__(self):
        if cfg.data.what_to_load == ["train", "val"]:
            self.train_data, self.val_data, _ = self.load_datasets()
            self.train_data = self.filter_data(self.train_data)
            self.val_data = self.filter_data(self.val_data)
        if cfg.data.what_to_load == ["train", "test"]:
            self.train_data, _, self.test_data= self.load_datasets()
            self.train_data = self.filter_data(self.train_data)
            self.test_data = self.filter_data(self.test_data)
        logger.info("Loaded data and applied filter. Ready for scaling.")
        if self.train_data is not None:
            print("Train data first entry", self.train_data[0][0], self.train_data[1][0], self.train_data[2][0]) 

    
    def get_train_dataset(self):
        return self.train_data
    
    def get_val_dataset(self):
        return self.val_data
    
    def get_test_dataset(self):
        return self.test_data
        
    def filter_data(self, dataset):
        for idx, label in enumerate(tqdm(dataset, desc="Filtering data")):
            trace, label, meta = dataset[0][idx], dataset[1][idx], dataset[2][idx]
            meta['is_filtered'] = False
            if cfg.filters.use_filters:
                trace = self.apply_filter(trace, meta)
                meta['is_filtered'] = True
            meta['is_scaled'] = False
            meta['dataset_idx'] = idx
            dataset[idx] = (trace, label, meta)
        return dataset  # Optional, as the original dataset is modified
            
            
        
    def load_datasets(self):
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
    
    def load_data(self, data_description):
        traces, label, metadata = None, None, None
        traces = np.load(os.path.join(cfg.paths.loaded_path, f'{data_description}_traces.npy'), allow_pickle=True)
        label = np.load(os.path.join(cfg.paths.loaded_path, f'{data_description}_labels.npy'), allow_pickle=True)
        with open(os.path.join(cfg.paths.loaded_path, f'{data_description}_metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        return traces, label, metadata
    
    def filter_all_data(self, dataset):
        for idx, entry in enumerate(dataset):
            data, _, meta = entry
            dataset[idx][0] = self.apply_filter(data, meta)
        return dataset
        
        
    def apply_filter(self, trace, meta):
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
    
if __name__ == '__main__':
    loadData = LoadData()
