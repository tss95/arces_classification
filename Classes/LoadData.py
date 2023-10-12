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
        print(cfg)
        print("CFG USE_FILTERED::", cfg.data.load_filtered)
        
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
        logger.info("Loaded data and applied filter. Ready for scaling.")
        #if self.train_data is not None:
        #    print("Train data first entry", self.train_data[0][0], self.train_data[1][0], self.train_data[2][0]) 

    
    def get_train_dataset(self):
        return self.train_data
    
    def get_val_dataset(self):
        return self.val_data
    
    def get_test_dataset(self):
        return self.test_data
        
    def filter_data(self, dataset):
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
    
    def remove_induced_events(self, traces, label, metadata):
        relevant_idx = np.where(np.array(metadata['labels']) != 'induced or triggered event')[0]
        metadata = {k: v for k, v in metadata.items() if k in relevant_idx}
        return traces[relevant_idx], label[relevant_idx], metadata

    def change_key_to_index(self, input_dict):
        output_dict = {}
        for idx, key in enumerate(list(input_dict.keys())):
            output_dict[idx] = input_dict[key]
            output_dict[idx]['path'] = key
        return output_dict

    
    def filter_all_data(self, dataset):
        if cfg.data.load_filtered:
            logger.warning("Data assumed to be prefiltered, skipping filtering process.")
            return dataset
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

    def save_filtered_data(self, train_data, val_data, test_data):
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



    
