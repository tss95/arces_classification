# Your existing imports and setup here
from global_config import logger, cfg
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm
import os
import h5py
import pickle

# Paths
data_csv_path = cfg.paths.data_csv
data_path = cfg.paths.data_path
output_folder = cfg.paths.output_folder

# Data configurations
train_prop = cfg.data.train_prop
val_prop = cfg.data.val_prop
first_date = cfg.data.first_date


# Read the CSV file
logger.info("Reading data CSV.")
df = pd.read_csv(data_csv_path, header=None, names=['filename'])

# Parse filename to extract timestamp
df['timestamp'] = pd.to_datetime(df['filename'].apply(lambda x: x.split('/')[-1].split('.')[0]))

# Filter by date
logger.info(f"Filtering data from {first_date} onwards.")
df_filtered = df[df['timestamp'] >= pd.Timestamp(first_date)]

# Split into training, validation, and test sets
logger.info("Splitting data into training, validation, and test sets.")
train_size = int(len(df_filtered) * train_prop)
val_size = int((len(df_filtered) - train_size) * val_prop)
train_data, temp_data = train_test_split(df_filtered, train_size=train_size, random_state=42)
val_data, test_data = train_test_split(temp_data, train_size=val_size, random_state=42)

# Function to load data (Placeholder)
def path_to_trace(path):
    with h5py.File(path, 'r') as dp:
        trace_array = np.array(dp.get('traces'))
        info = np.array(dp.get('event_info'))
        info = str(info)    
        info = info[2:len(info)-1]
        info = json.loads(info)
    return trace_array, info

def load_data(dataset, dataset_desc):
    metadata = {}
    traces = []
    labels = []
    core_path = cfg.paths.data_path
    for path in tqdm(dataset, desc=f"Loading {dataset_desc}"):
        trace, meta = path_to_trace(core_path+'/'+path)
        metadata[path] = meta
        try:
            if meta['event_type'] == 'induced':
                labels.append("earthquake")
            else:
                labels.append(meta['event_type'])
        except Exception:
            labels.append('noise')
        traces.append(trace)
    return np.array(traces), labels, metadata 

# Load and save the data in NumPy format
logger.info("Loading and saving the data.")
for dataset, name in [(train_data, 'train'), (val_data, 'val'), (test_data, 'test')]:
    traces, labels, metadata = load_data(dataset["filename"], name)
    np.save(os.path.join(output_folder, f'{name}_traces.npy'), traces)
    np.save(os.path.join(output_folder, f'{name}_labels.npy'), labels)
    with open(os.path.join(output_folder, f'{name}_metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
logger.info("Data preparation completed.")