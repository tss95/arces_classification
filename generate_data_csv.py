import os
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from types import SimpleNamespace
import h5py
from tqdm import tqdm
import time
import logging
import logging.config
import csv
from config.logging_config import LOGGING_CONFIG

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('ARCES Classification')

def dict_to_namespace(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_namespace(v)
    return SimpleNamespace(**d)

config_dir = "/staff/tord/Workspace/arces_classification/config"
args = OmegaConf.load(f'{config_dir}/data_config.yaml')
args = OmegaConf.create(args)
OmegaConf.set_struct(args, False)
cfg = dict_to_namespace(args)

def make_data_dictionary(cfg):
    data_path = cfg.paths.data_path
    output_csv_path = cfg.paths.output_folder + "/" + "data.csv"
    all_paths = set()  # Using a set to ensure uniqueness
    total_dirs = len([entry.name for entry in os.scandir(data_path) if entry.is_dir()])
    
    print(f"Total directories to process: {total_dirs}")
    
    with os.scandir(data_path) as it:
        for dir_num, entry in enumerate(it, 1):
            if entry.is_dir():
                label = entry.name  # Get the label name
                print(f"\nProcessing directory {dir_num}/{total_dirs}: {label}")
                
                dir_start_time = time.time()
                elapsed_time = 0  # Initialize elapsed_time
                
                with os.scandir(entry.path) as it2:
                    for file_num, subentry in enumerate(it2, 1):
                        relative_path = os.path.join(label, subentry.name)  # Construct the relative path
                        all_paths.add(relative_path)
                            
                        if file_num % 5000 == 0:  # Update the console output every 5000 files
                            elapsed_time = time.time() - dir_start_time
                            print(f"\rProcessed {file_num} files in {elapsed_time:.2f} seconds", end="", flush=True)
                            
                print(f"\rProcessed {file_num} files in {elapsed_time:.2f} seconds", end="", flush=True)
    
    # Convert the set to a list and sort it
    all_paths = sorted(list(all_paths))
    
    # Save to CSV
    with open(output_csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for path in all_paths:
            csvwriter.writerow([path])
    
    return all_paths

all_paths = make_data_dictionary(cfg)