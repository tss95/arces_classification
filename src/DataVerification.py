
import numpy as np
import pandas as pd
from global_config import logger
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import datetime

class Verficiation:
    
    def __init__(self, dataloader, events, cfg):
        self.dataloader = dataloader
        self.events = events
        self.cfg = cfg
        
    def plot_waveform(self, ax, waveform, window, sample_rate, start_index, num_ticks=4):
        num_channels = waveform.shape[0]
        time = np.arange(waveform.shape[1]) / sample_rate
        time += window[0]  # adjust time to start at the start of the window

        for i in range(num_channels):
            ax[i].plot(time, waveform[i])
            #ax[i].set_xlim(window)

            #if i == num_channels - 1:
            #    ax[i].set_xlabel('Time (s)')
            #    ticks = np.linspace(window[0], window[1], num_ticks)
            #    ax[i].set_xticks(ticks)
            #    ax[i].set_xticklabels([datetime.datetime.fromtimestamp(tick).strftime('%Y-%m-%d %H:%M:%S') for tick in ticks])
            #else:
            #    ax[i].set_xticklabels([])

        # Convert the start time to a datetime object and format it as a string
        if start_index:
            start_time = datetime.datetime.fromtimestamp(window[0] + start_index / sample_rate)
        else: 
            start_time = datetime.datetime.fromtimestamp(window[0] / sample_rate)
        start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
        return ax, start_time_str
        
    def check_one_batch(self):
        dataiter = iter(self.dataloader)
        data, _, ids = next(dataiter)
        labels = []
        # send data to cpu and numpy:
        data = data.cpu().numpy()
        for id in ids:
            labels.append(self.events[id]['Y'])
        logger.info(f"Batch sample shape: {data.shape}")
        # Plot one waveform per label:
        unique_labels = np.unique(np.array(labels))
        for label in unique_labels:
            # Use plot_waveforms to plot one waveform per label
            # Find the first index of the label
            logger.warning(f"Plotting waveform for label {label}")
            index = labels.index(label)
            sample, label, id = data[index], labels[index], ids[index]
            start_index = self.events[id]['start_index']
            window = self.events[id]['window']
            fig, axs = plt.subplots(3, figsize=(10, 6))
            axs, start_time_str = self.plot_waveform(axs, data[index], window, self.cfg.data.sample_rate, start_index)
            axs[0].set_title(f"Waveform for label {label} starting at {start_time_str}")
            plt.savefig(os.path.join(self.cfg.project_paths.output_folders.plots_folder, f"waveform_label_{label}.png"))
    
    def check_raw_data(self):
        # Checks for the raw data:
        raw = []
        for id in self.events.keys():
            raw.append(self.events[id]['X'].numpy())
        raw = np.array(raw)
        logger.info(f"Raw data shape shape: {raw.shape}")

        self.plot_distribution(raw, "Raw data distribution", "raw_data_distribution.png")
        
    def check_processed_data(self):
        dataiter = iter(self.dataloader)
        data, labels, ids = [], [], []
        for batch in dataiter:
            data.append(batch[0])
            for id in batch[2]:
                labels.append(self.events[id]['Y'])
            ids.append(batch[2])
        data = torch.cat(data, dim=0)
        data = data.cpu().numpy()
        logger.info(f"Processed data shape: {data.shape}")
        self.plot_distribution(data, "Processed data distribution", "processed_data_distribution.png")
        # Now lets look at per labels distributions:
        unique_labels = np.unique(labels)
        for label in unique_labels:
            label_data = data[labels == label]
            self.plot_distribution(label_data, f"Processed data distribution for label {label}", f"processed_data_distribution_label_{label}.png")
        
    def plot_distribution(self, data, title, filename):
        # Create a figure with 3 subplots, one for each channel
        fig, axs = plt.subplots(3, figsize=(10, 6))

        # Iterate over the channels
        for i in range(3):
            # Select the data for this channel
            channel_data = data[:, i, :].flatten()
            # Compute the histogram
            hist, bins = np.histogram(channel_data, bins='auto')
            # Plot the histogram on the corresponding subplot
            axs[i].bar(bins[:-1], hist, width=(bins[1]-bins[0]), color='blue', alpha=0.7)
            axs[i].set_title(f'Distribution of values in channel {i+1}')
            axs[i].set_xlabel('Value')
            axs[i].set_ylabel('Frequency')

        # Adjust the layout and show the plot
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.project_paths.plots_folder, filename))
        plt.close(fig)