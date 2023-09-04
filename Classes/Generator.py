from global_config import logger, cfg
import numpy as np
import tensorflow as tf

class Generator:

    def __init__(self, dataset):
        self.data = dataset[0]
        self.labels = dataset[1]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.data))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size

        batch_indexes = self.indexes[start_idx:end_idx]
        
        X = np.array([self.data[i] for i in batch_indexes])
        y = np.array([self.labels[i] for i in batch_indexes])

        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

class TrainGenerator(Generator):

    def __init__(self, dataset):
        super


        