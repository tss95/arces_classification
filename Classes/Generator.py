from global_config import cfg
import tensorflow as tf
import numpy as np
from Classes.Augment import augment_pipeline
from tensorflow.keras.utils import Sequence
from functools import reduce

class Generator(Sequence):

    def __init__(self, data, labels, shuffle=False, chunk_size=10000):
        self.indices = np.arange(len(data))
        if shuffle:
            np.random.shuffle(self.indices)

        with tf.device('/cpu:0'):
            # Initialize list to hold chunk datasets
            chunk_datasets = []
            for i in range(0, len(data), chunk_size):
                chunk = tf.data.Dataset.from_tensor_slices((data[i:i+chunk_size], labels[i:i+chunk_size]))
                chunk_datasets.append(chunk)

            # Use functools.reduce to concatenate all chunk datasets
            self.tf_dataset = reduce(lambda ds1, ds2: ds1.concatenate(ds2), chunk_datasets)

            # Batch, shuffle, and prefetch
            self.tf_dataset = self.tf_dataset.batch(cfg.optimizer.batch_size)
            if shuffle:
                self.tf_dataset = self.tf_dataset.shuffle(buffer_size=1000)
            self.tf_dataset = self.tf_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        
        self.iterator = iter(self.tf_dataset)

    def __len__(self):
        return tf.data.experimental.cardinality(self.tf_dataset).numpy()

    def __getitem__(self, index):
        return next(self.iterator)

    def get_item_with_index(self, index):
        data, labels = self.__getitem__(index)
        start_idx = index * cfg.optimizer.batch_size
        end_idx = start_idx + len(data)
        original_indices = self.indices[start_idx:end_idx]
        return data, labels, original_indices


class TrainGenerator(Generator):

    def __init__(self, data, labels):
        super().__init__(data, labels, shuffle=True)

    def __getitem__(self, index):
        batch_data, batch_labels = super().__getitem__(index)
        batch_data, batch_labels = self.__augment_batch(batch_data, batch_labels)
        return batch_data, batch_labels

    def get_item_with_index(self, index):
        batch_data, batch_labels, original_indices = super().get_item_with_index(index)
        batch_data, batch_labels = self.__augment_batch(batch_data, batch_labels)
        return batch_data, batch_labels, original_indices


    def __augment_batch(self, batch_data, batch_labels):
        return augment_pipeline(batch_data, batch_labels)

    def on_epoch_end(self):
        self.iterator = iter(self.tf_dataset)  # Reset the iterator


class ValGenerator(Generator):

    def __init__(self, data, labels):
        super().__init__(data, labels, shuffle=False)

    def on_epoch_end(self):
        self.iterator = iter(self.tf_dataset)  # Reset the iterator
