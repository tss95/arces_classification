from global_config import cfg, logger
import tensorflow as tf
import numpy as np
from Classes.Augment import augment_pipeline
from tensorflow.keras.utils import Sequence
from functools import reduce
import math

class Generator(Sequence):

    def __init__(self, data, labels, shuffle=False, chunk_size=10000):
        unique_det, counts_true_detector = np.unique(labels['detector'], return_counts=True)
        unique_cls, counts_true_cls = np.unique(labels['classifier'], return_counts=True)

        true_detector_dist = dict(zip(unique_det, counts_true_detector))
        true_cls_dist = dict(zip(unique_cls, counts_true_cls))

        logger.debug(f"generator input det dist: {true_detector_dist}")
        logger.debug(f"generator input cls dist: {true_cls_dist}")

        self.indices = np.arange(len(data))
        if shuffle:
            np.random.shuffle(self.indices)

        with tf.device('/cpu:0'):
            # Initialize list to hold chunk datasets
            chunk_datasets = []
            for i in range(0, len(data), chunk_size):
                data_chunk = data[i:i+chunk_size]
                labels_chunk = {key: value[i:i+chunk_size] for key, value in labels.items()}
                
                chunk = tf.data.Dataset.from_tensor_slices((data_chunk, labels_chunk))
                chunk_datasets.append(chunk)

            # Use functools.reduce to concatenate all chunk datasets
            self.tf_dataset = reduce(lambda ds1, ds2: ds1.concatenate(ds2), chunk_datasets)

            # Batch, shuffle, and prefetch
            self.tf_dataset = self.tf_dataset.batch(cfg.optimizer.batch_size)
            if shuffle:
                self.tf_dataset = self.tf_dataset.shuffle(buffer_size=10000)
            self.tf_dataset = self.tf_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            self.tf_dataset = self.tf_dataset.repeat()  # Makes the dataset cycle indefinitely
        self.iterator = iter(self.tf_dataset)

    def __len__(self):
        return math.floor(len(self.indices) / cfg.optimizer.batch_size)

    def __getitem__(self, index):
        batch_data, batch_labels = next(self.iterator)
        unique, counts = np.unique(batch_labels['detector'], return_counts=True)
        logger.debug(f"__getitem__ batch_labels['detector'] distribution: {dict(zip(unique, counts))}")
        return batch_data, {'detector': batch_labels['detector'], 'classifier': batch_labels['classifier']}

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
