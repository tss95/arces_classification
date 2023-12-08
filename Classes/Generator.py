from global_config import cfg, logger
import tensorflow as tf
import numpy as np
from Classes.Augment import augment_pipeline
from tensorflow.keras.utils import Sequence
from typing import Dict, Any, Tuple, List
from functools import reduce
import math

class Generator(Sequence):
    """
    A generator class for yielding batches of data and labels for training or validation.

    Attributes:
        data (np.ndarray): The input data.
        labels (Dict[str, np.ndarray]): The labels for the data.
        scaler (Any): A scaler object for data normalization.
        indices (np.ndarray): Array of indices of the data.
        tf_dataset (tf.data.Dataset): TensorFlow Dataset object.
        iterator (Iterator): Iterator for the TensorFlow Dataset.
    """

    def __init__(self, data: np.ndarray, labels: Dict[str, np.ndarray], scaler: Any, shuffle: bool = False, chunk_size: int = 10000):
        """
        Initializes the Generator instance.

        Args:
            data (np.ndarray): The input data.
            labels (Dict[str, np.ndarray]): The labels for the data.
            scaler (Any): A scaler object used for normalizing the data.
            shuffle (bool): Whether to shuffle the data.
            chunk_size (int): Size of chunks to divide the data into for processing.
        """
        self.data = data
        self.labels = labels
        self.scaler = scaler
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

    def __len__(self) -> int:
        """
        Returns the number of batches in the sequence.

        Returns:
            int: The number of batches.
        """
        return math.floor(len(self.indices) / cfg.optimizer.batch_size)
    
    def augment_batch(self, batch_data: np.ndarray, batch_labels: Dict[str, np.ndarray]):
        """
        Augments a batch of data and labels.

        Args:
            batch_data (np.ndarray): The batch of data to augment.
            batch_labels (Dict[str, np.ndarray]): The labels of the batch.

        Returns:
            Tuple[np.ndarray, Dict[str, np.ndarray]]: The augmented data and labels.
        """
        return augment_pipeline(batch_data, batch_labels)


    def __getitem__(self, index: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Retrieves a batch at the given index.

        Args:
            index (int): The index of the batch to retrieve.

        Returns:
            Tuple[np.ndarray, Dict[str, np.ndarray]]: The data and labels of the batch.
        """
        batch_data, batch_labels = next(self.iterator)
        logger.debug(f"__getitem__ batch_labels['detector'] distribution: {dict(zip(*np.unique(batch_labels['detector'], return_counts=True)))}")
        return batch_data, {'detector': batch_labels['detector'], 'classifier': batch_labels['classifier']}

    def get_item_with_index(self, index: int) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        """
        Retrieves a batch at the given index along with the original indices.

        Args:
            index (int): The index of the batch to retrieve.

        Returns:
            Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]: The data and labels of the batch, and the original indices.
        """
        # Calculate the start and end index for the specific batch
        start_idx = index * cfg.optimizer.batch_size
        end_idx = start_idx + cfg.optimizer.batch_size

        # Slice the data and labels based on the calculated indices
        data_slice = self.data[start_idx:end_idx]
        labels_slice = {key: value[start_idx:end_idx] for key, value in self.labels.items()}

        # Extract the original indices
        original_indices = self.indices[start_idx:end_idx]

        return data_slice, {'detector': labels_slice['detector'], 'classifier': labels_slice['classifier']}, original_indices


class TrainGenerator(Generator):
    """
    A generator class for training, inheriting from the base Generator class.

    This class is specifically for the training data, where data augmentation and shuffling are applied.
    """

    def __init__(self, data: np.ndarray, labels: Dict[str, np.ndarray], scaler: Any):
        """
        Initializes the TrainGenerator instance.

        Args:
            data (np.ndarray): The training data.
            labels (Dict[str, np.ndarray]): The labels for the training data.
            scaler (Any): A scaler object used for normalizing the data.
        """
        super().__init__(data, labels, scaler, shuffle=True)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Retrieves and processes a batch at the given index for training.

        Args:
            index (int): The index of the batch to retrieve.

        Returns:
            Tuple[np.ndarray, Dict[str, np.ndarray]]: The processed data and labels of the batch.
        """
        batch_data, batch_labels = super().__getitem__(index)
        batch_data, batch_labels = self.augment_batch(batch_data, batch_labels)
        transformed_batch_data = self.scaler.transform(batch_data)
        return transformed_batch_data, batch_labels

    def get_item_with_index(self, index: int) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        """
        Retrieves and processes a batch at the given index for training, along with the original indices.

        Args:
            index (int): The index of the batch to retrieve.

        Returns:
            Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]: The processed data and labels of the batch, and the original indices.
        """
        batch_data, batch_labels, original_indices = super().get_item_with_index(index)
        batch_data, batch_labels = self.augment_batch(batch_data, batch_labels)
        transformed_batch_data = self.scaler.transform(batch_data)
        return transformed_batch_data, batch_labels, original_indices

    def on_epoch_end(self):
        """
        Reset the iterator at the end of each epoch.
        """
        self.iterator = iter(self.tf_dataset)  # Reset the iterator

class ValGenerator(Generator):
    """
    A generator class for validation, inheriting from the base Generator class.

    This class is specifically for the validation data, where data is not shuffled.
    """

    def __init__(self, data: np.ndarray, labels: Dict[str, np.ndarray], scaler: Any):
        """
        Initializes the ValGenerator instance.

        Args:
            data (np.ndarray): The validation data.
            labels (Dict[str, np.ndarray]): The labels for the validation data.
            scaler (Any): A scaler object used for normalizing the data.
        """
        super().__init__(data, labels, scaler, shuffle=False)

    def on_epoch_end(self):
        """
        Reset the iterator at the end of each epoch.

        This method is called at the end of every epoch during validation. It resets the iterator 
        to ensure that the validation dataset is processed from the beginning in the next epoch.
        """
        self.iterator = iter(self.tf_dataset)  # Reset the iterator

    def __getitem__(self, index: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Retrieves and processes a batch at the given index for validation.

        Args:
            index (int): The index of the batch to retrieve.

        Returns:
            Tuple[np.ndarray, Dict[str, np.ndarray]]: The processed data and labels of the batch.
        """
        batch_data, batch_labels = super().__getitem__(index)
        transformed_batch_data = self.scaler.transform(batch_data)
        return transformed_batch_data, batch_labels
    
    def get_item_with_index(self, index: int) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        """
        Retrieves and processes a batch at the given index for validation, along with the original indices.

        Args:
            index (int): The index of the batch to retrieve.

        Returns:
            Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]: The processed data and labels of the batch, and the original indices.
        """
        batch_data, batch_labels, original_indices = super().get_item_with_index(index)
        # Careful about the augmentations being applied here:
        batch_data, batch_labels = self.augment_batch(batch_data, batch_labels)
        transformed_batch_data = self.scaler.transform(batch_data)
        return transformed_batch_data, batch_labels, original_indices
