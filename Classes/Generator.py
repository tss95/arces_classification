from global_config import cfg
import tensorflow as tf
from Classes.Augment import augment_pipeline
from tensorflow.keras.utils import Sequence

class Generator(Sequence):

    def __init__(self, data, labels, shuffle=False, chunk_size=10000):
        with tf.device('/cpu:0'):
            dataset = tf.data.Dataset.from_tensor_slices(([], []))
            for i in range(0, len(data), chunk_size):
                chunk = tf.data.Dataset.from_tensor_slices((data[i:i+chunk_size], labels[i:i+chunk_size]))
                dataset = dataset.concatenate(chunk)
            
            self.tf_dataset = dataset.batch(cfg.optimizer.batch_size)
            if shuffle:
                self.tf_dataset = self.tf_dataset.shuffle(buffer_size=1000)
            self.tf_dataset = self.tf_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        
        self.iterator = iter(self.tf_dataset)

    def __len__(self):
        return tf.data.experimental.cardinality(self.tf_dataset).numpy()

    def __getitem__(self, index):
        return next(self.iterator)


class TrainGenerator(Generator):

    def __init__(self, data, labels):
        super().__init__(data, labels, shuffle=True)

    def __getitem__(self, index):
        batch_data, batch_labels = super().__getitem__(index)
        batch_data, batch_labels = self.__augment_batch(batch_data, batch_labels)
        return batch_data, batch_labels

    def __augment_batch(self, batch_data, batch_labels):
        return augment_pipeline(batch_data, batch_labels)

    def on_epoch_end(self):
        self.iterator = iter(self.tf_dataset)  # Reset the iterator


class ValGenerator(Generator):

    def __init__(self, data, labels):
        super().__init__(data, labels, shuffle=False)

    def on_epoch_end(self):
        self.iterator = iter(self.tf_dataset)  # Reset the iterator
