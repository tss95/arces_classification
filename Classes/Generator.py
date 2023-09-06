from global_config import cfg
import tensorflow as tf
from Classes.Augment import augment_pipeline
from tensorflow.keras.utils import Sequence

class Generator(Sequence):

    def __init__(self, data, labels, shuffle=False, chunk_size=10000):
        # Convert to tf.data.Dataset in chunks
        with tf.device('/cpu:0'):
            # Convert to tf.data.Dataset in chunks
            dataset = tf.data.Dataset.from_tensor_slices(([], []))
            for i in range(0, len(data), chunk_size):
                chunk = tf.data.Dataset.from_tensor_slices((data[i:i+chunk_size], labels[i:i+chunk_size]))
                dataset = dataset.concatenate(chunk)
            
            self.tf_dataset = dataset.batch(cfg.optimizer.batch_size)
            if shuffle:
                self.tf_dataset = self.tf_dataset.shuffle(buffer_size=1000)
            self.tf_dataset = self.tf_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    def __len__(self):
        return tf.data.experimental.cardinality(self.tf_dataset).numpy()

    def __iter__(self):
        return iter(self.tf_dataset)
class TrainGenerator(Generator):

    def __init__(self, data, labels):
        super().__init__(data, labels, shuffle=True)

    def __iter__(self):
        for batch_data, batch_labels in super().__iter__():
            augmented_batch_data = self.__augment_batch(batch_data)
            yield augmented_batch_data, batch_labels

    def __augment_batch(self, batch):
        return augment_pipeline(batch)

class ValGenerator(Generator):

    def __init__(self, data, labels):
        super().__init__(data, labels, shuffle=False)

        
        
        


        