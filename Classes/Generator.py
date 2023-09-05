from global_config import logger, cfg
import numpy as np
import tensorflow as tf
from sklearn.pipeline import Pipeline
from Classes.Augment import augment_pipeline

class Generator:

    def __init__(self, tf_dataset, shuffle=False):
        self.tf_dataset = tf_dataset.batch(cfg.optimizer.batch_size)
        if shuffle:
            self.tf_dataset = self.tf_dataset.shuffle(buffer_size=1000)  # Adjust buffer_size as needed
        self.tf_dataset = self.tf_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    def __len__(self):
        return tf.data.experimental.cardinality(self.tf_dataset).numpy()

    def __iter__(self):
        return iter(self.tf_dataset)

class TrainGenerator(Generator):

    def __init__(self, tf_dataset):
        super().__init__(tf_dataset, shuffle=True)
        self.tf_dataset = self.tf_dataset.map(lambda x, y: (self.__augment_batch(x), y))

    def __augment_batch(self, batch):
        return augment_pipeline(batch)

class ValGenerator(Generator):

    def __init__(self, tf_dataset):
        super().__init__(tf_dataset, shuffle=False)

        
        
        


        