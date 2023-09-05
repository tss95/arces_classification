from global_config import cfg, model_cfg
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl 
import tensorflow.keras.backend as K


def get_model():
    if cfg.model == "cnn_dense":
        return CNN_dense()
    else:
        raise ValueError("Model not found.")

class CNN_dense(tf.keras.Model):
    def __init__(self):
        self.initializer = self.get_initializer(model_cfg.initializer)
        if model_cfg.conv_type == "seperable":
            self.conv_layer = tfl.SeparableConv1D
        else:
            self.conv_layer = tfl.Conv1D

        if model_cfg.pool_type == "max":
            self.pool_layer = tfl.MaxPool1D
        else:
            self.pool_layer = tfl.AveragePooling1D
    
    def build(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        regularizer = tf.keras.regularizers.l1_l2(l1=model_cfg.l1, l2=model_cfg.l2)
        for i in range(len(model_cfg.filters)):
            if i == 0:
                x = self.conv_layer(model_cfg.filters[i], 
                                    model_cfg.kernel_size[i], 
                                    activation=None,
                                    kernel_initializer = self.initializer,
                                    kernel_regularizer = regularizer)(inputs)
            else:
                x = self.conv_layer(model_cfg.filters[i], model_cfg.kernel_size[i], activation=None)(x)
            x = tfl.BatchNormalization()(x)
            x = tfl.Activation(model_cfg.activation)(x)
            x = tfl.Dropout(model_cfg.dropout)(x)
            x = self.pool_layer()(x)
        x = tfl.Flatten()(x)
        for i in range(len(model_cfg.dense_units)):
            x = tfl.Dense(model_cfg.dense_units[i], activation=model_cfg.activation)(x)
            x = tfl.Dropout(model_cfg.dropout)(x)
        outputs = tfl.Dense(cfg.data.num_classes, activation="softmax" if model_cfg.num_classes > 2 else "sigmoid")(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
            
    
    def get_initializer(self, initializer):
        if initializer == "glorot_uniform":
            return tfl.initializers.glorot_uniform()
        elif initializer == "glorot_normal":
            return tfl.initializers.glorot_normal()
        elif initializer == "he_uniform":
            return tfl.initializers.he_uniform()
        elif initializer == "he_normal":
            return tfl.initializers.he_normal()
        else:
            raise ValueError("Initializer not found.")
        
    @property
    def num_parameters(self):
        return sum([np.prod(K.get_value(w).shape) for w in self.model.trainable_weights])
    
    def summary(self):
        return self.model.summary()

    def call(self, inputs):
        return self.model(inputs)
