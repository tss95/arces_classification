from global_config import cfg, model_cfg
from Classes.Loop import Loop
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl 
import tensorflow.keras.backend as K


def get_model(label_map_detector, label_map_classifier, detector_metrics, classifier_metrics,
              detector_class_weights, classifier_class_weights):
    if cfg.model == "cnn_dense":
        return CNN_dense(label_map_detector, label_map_classifier, detector_metrics, classifier_metrics,
                         detector_class_weights, classifier_class_weights)
    else:
        raise ValueError("Model not found.")

class CNN_dense(Loop):
    def __init__(self, label_map_detector, label_map_classifier, detector_metrics, classifier_metrics, 
                 detector_class_weights, classifier_class_weights):
        super(CNN_dense, self).__init__(label_map_detector, label_map_classifier, detector_metrics, 
                                        classifier_metrics, detector_class_weights, classifier_class_weights)
        self.initializer = self.get_initializer("glorot_uniform")
        self.conv_layers = []
        self.pool_layers = []
        self.dense_layers = []

        for i in range(len(model_cfg.filters)):
            if model_cfg.conv_type == "separable":
                self.conv_layers.append(tfl.SeparableConv1D(model_cfg.filters[i], model_cfg.filter_size[i], activation=None))
            else:
                self.conv_layers.append(tfl.Conv1D(model_cfg.filters[i], model_cfg.filter_size[i], activation=None))
            
            self.conv_layers.append(tfl.BatchNormalization())
            self.conv_layers.append(tfl.Activation(model_cfg.activation))
            self.conv_layers.append(tfl.Dropout(model_cfg.dropout))
            
            if model_cfg.pool_type == "max":
                self.pool_layers.append(tfl.MaxPool1D())
            else:
                self.pool_layers.append(tfl.AveragePooling1D())

        self.flatten = tfl.Flatten()
        
        for units in model_cfg.dense_units:
            self.dense_layers.append(tfl.Dense(units, activation=model_cfg.activation))
            self.dense_layers.append(tfl.Dropout(model_cfg.dropout))
        
        self.final_dense_detector = tfl.Dense(1, activation=None)  # Binary output for detector
        self.final_dense_classifier = tfl.Dense(1, activation=None)  # Binary output for classifier


    def call(self, inputs, training=False):
        x = inputs
        for i in range(len(self.conv_layers)//4):  # Assuming each block has Conv, BN, Activation, Dropout
            for j in range(4):
                layer = self.conv_layers[4*i + j]
                x = layer(x, training=training)
            x = self.pool_layers[i](x)

        x = self.flatten(x)
        
        for layer in self.dense_layers:
            x = layer(x, training=training)
        
        output_detector = self.final_dense_detector(x)
        output_classifier = self.final_dense_classifier(x)

        return {'detector': output_detector, 'classifier': output_classifier}

    def get_initializer(self, initializer):
        if initializer == "glorot_uniform":
            return tf.initializers.glorot_uniform()
        elif initializer == "glorot_normal":
            return tf.initializers.glorot_normal()
        elif initializer == "he_uniform":
            return tf.initializers.he_uniform()
        elif initializer == "he_normal":
            return tf.initializers.he_normal()
        else:
            raise ValueError("Initializer not found.")