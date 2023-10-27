from global_config import cfg, model_cfg
from Classes.Loop import Loop
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
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
        # Fixed super call
        super(CNN_dense, self).__init__(label_map_detector, label_map_classifier, 
                                        detector_metrics, classifier_metrics, 
                                        detector_class_weights, classifier_class_weights)
        # Using the initializer
        self.initializer = self.get_initializer(model_cfg.initializer)
        self.conv_layers = []
        self.pool_layers = []
        self.dense_layers = []

        for i, (filter_, filter_size) in enumerate(zip(model_cfg.filters, model_cfg.filter_size)):
            # Setting layer names
            block_name = f"conv_block_{i}"
            if model_cfg.conv_type == "separable":
                self.conv_layers.append(tfl.SeparableConv1D(filter_, filter_size, activation=None, kernel_initializer=self.initializer, name=f"{block_name}_sepconv"))
            else:
                self.conv_layers.append(tfl.Conv1D(filter_, filter_size, activation=None, kernel_initializer=self.initializer, name=f"{block_name}_conv"))
            
            self.conv_layers.append(tfl.BatchNormalization(name=f"{block_name}_bn"))
            self.conv_layers.append(tfl.Activation(model_cfg.activation, name=f"{block_name}_act"))
            self.conv_layers.append(tfl.Dropout(model_cfg.dropout, name=f"{block_name}_drop"))
            
            if model_cfg.pool_type == "max":
                self.pool_layers.append(tfl.MaxPool1D(name=f"{block_name}_maxpool"))
            else:
                self.pool_layers.append(tfl.AveragePooling1D(name=f"{block_name}_avgpool"))

        self.flatten = tfl.Flatten(name="flatten")
        
        for i, units in enumerate(model_cfg.dense_units):
            dense_name = f"dense_{i}"
            self.dense_layers.append(tfl.Dense(units, activation=model_cfg.activation, kernel_initializer=self.initializer, name=dense_name))
            self.dense_layers.append(tfl.Dropout(model_cfg.dropout, name=f"{dense_name}_drop"))
        
        self.final_dense_detector = tfl.Dense(1, activation=None, kernel_initializer=self.initializer, name="final_dense_detector")  
        self.final_dense_classifier = tfl.Dense(1, activation=None, kernel_initializer=self.initializer, name="final_dense_classifier")


    @tf.function
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
        