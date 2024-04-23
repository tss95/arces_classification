from global_config import cfg, model_cfg
from src.Loop import Loop
from src.S4D import S4D
import numpy as np
import tensorflow as tf
from nais.Models import AlexNet1D
tf.get_logger().setLevel('ERROR')
import tensorflow.keras.layers as tfl 
import tensorflow.keras.backend as K
from typing import Dict, List, Callable


def get_initializer(initializer: str) -> tf.initializers.Initializer:
    """
    Selects a TensorFlow initializer based on a given string.

    Args:
        initializer (str): The name of the initializer to use. Supported values are
                           "glorot_uniform", "glorot_normal", "he_uniform", and "he_normal".

    Returns:
        tf.initializers.Initializer: The corresponding TensorFlow initializer.

    Raises:
        ValueError: If the initializer name is not recognized.

    """
    pass

def get_model(label_map_detector: Dict[int, str], 
              label_map_classifier: Dict[int, str], 
              detector_metrics: List[Callable], 
              classifier_metrics: List[Callable],
              detector_class_weights: Dict[str, float], 
              classifier_class_weights: Dict[str, float]) -> Loop:
    """
    Creates and returns a model based on configuration settings.

    Args:
        label_map_detector (Dict[int, str]): Mapping of labels for the detector.
        label_map_classifier (Dict[int, str]): Mapping of labels for the classifier.
        detector_metrics (List[Callable]): Metrics to be used for the detector.
        classifier_metrics (List[Callable]): Metrics to be used for the classifier.
        detector_class_weights (Dict[str, float]): Class weights for the detector.
        classifier_class_weights (Dict[str, float]): Class weights for the classifier.

    Returns:
        Loop: An instance of the Loop class representing the model.

    Raises:
        ValueError: If the model specified in the configuration is not found.

    """
    if cfg.model == "cnn_dense":
        return CNN_dense(label_map_detector, label_map_classifier, detector_metrics, classifier_metrics,
                         detector_class_weights, classifier_class_weights)

    if cfg.model == "alexnet":
        return AlexNet(label_map_detector, label_map_classifier, detector_metrics, classifier_metrics,
                       detector_class_weights, classifier_class_weights)
    if cfg.model == "s4d":
        return S4DModel(label_map_detector, label_map_classifier, detector_metrics, classifier_metrics,
                        detector_class_weights, classifier_class_weights)
    else:
        raise ValueError("Model not found.")
    

class AlexNet(Loop):
    """
    A custom implementation of the AlexNet model tailored for specific use cases.

    Attributes:
        label_map_detector (Dict[int, str]): Mapping of labels for the detector.
        label_map_classifier (Dict[int, str]): Mapping of labels for the classifier.
        detector_metrics (List[Callable]): Metrics for the detector.
        classifier_metrics (List[Callable]): Metrics for the classifier.
        detector_class_weights (Dict[str, float]): Class weights for the detector.
        classifier_class_weights (Dict[str, float]): Class weights for the classifier.
        backbone (AlexNet1D): The core AlexNet1D model.
        final_dense_detector (tfl.Dense): Final dense layer for the detector.
        final_dense_classifier (tfl.Dense): Final dense layer for the classifier.

    """
    def __init__(self, label_map_detector: Dict[int, str], label_map_classifier: Dict[int, str], 
                 detector_metrics: List[Callable], classifier_metrics: List[Callable],
                 detector_class_weights: Dict[str, float], classifier_class_weights: Dict[str, float]):
        """
        Initializes the AlexNet model with the specified parameters.

        Args:
            label_map_detector (Dict[int, str]): Mapping of labels for the detector.
            label_map_classifier (Dict[int, str]): Mapping of labels for the classifier.
            detector_metrics (List[Callable]): Metrics for the detector.
            classifier_metrics (List[Callable]): Metrics for the classifier.
            detector_class_weights (Dict[str, float]): Class weights for the detector.
            classifier_class_weights (Dict[str, float]): Class weights for the classifier.
        """
        super(AlexNet, self).__init__(label_map_detector, label_map_classifier, 
                                        detector_metrics, classifier_metrics, 
                                        detector_class_weights, classifier_class_weights)
        kernel_sizes = model_cfg.kernel_sizes if model_cfg.kernel_sizes != "None" else None
        filters = model_cfg.filters if model_cfg.filters != "None" else None
        self.initializer = get_initializer(model_cfg.initializer)
        self.backbone = AlexNet1D(kernel_sizes = kernel_sizes,
                                  filters = filters,
                                  num_outputs = None,
                                  pooling = model_cfg.pool_type)
        self.final_dense_detector = tfl.Dense(1, activation=None, kernel_initializer=self.initializer, name="final_dense_detector")  
        self.final_dense_classifier = tfl.Dense(1, activation=None, kernel_initializer=self.initializer, name="final_dense_classifier")


    @tf.function
    def call(self, inputs: tf.Tensor, training: bool = False) -> Dict[str, tf.Tensor]:
        """
        Forward pass of the model.

        Args:
            inputs (tf.Tensor): Input tensor to the model.
            training (bool, optional): Whether the model is in training mode. Defaults to False.

        Returns:
            Dict[str, tf.Tensor]: A dictionary with keys 'detector' and 'classifier', containing
                                  the output from the detector and classifier respectively.
        """
        x = inputs
        x = self.backbone(x, training=training)
        output_detector = self.final_dense_detector(x)
        output_classifier = self.final_dense_classifier(x)

        return {'detector': output_detector, 'classifier': output_classifier}
        

class CNN_dense(Loop):
    """
    A custom CNN model with dense layers, tailored for specific tasks.

    Attributes:
        conv_layers (List[tfl.Layer]): List of convolutional layers.
        pool_layers (List[tfl.Layer]): List of pooling layers.
        dense_layers (List[tfl.Layer]): List of dense layers.
        final_dense_detector (tfl.Dense): Final dense layer for the detector.
        final_dense_classifier (tfl.Dense): Final dense layer for the classifier.

    """
    def __init__(self, label_map_detector: Dict[int, str], label_map_classifier: Dict[int, str], 
                 detector_metrics: List[Callable], classifier_metrics: List[Callable], 
                 detector_class_weights: Dict[str, float], classifier_class_weights: Dict[str, float]):
        """
        Initializes the CNN_dense model with the specified parameters.

        Args:
            label_map_detector (Dict[int, str]): Mapping of labels for the detector.
            label_map_classifier (Dict[int, str]): Mapping of labels for the classifier.
            detector_metrics (List[Callable]): Metrics for the detector.
            classifier_metrics (List[Callable]): Metrics for the classifier.
            detector_class_weights (Dict[str, float]): Class weights for the detector.
            classifier_class_weights (Dict[str, float]): Class weights for the classifier.
        """
        # Fixed super call
        super(CNN_dense, self).__init__(label_map_detector, label_map_classifier, 
                                        detector_metrics, classifier_metrics, 
                                        detector_class_weights, classifier_class_weights)
        # Using the initializer
        self.initializer = get_initializer(model_cfg.initializer)
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
    def call(self, inputs: tf.Tensor, training: bool = False) -> Dict[str, tf.Tensor]:
        """
        Forward pass of the model.

        Args:
            inputs (tf.Tensor): Input tensor to the model.
            training (bool, optional): Whether the model is in training mode. Defaults to False.

        Returns:
            Dict[str, tf.Tensor]: A dictionary with keys 'detector' and 'classifier', containing
                                  the output from the detector and classifier respectively.
        """
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
    

if __name__ == "__main__":
    from global_config import logger, cfg, model_cfg

    """Test to see if the model has readable output shapes."""
    detector_label_map = {0: "noise", 1: "event"}
    classifier_label_map = {0:"earthquake", 1:"exlposion"}
    label_maps = {"detector": detector_label_map, "classifier": classifier_label_map}


    input_shape = (cfg.live.length*cfg.live.sample_rate + 1, 3)
    detector_class_weight_dict = {"noise": 1, "event": 1}
    classifier_class_weight_dict = {"earthquake": 1, "explosion": 1}

    logger.info("Input shape to the model: " + str(input_shape))
    classifier_metrics = [None]
    detector_metrics = [None]
    model = get_model(detector_label_map, classifier_label_map, detector_metrics, classifier_metrics, 
                    detector_class_weight_dict, classifier_class_weight_dict)
    model.build(input_shape=(None, 1000, 1))
    model.summary()
    print(model(tf.random.normal((1, 1000, 1)), training=True))


        