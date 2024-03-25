from global_config import model_cfg, logger
from src.Loop_torch import Loop
import numpy as np
#from nais.Models import AlexNet1D
import torch
import pytorch_lightning as pl
from torch.nn import functional as F
import torch.nn as nn
from typing import Dict, List, Callable


def get_model(input_shape,
              detector_metrics_list: List[str],
              classifier_metrics_list: list[str],
              label_map_detector: Dict[str, int], 
              label_map_classifier: Dict[str, int],
              detector_class_weights: Dict[str, float], 
              classifier_class_weights: Dict[str, float],
              cfg):
    """
    Creates and returns a model based on configuration settings.

    Args:
        input_shape (int): The shape of (chanenls, timesteps).
        label_map_detector (Dict[int, str]): Mapping of labels for the detector.
        label_map_classifier (Dict[int, str]): Mapping of labels for the classifier.
        detector_class_weights (Dict[str, float]): Class weights for the detector.
        classifier_class_weights (Dict[str, float]): Class weights for the classifier.

    Returns:
        Loop: An instance of the Loop class representing the model.

    Raises:
        ValueError: If the model specified in the configuration is not found.

    """
    if cfg.model == "cnn_dense":
        model = CNN_dense(input_shape,
                        detector_metrics_list,
                        classifier_metrics_list,
                        label_map_detector, 
                        label_map_classifier,
                        detector_class_weights, 
                        classifier_class_weights,
                        cfg)
    else:
        raise ValueError("Model not found.")
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model
    

def get_initializer(initializer: str):
    if initializer == 'normal':
        return nn.init.normal_
    elif initializer == 'uniform':
        return nn.init.uniform_
    elif initializer == 'xavier_uniform':
        return nn.init.xavier_uniform_
    elif initializer == 'xavier_normal':
        return nn.init.xavier_normal_
    elif initializer == 'kaiming_uniform':
        return nn.init.kaiming_uniform_
    elif initializer == 'kaiming_normal':
        return nn.init.kaiming_normal_
    else:
        raise ValueError(f'Unknown initializer: {initializer}')
        

class CNN_dense(Loop):
    """
    A custom CNN model with dense layers, tailored for specific tasks.
    PyTorch expects data to be in shape (batch, channels, timesteps).

    Attributes:
        conv_layers (List[tfl.Layer]): List of convolutional layers.
        pool_layers (List[tfl.Layer]): List of pooling layers.
        dense_layers (List[tfl.Layer]): List of dense layers.
        final_dense_detector (tfl.Dense): Final dense layer for the detector.
        final_dense_classifier (tfl.Dense): Final dense layer for the classifier.

    """

    def __init__(self, input_shape,
                detector_metrics_list: List[str],
                classifier_metrics_list: list[str],
                label_map_detector: Dict[str, int], 
                label_map_classifier: Dict[str, int],
                detector_class_weights: Dict[str, float], 
                classifier_class_weights: Dict[str, float],
                cfg):
        
        super(CNN_dense, self).__init__(input_shape,
                                        detector_metrics_list,
                                        classifier_metrics_list,
                                        label_map_detector, 
                                        label_map_classifier,
                                        detector_class_weights, 
                                        classifier_class_weights,
                                        cfg)
        self.conv_blocks = nn.ModuleDict()
        self.pool_layers = nn.ModuleList()
        self.dense_blocks = nn.ModuleDict()

        for i, (filter_, filter_size) in enumerate(zip(model_cfg.filters, model_cfg.filter_size)):
            block_name = f"conv_block_{i}"
            if model_cfg.conv_type == "separable":
                raise NotImplementedError("Separable convolutions are not implemented yet.")
            else:
                if i == 0:
                    self.conv_blocks[block_name] = nn.Sequential(
                        nn.Conv1d(self.channels, filter_, filter_size),
                        nn.BatchNorm1d(filter_),
                        nn.ReLU(),
                        nn.Dropout(model_cfg.dropout)
                    )
                else:
                    self.conv_blocks[block_name] = nn.Sequential(
                        nn.Conv1d(model_cfg.filters[i-1], filter_, filter_size),
                        nn.BatchNorm1d(filter_),
                        nn.ReLU(),
                        nn.Dropout(model_cfg.dropout)
                    )
            if model_cfg.pool_type == "max":
                self.pool_layers.append(nn.MaxPool1d(filter_size))
            else:
                self.pool_layers.append(nn.AvgPool1d(filter_size)) 

        output_size = self.timesteps  # assuming input_shape is (channels, timesteps)
        for filter_size in model_cfg.filter_size:
            output_size = (output_size - filter_size + 1)  # convolution
            output_size = ((output_size - filter_size) // filter_size) + 1  # pooling
        dense_input_size = output_size * model_cfg.filters[-1]
        logger.info(f"dense_input_shape: {dense_input_size}, output_size: {output_size}, timesteps: {self.timesteps}")
        for i, units in enumerate(model_cfg.dense_units):
            if i == 0:
                self.dense_blocks[f"dense_{i}"] = nn.Sequential(
                    nn.Linear(dense_input_size, units),
                    nn.ReLU(),
                    nn.Dropout(model_cfg.dropout)
                )
            else:
                dense_name = f"dense_{i}"
                self.dense_blocks[dense_name] = nn.Sequential(
                    nn.Linear(model_cfg.dense_units[i-1], units),
                    nn.ReLU(),
                    nn.Dropout(model_cfg.dropout)
                )
        self.final_dense_detector = nn.Linear(model_cfg.dense_units[-1], 1)
        self.final_dense_classifier = nn.Linear(model_cfg.dense_units[-1], 1)
        self.initialize_weights()

    def initialize_weights(self):
        initializer = get_initializer(model_cfg.initializer)
        for block in self.conv_blocks.values():
            for layer in block:
                if isinstance(layer, nn.Conv1d):
                    initializer(layer.weight)
        for block in self.dense_blocks.values():
            for layer in block:
                if isinstance(layer, nn.Linear):
                    initializer(layer.weight)
        initializer(self.final_dense_detector.weight)
        initializer(self.final_dense_classifier.weight)

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            inputs (torch.Tensor): Input tensor to the model.

        Returns:
            Dict[str, torch.Tensor]: A dictionary with keys 'detector' and 'classifier', containing
                                    the output from the detector and classifier respectively.
        """
        x = inputs
        for i in range(len(self.conv_blocks)):
            x = self.conv_blocks[f"conv_block_{i}"](x)
            x = self.pool_layers[i](x)
        x = torch.flatten(x, start_dim = 1)

        for i in range(len(self.dense_blocks)):
            x = self.dense_blocks[f"dense_{i}"](x)
        
        output_detector = self.final_dense_detector(x)
        output_classifier = self.final_dense_classifier(x)

        return {'detector': output_detector, 'classifier': output_classifier}
    

if __name__ == "__main__":
    from global_config import logger, cfg, model_cfg
    from pytorch_lightning import Trainer
    from torchsummary import summary

    """Test to see if the model has readable output shapes."""
    detector_label_map = {0: "noise", 1: "event"}
    classifier_label_map = {0:"earthquake", 1:"exlposion"}
    label_maps = {"detector": detector_label_map, "classifier": classifier_label_map}


    input_shape = (3, cfg.live.length*cfg.live.sample_rate + 1)
    detector_class_weight_dict = {"noise": 1.0, "event": 1.0}
    classifier_class_weight_dict = {"earthquake": 1.0, "explosion": 1.0}

    logger.info("Input shape to the model: " + str(input_shape))
    model = get_model(input_shape,
                        detector_metrics_list,
                        classifier_metrics_list,
                        label_map_detector, 
                        label_map_classifier,
                        detector_class_weights, 
                        classifier_class_weights,
                        cfg)

    # Create a trainer
    trainer = Trainer(max_epochs=1)

    # Print model summary
    print(model)

    # Test the model with a batch of random data
    input_data = torch.randn(1, 3, 9601, device="cuda" if torch.cuda.is_available() else "cpu")
    output = model(input_data)
    print(output)
    summary(model, input_size=(3, 9601))


        