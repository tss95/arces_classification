from global_config import model_cfg, logger
from src.Loop_torch import Loop
import numpy as np
#from nais.Models import AlexNet1D
import torch
import pytorch_lightning as pl
from torch.nn import functional as F
import torch.nn as nn
from typing import Dict, List, Callable
import torch.nn.init as init
from torchsummary import summary


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
    if cfg.model_name == "cnn_dense":
        model = CNN_dense(input_shape,
                        detector_metrics_list,
                        classifier_metrics_list,
                        label_map_detector, 
                        label_map_classifier,
                        detector_class_weights, 
                        classifier_class_weights,
                        cfg)
    if cfg.model_name == "alexnet":
        model = AlexNet1D(input_shape,
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
    
    
class AlexNet1D(Loop):
    def __init__(self, 
                 input_shape,
                 detector_metrics_list,
                classifier_metrics_list,
                label_map_detector, 
                label_map_classifier,
                detector_class_weights, 
                classifier_class_weights,
                cfg,
                kernel_sizes=None, 
                filters=None, 
                pooling='max'):
        super(AlexNet1D, self).__init__(input_shape,
                                        detector_metrics_list,
                                        classifier_metrics_list,
                                        label_map_detector, 
                                        label_map_classifier,
                                        detector_class_weights, 
                                        classifier_class_weights,
                                        cfg)
        num_channels = input_shape[0]
        
        if kernel_sizes is None:
            kernel_sizes = [11, 5, 3, 3, 3]
        if filters is None:
            filters = [96, 256, 384, 384, 256]
        
        self.conv_blocks = nn.ModuleDict()
        self.pool_layers = nn.ModuleDict()

        pooling_layer = nn.MaxPool1d if pooling == 'max' else nn.AvgPool1d
        if pooling in [None, 'none']:
            pooling_layer = lambda kernel_size, stride: nn.Identity()

        in_channels = num_channels
        for i, (filter_size, kernel_size) in enumerate(zip(filters, kernel_sizes)):
            padding = (kernel_size - 1) // 2
            self.conv_blocks[f"conv_block_{i}"] = nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=filter_size, kernel_size=kernel_size, stride=4 if i == 0 else 1, padding=padding if i != 0 else 0),
                nn.ReLU(),
                nn.BatchNorm1d(filter_size, eps=1e-5),
                )
            if i < len(filters) - 1:
                self.pool_layers[f"pool_layer_{i}"] = pooling_layer(kernel_size=3, stride=2)
            in_channels = filter_size

        # Final pooling layer to ensure consistency in tensor shape before flattening
        self.pool_layers["pool_layer_before_flattening"] = pooling_layer(kernel_size=3, stride=2)

        self.flatten = nn.Flatten()
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)  # Create a dummy input of the correct shape
            output_size = self._calculate_conv_output_size(dummy_input)

        # Define dense (fully connected) blocks for shared features
        self.dense_blocks = nn.ModuleDict({
            "dense_0": nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(output_size, 4096),  # Placeholder for actual calculation
                nn.ReLU(),
                nn.Dropout(0.5)
            ),
            "dense_1": nn.Sequential(
                nn.Linear(4096, 4096),
                nn.ReLU()
            )
        })

        # Output layers for detector and classifier without activation functions
        self.final_dense_detector = nn.Linear(4096, 1)
        self.final_dense_classifier = nn.Linear(4096, 1)
        self._initialize_weights()
        
    def _calculate_conv_output_size(self, x):
        # Simulate a forward pass through the conv and pool layers
        for i in range(len(self.conv_blocks)):
            x = self.conv_blocks[f"conv_block_{i}"](x)
            if i < len(self.pool_layers) - 1:
                x = self.pool_layers[f"pool_layer_{i}"](x)
        x = self.pool_layers["pool_layer_before_flattening"](x)
        x = self.flatten(x)
        return x.numel()  # Return the total number of elements in the flattened output

    
    def forward(self, inputs):
        x = inputs
        for i in range(len(self.conv_blocks)):
            x = self.conv_blocks[f"conv_block_{i}"](x)
            if i < len(self.pool_layers) - 1:
                x = self.pool_layers[f"pool_layer_{i}"](x)
        x = self.pool_layers["pool_layer_before_flattening"](x)
        x = self.flatten(x)

        for i in range(len(self.dense_blocks)):
            x = self.dense_blocks[f"dense_{i}"](x)
        
        output_detector = self.final_dense_detector(x)
        output_classifier = self.final_dense_classifier(x)

        return {'detector': output_detector, 'classifier': output_classifier}
        
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                init.constant_(m.bias, 0)
                
    def verbose_output(self, tensor, label):
        # Check for NaN values in the tensor
        has_nan = torch.isnan(tensor).any().item()
        
        # Print statistics about the tensor
        print(f"{label} - Min: {tensor.min().item()}, Max: {tensor.max().item()}, Mean: {tensor.mean().item()}, Has NaN: {has_nan}")
        
        """
    def forward(self, inputs):
        x = inputs
        self.verbose_output(x, "Input to Model")

        # Iterate through each conv block
        for i, conv_block in enumerate(self.conv_blocks.values()):
            # Assuming conv_block is an instance of nn.Sequential or similar
            for j, layer in enumerate(conv_block):
                x = layer(x)
                self.verbose_output(x, f"Conv Block {i}, Layer {j}")
            
            if i < len(self.pool_layers) - 1:
                x = self.pool_layers[f"pool_layer_{i}"](x)
                self.verbose_output(x, f"After Pool Layer {i}")
        x = self.pool_layers["pool_layer_before_flattening"](x)
        self.verbose_output(x, "Pool before Flattening")
        x = self.flatten(x)
        self.verbose_output(x, "After Flattening")

        # Iterate through each dense block
        for i, dense_block in enumerate(self.dense_blocks.values()):
            # Assuming dense_block is an instance of nn.Sequential or similar
            for j, layer in enumerate(dense_block):
                x = layer(x)
                self.verbose_output(x, f"Dense Block {i}, Layer {j}")
        
        output_detector = self.final_dense_detector(x)
        output_classifier = self.final_dense_classifier(x)

        self.verbose_output(output_detector, "Output Detector")
        self.verbose_output(output_classifier, "Output Classifier")

        return {'detector': output_detector, 'classifier': output_classifier}
"""
        

if __name__ == "__main__":
    from global_config import logger, cfg, model_cfg
    from pytorch_lightning import Trainer
    from torchsummary import summary

    """Test to see if the model has readable output shapes."""
    detector_label_map = {0: "noise", 1: "event"}
    classifier_label_map = {0:"earthquake", 1:"exlposion"}
    label_maps = {"detector": detector_label_map, "classifier": classifier_label_map}


    detector_class_weight_dict = {"noise": 1.0, "event": 1.0}
    classifier_class_weight_dict = {"earthquake": 1.0, "explosion": 1.0}
    detector_metrics_list = ["auroc","accuracy"]
    classifier_metrics_list = ["auroc","accuracy"]
    input_data = torch.randn((3, 4000), device="cuda" if torch.cuda.is_available() else "cpu")
    
    

    logger.info("Input shape to the model: " + str(input_data.shape))
    model = get_model(input_data.shape,
                        detector_metrics_list,
                        classifier_metrics_list,
                        detector_label_map, 
                        classifier_label_map,
                        detector_class_weight_dict, 
                        classifier_class_weight_dict,
                        cfg)
    summary(model, input_size=(input_data.shape))
    
    
    # Create a trainer
    trainer = Trainer(max_epochs=1)

    # Print model summary
    dummy_labels = {"detector": torch.tensor([1]), "classifier": torch.tensor([0])}
    # Test the model with a batch of random data
    input_data = input_data.unsqueeze(0)
    print(f"batch_of_data shape: {input_data.shape}")
    trainer.fit(model, torch.tensor(input_data), dummy_labels)


        