import wandb
from global_config import model_cfg, cfg
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from src.Utils import prepare_labels_and_weights, swap_labels, prep_data
from src.Generator import TrainGenerator, ValGenerator
from src.Models import get_model
from src.Metrics import get_least_frequent_class_metrics
from src.Callbacks import InPlaceProgressCallback, WandbLoggingCallback, ValidationConfusionMatrixCallback
import socket
import yaml  # Import yaml to read the sweep_config file

def main(config):
    # Initialize wandb
    wandb.init(config=config)

    # Get the sweep parameters
    sweep_params = wandb.config

    # Update model_cfg and cfg based on the sweep parameters
    for key, value in sweep_params.items():
        if hasattr(model_cfg, key):
            setattr(model_cfg, key, value)
        elif hasattr(cfg, key):
            setattr(cfg, key, value)

    # Rest of your train.py script
    # Initialize the device and data type
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    # Data preparation and model initialization
    train_data, train_labels_dict, val_data, val_labels_dict, label_map, detector_class_weight_dict, classifier_class_weight_dict = prep_data()
    train_gen = TrainGenerator(train_data, train_labels_dict)
    val_gen = ValGenerator(val_data, val_labels_dict)
    input_shape = train_data.shape[1:]
    classifier_metrics = ["f1_score", "precision", "recall", "accuracy"]
    detector_metrics = ["accuracy"]
    model = get_model(detector_label_map, classifier_label_map, detector_metrics, classifier_metrics, detector_class_weight_dict, classifier_class_weight_dict)
    model.build(input_shape=(None, *input_shape))

    # Compile the model
    opt = Adam(learning_rate=model_cfg.optimizer_kwargs.lr, weight_decay=model_cfg.optimizer_kwargs.weight_decay)
    model.compile(optimizer=opt, loss=CategoricalCrossentropy(from_logits=True), metrics='accuracy')

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_total_loss', mode='min', patience=model_cfg.callbacks.early_stopping_patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_total_loss', factor=0.1, patience=model_cfg.callbacks.reduce_lr_patience, mode='min'),
        InPlaceProgressCallback(),
        WandbLoggingCallback()
    ]

    # Model fitting
    model.fit(
        train_gen,
        epochs=model_cfg.optimizer.max_epochs,
        validation_data=val_gen,
        verbose=0,
        callbacks=callbacks
    )

if __name__ == "__main__":
    sweep_config = None
    with open(cfg.paths.sweep_config, 'r') as f:
        sweep_config = yaml.safe_load(f)

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project="ARCES classification")

    # Run the sweep agent
    wandb.agent(sweep_id, function=main)