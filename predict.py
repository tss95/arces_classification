from global_config import logger, cfg, model_cfg
import numpy as np
import os

import tensorflow as tf
tf.get_logger().setLevel('DEBUG')

from Classes.Utils import prepare_labels_and_weights, swap_labels, prep_data
from Classes.Generator import TrainGenerator
from Classes.Generator import ValGenerator
from Classes.Models import get_model
from Classes.Metrics import get_least_frequent_class_metrics
from Classes.UMAPCallback import UMAPCallback
from Classes.Analysis import Analysis
from Classes.MetricsCallback import MetricsCallback
from Classes.Callbacks import InPlaceProgressCallback, WandbLoggingCallback, ValidationConfusionMatrixCallback
import socket

import wandb 
from wandb.keras import WandbMetricsLogger

from tensorflow.keras import mixed_precision
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.config.experimental import list_physical_devices, set_memory_growth

gpus = list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
    mixed_precision.set_global_policy('mixed_float16')

train_data, train_labels_dict, val_data, val_labels_dict, label_map, detector_class_weight_dict, classifier_class_weight_dict, classifier_label_map, detector_label_map, date_and_time, train_meta, val_meta, scaler, unswapped_labels = prep_data()
if cfg.data.include_induced: 
    assert 'induced or triggered event' in unswapped_labels, "Oh no, induced or triggered event is not in the labels"
else:
    assert 'induced or triggered event' not in unswapped_labels, "Oh no, induced or triggered event is in the labels"
# Now, use these arrays to create your data generators
train_gen = TrainGenerator(train_data, train_labels_dict, scaler)
val_gen = ValGenerator(val_data, val_labels_dict, scaler)

input_shape = train_data.shape[1:]
logger.info("Input shape to the model: " + str(input_shape))
classifier_metrics = [None]
detector_metrics = [None]
model = get_model(detector_label_map, classifier_label_map, detector_metrics, classifier_metrics, 
                   detector_class_weight_dict, classifier_class_weight_dict)
model.build(input_shape=(None, *input_shape))  # Explicitly building the model here

model.load_weights(os.path.join(cfg.paths.model_save_folder, cfg.model_name))
logger.info(f"Loaded model weights from {os.path.join(cfg.paths.model_save_folder, cfg.model_name)}")

analysis = Analysis(model, val_gen, label_map, date_and_time)

#analysis.main()
#analysis.collect_and_plot_samples(val_gen, val_meta)
analysis.plot_d_mistakes_by_msdr(val_meta)
analysis.plot_dnc_mistakes_by_msrd(val_meta)
#analysis.plot_dnc_mistakes_by_distance(val_meta)
#analysis.plot_d_mistakes_by_distance(val_meta)
#analysis.plot_events_on_map(val_meta)


#if cfg.data.include_induced:
#    analysis.explore_induced_events(val_gen, val_meta, unswapped_labels)
#analysis.explore_regular_events(val_gen, val_meta, "noise")
#analysis.explore_regular_events(val_gen, val_meta, "earthquake")
#analysis.explore_regular_events(val_gen, val_meta, "explosion")
