from global_config import logger, cfg, model_cfg
import numpy as np
from Classes.LoadData import LoadData
from Classes.Scaler import Scaler
import tensorflow as tf
from tf.keras.utils import to_categorical
from tf.keras.losses import CategoricalCrossentropy
from tf.keras.callbacks import EarlyStopping, ModelCheckpoint
from tf.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime

from Classes.Utils import prepare_labels_and_weights, swap_labels
from Classes.Generator import TrainGenerator
from Classes.Generator import ValGenerator
from Classes.Models import get_model
from Classes.Metrics import get_least_frequent_class_metrics
from Classes.UMAPCallback import UMAPCallback
from Classes.Analysis import Analysis
from Classes.MetricsCallback import MetricsCallback
import socket

import wandb 
from wandb.keras import WandbMetricsLogger

from tf.keras import mixed_precision
from tf.config.experimental import list_physical_devices, set_memory_growth



gpus = list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
    mixed_precision.set_global_policy('mixed_float16')


if socket.gethostname() != 'saturn.norsar.no':
    config_dict = {}
    for key, value in vars(model_cfg).items():
        config_dict[key] = value
    for key, value in vars(cfg).items():
        config_dict[key] = value
    wandb.init(name=cfg.model, entity="norsar_ai", project="ARCES classification", config=config_dict)

now = datetime.now()
date_and_time = now.strftime("%Y%m%d_%H%M%S")

loadData = LoadData()
train_dataset = loadData.get_train_dataset()
val_dataset = loadData.get_val_dataset()

train_data, train_labels, train_meta = train_dataset[0],train_dataset[1],train_dataset[2]
logger.info("Train data shape: " + str(train_data.shape))
train_data = np.transpose(train_data, (0,2,1))
logger.info("Train data shape after transpose: " + str(train_data.shape))
val_data, val_labels, val_meta = val_dataset[0],val_dataset[1],val_dataset[2]
val_data = np.transpose(val_data, (0,2,1))



train_labels = swap_labels(train_labels)
val_labels = swap_labels(val_labels)

print(np.unique(train_labels))
print(np.unique(val_labels))

logger.info(f"Before scaling training shape is {train_data.shape}, with (min, max) ({np.min(train_data)}, {np.max(train_data)})")
logger.info(f"Before scaling validation shape is {val_data.shape}, with (min, max) ({np.min(val_data)}, {np.max(val_data)})")

scaler = Scaler()
scaler.fit(train_data)
train_data = scaler.transform(train_data)
val_data = scaler.transform(val_data)

logger.info(f"After scaling training shape is {train_data.shape}, with (min, max) ({np.min(train_data)}, {np.max(train_data)})")
logger.info(f"After scaling validation shape is {val_data.shape}, with (min, max) ({np.min(val_data)}, {np.max(val_data)})")

        

# Prepare labels for training and validation data
nested_label_dict_train, detector_class_weights_train, classifier_class_weights_train, label_encoder_train, detector_label_map, classifier_label_map = prepare_labels_and_weights(train_labels)
nested_label_dict_val, _, _, _, _, _ = prepare_labels_and_weights(val_labels, label_encoder=label_encoder_train)

logger.info(f"Label encoder:{label_encoder_train}")

# Create a translational dictionary for label strings based on training data
detector_label_map = {index: label for index, label in enumerate(label_encoder_train['detector'].classes_)}
classifier_label_map = {index: label for index, label in enumerate(label_encoder_train['classifier'].classes_)}

# Convert class weights to dictionaries
detector_class_weight_dict = {i: detector_class_weights_train[i] for i in range(len(detector_class_weights_train))}
classifier_class_weight_dict = {i: classifier_class_weights_train[i] for i in range(len(classifier_class_weights_train))}

# Combine the detector and classifier labels into single nested dictionaries
# This should already be done by prepare_labels_and_weights as it returns nested_label_dict_train and nested_label_dict_val

logger.info(f"Detector label map: {detector_label_map}")
logger.info(f"Classifier label map: {classifier_label_map}")
logger.info(f"Detector class weights: {detector_class_weight_dict}")
logger.info(f"Classifier class weights: {classifier_class_weight_dict}")



# Convert nested dictionaries to NumPy arrays
detector_labels_array_train = np.argmax(np.array([nested_label_dict_train[i]['detector'] for i in range(len(nested_label_dict_train))]), axis=1)
classifier_labels_array_train = np.argmax(np.array([nested_label_dict_train[i]['classifier'] for i in range(len(nested_label_dict_train))]), axis=1)

detector_labels_array_val = np.argmax(np.array([nested_label_dict_val[i]['detector'] for i in range(len(nested_label_dict_val))]), axis=1)
classifier_labels_array_val = np.argmax(np.array([nested_label_dict_val[i]['classifier'] for i in range(len(nested_label_dict_val))]), axis=1)



train_labels_dict = {'detector': detector_labels_array_train[:, np.newaxis], 'classifier': classifier_labels_array_train[:, np.newaxis]}
val_labels_dict = {'detector': detector_labels_array_val[:, np.newaxis], 'classifier': classifier_labels_array_val[:, np.newaxis]}

logger.info(f"3 first classifier labels: {val_labels_dict['classifier'][:3]}")
logger.info(f"3 first detector labels: {val_labels_dict['detector'][:3]}")

# Now, use these arrays to create your data generators
train_gen = TrainGenerator(train_data, train_labels_dict)
val_gen = ValGenerator(val_data, val_labels_dict)

#metrics = get_least_frequent_class_metrics(train_labels_onehot, label_map, 
#                                           sample_weight = class_weights, 
#                                           metrics_list = ['accuracy','f1_score', 'precision', 'recall'])
input_shape = train_data.shape[1:]
logger.info("Input shape to the model: " + str(input_shape))
classifier_metrics = ["f1_score", "precision", "recall", "accuracy"]
detector_metrics = ["accuracy"]
model = get_model(detector_label_map, classifier_label_map, detector_metrics, classifier_metrics, 
                  detector_class_weight_dict, classifier_class_weight_dict)
model.build(input_shape=(None, *input_shape))  # Explicitly building the model here
opt = Adam(learning_rate=cfg.optimizer.optimizer_kwargs.lr, weight_decay=cfg.optimizer.optimizer_kwargs.weight_decay)
model.compile(optimizer=opt, loss=CategoricalCrossentropy(from_logits = True), metrics='accuracy')
model.summary()

#analysis = Analysis(model, val_data, val_labels_onehot, label_map, date_and_time)
#analysis.collect_and_plot_samples(train_gen, train_meta)


# Callbacks
callbacks = []
early_stopping = EarlyStopping(
    monitor='f1_score', 
    patience=cfg.callbacks.early_stopping_patience,  # number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # restore the best weights saved when stopping
)
callbacks.append(early_stopping)

model_checkpoint = ModelCheckpoint(
    filepath=f'{cfg.paths.model_save_folder}/{model}_{date_and_time}.h5', 
    monitor='f1_score',
    save_best_only=True  # only save the best model
    )
callbacks.append(model_checkpoint)

if cfg.callbacks.umap:
    umap_callback = UMAPCallback(val_data, 
                                val_labels_onehot, 
                                label_map, 
                                interval=cfg.callbacks.umap_interval
                                )
    callbacks.append(umap_callback)
#metrics_callback = MetricsCallback(val_labels_onehot, label_map)
#callbacks.append(metrics_callback)

#if socket.gethostname() != 'saturn.norsar.no':
#    wandbCallback = WandbMetricsLogger(list(metrics.keys()))
#    callbacks.append(wandbCallback)




model.fit(
    train_gen, 
    epochs=cfg.optimizer.max_epochs, 
    val_generator=val_gen, 
    callbacks=callbacks, 
)

analysis = Analysis(model, val_gen, val_labels_dict, date_and_time)

analysis.main()