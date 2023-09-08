from global_config import logger, cfg, model_cfg
import numpy as np
from Classes.LoadData import LoadData
from Classes.Scaler import Scaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime

from Classes.Utils import prepare_labels, swap_labels
from Classes.Generator import TrainGenerator
from Classes.Generator import ValGenerator
from Classes.Model import get_model
from Classes.Metrics import get_least_frequent_class_metrics
from Classes.UMAPCallback import UMAPCallback
from Classes.Analysis import Analysis
from Classes.MetricsCallback import MetricsCallback
import socket

import wandb 
from wandb.keras import WandbMetricsLogger

from tensorflow.keras import mixed_precision
from tensorflow.config.experimental import list_physical_devices, set_memory_growth



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

        

# Prepare labels for training data
train_detector_labels_onehot, train_classifier_labels_onehot, label_encoder = prepare_labels(train_labels)

# Prepare labels for validation data using the same label_encoder
val_detector_labels_onehot, val_classifier_labels_onehot, _ = prepare_labels(val_labels, label_encoder=label_encoder)

# Create a translational dictionary for label strings based on training data
detector_label_map = {index: label for index, label in enumerate(label_encoder['detector'].classes_)}
classifier_label_map = {index: label for index, label in enumerate(label_encoder['classifier'].classes_)}

# Calculate class weights for detector and classifier based on training data
detector_class_weights = compute_class_weight('balanced', classes=np.unique(label_encoder['detector'].classes_), y=label_encoder['detector'].transform(detector_labels))
classifier_class_weights = compute_class_weight('balanced', classes=np.unique(label_encoder['classifier'].classes_), y=label_encoder['classifier'].transform(classifier_labels))

detector_class_weight_dict = {i: detector_class_weights[i] for i in range(len(detector_class_weights))}
classifier_class_weight_dict = {i: classifier_class_weights[i] for i in range(len(classifier_class_weights))}

# Combining the detector and classifier labels into a single array
train_labels_combined = np.stack([train_labels_detector, train_labels_classifier], axis=-1)

# Similarly for validation data
val_labels_combined = np.stack([val_labels_detector, val_labels_classifier], axis=-1)


train_gen = TrainGenerator(train_data, train_labels_combined)
val_gen = ValGenerator(val_data, val_labels_combined)

#metrics = get_least_frequent_class_metrics(train_labels_onehot, label_map, 
#                                           sample_weight = class_weights, 
#                                           metrics_list = ['accuracy','f1_score', 'precision', 'recall'])
input_shape = train_data.shape[1:]
logger.info("Input shape to the model: " + str(input_shape))
model = get_model(num_classes = 3)
model.build(input_shape=(None, *input_shape))  # Explicitly building the model here
opt = Adam(learning_rate=cfg.optimizer.optimizer_kwargs.lr, weight_decay=cfg.optimizer.optimizer_kwargs.weight_decay)
model.compile(optimizer=opt, loss=CategoricalCrossentropy(from_logits = True), metrics='accuracy')
model.summary()

analysis = Analysis(model, val_data, val_labels_onehot, label_map, date_and_time)
analysis.collect_and_plot_samples(train_gen, train_meta)


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

umap_callback = UMAPCallback(val_data, 
                             val_labels_onehot, 
                             label_map, 
                             interval=cfg.callbacks.umap_interval
                             )
callbacks.append(umap_callback)
metrics_callback = MetricsCallback(val_labels_onehot, label_map)
callbacks.append(metrics_callback)

#if socket.gethostname() != 'saturn.norsar.no':
#    wandbCallback = WandbMetricsLogger(list(metrics.keys()))
#    callbacks.append(wandbCallback)



print("Class weights dictionary:", class_weight_dict)

model.fit(
    train_gen, 
    epochs=cfg.optimizer.max_epochs, 
    validation_data=val_gen, 
    callbacks=callbacks, 
    class_weight=class_weight_dict
)

analysis.main()