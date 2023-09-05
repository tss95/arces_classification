from global_config import logger, cfg, model_cfg
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from Classes.LoadData import LoadData
from Classes.Scaler import Scaler
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime

from Classes.Generator import TrainGenerator
from Classes.Generator import ValGenerator
from Classes.Model import get_model
from Classes.Metrics import get_metrics
from Classes.UMAPCallback import UMAPCallback

import wandb 
from wandb.keras import WandbCallback

config_dict = {}
for key, value in vars(model_cfg).items():
    config_dict[key] = value
for key, value in vars(cfg).items():
    config_dict[key] = value
wandb.init(name=cfg.model, entity="norsar_ai", project="ARCES classification", config=config_dict)
now = datetime.now()
date_and_time = now.strftime("%Y%m%d_%H%M%S")

loadData = LoadData()

train_data = loadData.get_train_dataset()
val_data = loadData.get_val_dataset()

train_dataset = loadData.filter_all_data(train_data)
val_dataset = loadData.filter_all_data(val_data)



train_data, train_labels, train_meta = train_dataset
val_data, val_labels, val_meta = val_dataset

scaler = Scaler()
scaler.fit(train_data)
train_data = scaler.transform(train_data)
val_data = scaler.transform(val_data)

# Step 1: One-hot encoding of labels
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
val_labels_encoded = label_encoder.transform(val_labels)

train_labels_onehot = to_categorical(train_labels_encoded)
val_labels_onehot = to_categorical(val_labels_encoded)

# Step 2: Create a translational dictionary for label strings
label_map = {index: label for index, label in enumerate(label_encoder.classes_)}

# Step 3: Convert to tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels_onehot))
val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels_onehot))

train_gen = TrainGenerator(train_dataset)
val_gen = ValGenerator(val_dataset)

metrics = get_metrics(train_labels_onehot, label_map, ['accuracy', 'f1_score', 'recall', 'precision'])

input_shape = train_dataset.element_spec[0].shape
model = get_model()
model = model.build(input_shape = input_shape[1:])
model.compile(optimizer = cfg.optimizer.optimizer, loss = tf.keras.losses.CategoricalCrossentropy(), metrics = metrics)

# Callbacks
early_stopping = EarlyStopping(
    monitor='f1_score', 
    patience=cfg.callbacks.early_stopping_patience,  # number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # restore the best weights saved when stopping
)

model_checkpoint = ModelCheckpoint(
    filepath=f'{cfg.paths.model_save_folder}/{model}_{date_and_time}.h5', 
    monitor='f1_score',
    save_best_only=True  # only save the best model
)

umap_callback = UMAPCallback(val_data, val_labels_onehot, label_map, interval=cfg.callbacks.umap_interval)
callbacks = [WandbCallback(), early_stopping, model_checkpoint, umap_callback]
# Calculate class weights
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels_encoded), y=train_labels_encoded)

# Create a dictionary to pass it to the training configuration
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

print("Class weights dictionary:", class_weight_dict)



model.fit(
    train_gen, 
    epochs=cfg.optimizer.max_epochs, 
    validation_data=val_gen, 
    callbacks=callbacks, 
    class_weight=class_weight_dict
)






