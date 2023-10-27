from global_config import logger, cfg, model_cfg
import numpy as np
from haikunator import Haikunator

import tensorflow as tf
tf.get_logger().setLevel('DEBUG')
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight

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
from tensorflow.config.experimental import list_physical_devices, set_memory_growth



gpus = list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
    mixed_precision.set_global_policy('mixed_float16')

haikunator = Haikunator()
model_name = f"{cfg.model}_{haikunator.haikunate()}"
if socket.gethostname() != 'saturn.norsar.no':
    config_dict = {}
    for key, value in vars(model_cfg).items():
        config_dict[key] = value
    for key, value in vars(cfg).items():
        config_dict[key] = value
    
    wandb.init(name = model_name, entity="norsar_ai", project="ARCES classification", config=config_dict)

train_data, train_labels_dict, val_data, val_labels_dict, label_map, detector_class_weight_dict, classifier_class_weight_dict, classifier_label_map, detector_label_map, date_and_time, train_meta, val_meta, scaler, unswapped_labels = prep_data()
if cfg.data.include_induced: 
    assert 'induced or triggered event' in unswapped_labels, "Oh no, induced or triggered event is not in the labels"
else:
    assert 'induced or triggered event' not in unswapped_labels, "Oh no, induced or triggered event is in the labels"
# Now, use these arrays to create your data generators
train_gen = TrainGenerator(train_data, train_labels_dict, scaler)
val_gen = ValGenerator(val_data, val_labels_dict, scaler)

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

# Callbacks
callbacks = []
early_stopping = EarlyStopping(
    monitor='val_total_loss', 
    verbose = True,
    mode="min",
    patience=cfg.callbacks.early_stopping_patience,  # number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # restore the best weights saved when stopping
)
callbacks.append(early_stopping)


if cfg.callbacks.umap:
    umap_callback = UMAPCallback(val_data, 
                                val_labels_dict, 
                                label_map, 
                                interval=cfg.callbacks.umap_interval
                                )
    callbacks.append(umap_callback)

reduceLR = ReduceLROnPlateau(monitor ='val_total_loss',
                            factor = 0.1,
                            verbose = True,
                            patience = cfg.callbacks.reduce_lr_patience,
                            mode='min')
callbacks.append(reduceLR)

valConfCallback = ValidationConfusionMatrixCallback(val_gen, label_map, unswapped_labels)
callbacks.append(valConfCallback)
callbacks.append(InPlaceProgressCallback())
callbacks.append(WandbLoggingCallback())
#callbacks.append(MetricsCallback(val_gen, label_map))

model.fit(
    train_gen, 
    epochs=cfg.optimizer.max_epochs, 
    validation_data=val_gen, 
    verbose=0,
    callbacks=callbacks
)
val_gen.on_epoch_end()

model_save_path = f"{cfg.paths.model_save_folder}/{model_name}_{date_and_time}_model_weights.h5"
model.save_weights(model_save_path)
logger.info(f"Model weights stored at: {model_save_path}")

def check_valgen(val_gen, model):
    counts = 0
    for batch_data, batch_labels in val_gen:
        pred_probs = model.predict(batch_data)
        
        # Counts for detector labels
        unique, counts_detector = np.unique(np.argmax(pred_probs["detector"], axis=1), return_counts=True)
        detector_dist = dict(zip(unique, counts_detector))
        
        # Counts for classifier labels
        unique, counts_classifier = np.unique(np.argmax(pred_probs["classifier"], axis=1), return_counts=True)
        classifier_dist = dict(zip(unique, counts_classifier))
        
        n_nn = len(pred_probs["detector"])
        n_ee = len(pred_probs["classifier"])
        
        print(f"n_nn {n_nn}, n_ee {n_ee}")
        print(f"Detector label distribution per batch: {detector_dist}")
        print(f"Classifier label distribution per batch: {classifier_dist}")
        
        # Counts for true labels in the batch
        unique, counts_true_detector = np.unique(np.argmax(batch_labels['detector'], axis=1), return_counts=True)
        true_detector_dist = dict(zip(unique, counts_true_detector))
        
        unique, counts_true_classifier = np.unique(np.argmax(batch_labels['classifier'], axis=1), return_counts=True)
        true_classifier_dist = dict(zip(unique, counts_true_classifier))
        
        print(f"True Detector label distribution per batch: {true_detector_dist}")
        print(f"True Classifier label distribution per batch: {true_classifier_dist}")
        
        counts += n_nn

    print(f"Total number of predictions from val_gen: {counts}")

if cfg.data.debug:
    check_valgen(val_gen, model)



logger.info(f"Length of val gen: {len(val_gen)}")


analysis = Analysis(model, val_gen, label_map, date_and_time)

#analysis.main()
#analysis.collect_and_plot_samples(val_gen, val_meta)
if cfg.data.include_induced:
    analysis.explore_induced_events(val_gen, val_meta, unswapped_labels)
analysis.explore_regular_events(val_gen, val_meta, "noise")
analysis.explore_regular_events(val_gen, val_meta, "earthquake")
analysis.explore_regular_events(val_gen, val_meta, "explosion")
