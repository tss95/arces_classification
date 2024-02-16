from global_config import logger, cfg, model_cfg
import numpy as np
from haikunator import Haikunator

from sklearn.utils.class_weight import compute_class_weight

from src.Utils import prepare_labels_and_weights, swap_labels, prep_data, collate_fn_train
from src.BeamDataset import BeamDataset
from src.Models_torch import get_model
import socket
import os

import wandb 
import torch
from torch.utils.data import DataLoader
from torchsummary import summary
# import the trainer
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# TODO Look into apex for mixed precision training

haikunator = Haikunator()
model_name = f"{cfg.model}_{haikunator.haikunate()}"
if socket.gethostname() != 'saturn.norsar.no':
    config_dict = {}
    for key, value in vars(model_cfg).items():
        config_dict[key] = value
    for key, value in vars(cfg).items():
        config_dict[key] = value
    
    wandb.init(name = model_name, entity="norsar_ai", project="ARCES classification torch", config=config_dict)

train_data, train_labels_dict, val_data, val_labels_dict, label_map, detector_class_weight_dict, classifier_class_weight_dict, classifier_label_map, detector_label_map, date_and_time, train_meta, val_meta, scaler, unswapped_labels = prep_data()
if cfg.data.include_induced: 
    assert 'induced or triggered event' in unswapped_labels, "Oh no, induced or triggered event is not in the labels"
else:
    assert 'induced or triggered event' not in unswapped_labels, "Oh no, induced or triggered event is in the labels"

# Creating the datasets:
train_dataset = BeamDataset(train_data, train_labels_dict, transform=None, scaler=scaler)
val_dataset = BeamDataset(val_data, val_labels_dict, transform=None, scaler=scaler)
# Creating the DataLoaders
train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True, 
                          num_workers=cfg.num_workers, collate_fn=collate_fn_train)
val_loader = DataLoader(val_dataset, batch_size=cfg.data.batch_size, shuffle=False, 
                        num_workers=cfg.num_workers, collate_fn=None)


input_shape = train_data.shape[1:]
logger.info("Input shape to the model: " + str(input_shape))
classifier_metrics = ["f1_score", "precision", "recall", "accuracy"]
detector_metrics = ["accuracy"]
model = get_model(input_shape, detector_label_map, classifier_label_map, 
                  detector_class_weight_dict, classifier_class_weight_dict)
summary(model, input_size = input_shape)


# Create a trainer
if cfg.data.debug:
    max_epochs = 1

wandb_logger = WandbLogger(name=model_name, entity="norsar_ai", project="ARCES classification torch", config=config_dict)
callbacks = []

checkpoint_callback = ModelCheckpoint(monitor='val_total_loss', mode='min')
callbacks.append(checkpoint_callback)


trainer = pl.Trainer(
    max_epochs=cfg.optimizer.max_epochs,
    devices=-1 if torch.cuda.is_available() else 0,
    precision=16 ,
    callbacks=callbacks
)

trainer.fit(model, train_dataloaders= train_loader, val_dataloaders=val_loader)


"""
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
"""