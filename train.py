from global_config import logger, cfg
import numpy as np
import os
from Classes.LoadData import LoadData
from Classes.Scaler import Scaler
from Classes.Generator import TrainGenerator
from Classes.Generator import ValidationGenerator
loadData = LoadData()

train_data = loadData.get_train_dataset()
val_data = loadData.get_val_dataset()

train_data = loadData.filter_all_data(train_data)
val_data = loadData.filter_all_data(val_data)

#TODO: Need to taper in generator

scaler = Scaler()
scaler.fit(train_data)

train_gen = TrainGenerator(train_data)
val_gen = ValidationGenerator(val_data)




