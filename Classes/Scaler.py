# Your existing imports and setup here
from global_config import logger, cfg
import pandas as pd
import numpy as np


class Scaler:
    
    def __init__(self)
        self.global_or_local = cfg.scaling.global_or_local
        self.per_channel = cfg.scaling.per_channel
        self.scaler_type = cfg.scaling.scaler_type
        

    def parameter_validation(self):
        # Validate scaler_type
        if self.scaler_type not in (valid_scalers:=["minmax", "standard", "robust"]):
            raise Exception(f"Invalid scaler type ({self.scaler_type}). Must be one of {str(valid_scalers)}")
        # Validate per_channel
        if not isinstance(self.per_channel, bool):
            raise Exception(f"Invalid type for per_channel ({type(self.per_channel).__name__}). Must be a boolean.")
        # Validate global_or_local
        if self.global_or_local not in (valid_global_or_local:=["local", "global"]):
            raise Exception(f"Invalid value for global_or_local ({self.global_or_local}). Must be one of {str(valid_global_or_local)}")
        
    def fit(self, X):
        pass
    
    def transform(self, X):
        pass
    
    def print_parameters(self):
        print("")
    

class MinMaxScaler(Scaler):
    def __init__(self):
        super().__init__()
    
    def fit(self, X):
        if self.global_or_local == "global":
    
    