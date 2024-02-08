# Your existing imports and setup here
from global_config import logger, cfg
import numpy as np
import torch
import pickle 
import os
from typing import Any, Union

class Scaler:
    # TODO: Memory leak present. If you need to run transform multiple times 
    # (i.e. you cannot hold all of your data in memory) resolve the leak issue
    def __init__(self):
        """
        Initializes the Scaler object with configuration settings.
        """
        self.global_or_local = cfg.scaling.global_or_local
        self.per_channel = cfg.scaling.per_channel
        self.scaler_type = cfg.scaling.scaler_type
        self.parameter_validation()
        self.scaler = self.get_scaler()

        
    def get_scaler(self):
        """
        Determines and returns the appropriate scaler based on the scaler_type.

        Returns:
            Union[MinMaxScaler, StandardScaler, RobustScaler, LogScaler]: An instance of the selected scaler.
        """
        if self.scaler_type == "minmax":
            return MinMaxScaler()
        if self.scaler_type == "standard":
            return StandardScaler()
        if self.scaler_type == "log":
            return LogScaler()
        raise NotImplementedError("Only minmax, standard and log scalers are currently implemented")
        

    def parameter_validation(self):
        """
        Validates the parameters of the scaler configuration.
        """
        # Validate scaler_type
        if self.scaler_type not in (valid_scalers:=["minmax", "standard", "robust", "log"]):
            raise Exception(f"Invalid scaler type ({self.scaler_type}). Must be one of {str(valid_scalers)}")
        # Validate per_channel
        if not isinstance(self.per_channel, bool):
            raise Exception(f"Invalid type for per_channel ({type(self.per_channel).__name__}). Must be a boolean.")
        # Validate global_or_local
        if self.global_or_local not in (valid_global_or_local:=["local", "global"]):
            raise Exception(f"Invalid value for global_or_local ({self.global_or_local}). Must be one of {str(valid_global_or_local)}")

    def fit(self, X: Any):
        """
        Fits the scaler to the data.

        Args:
            X (Any): The data to fit the scaler on.
        """
        logger.info("Fitting scaler.")
        self.scaler.fit(X)
        logger.info("Scaler fitted.")

    def load_fitted(self):
        """
        Loads a previously fitted scaler from a file.
        """
        logger.info("Loading fitted scaler.")
        scaler_path = os.path.join(cfg.paths.scaler_path, f"{self.scaler_type}_{self.global_or_local}.pkl")
        self.scaler = pickle.load(open(scaler_path, "rb"))
        logger.info("Scaler loaded.")

    def save_scaler(self):
        """
        Saves the fitted scaler to a file.
        """
        logger.info("Saving fitted scaler.")
        scaler_path = os.path.join(cfg.paths.scaler_path, f"{self.scaler_type}_{self.global_or_local}.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Scaler saved to {scaler_path}.")

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Transforms the data using the fitted scaler.

        Args:
            X (Any): The data to be transformed.

        Returns:
            Any: The transformed data.
        """
        return self.scaler.transform(X)


class MinMaxScaler(Scaler):
    def __init__(self):
        pass

    def fit(self, X: torch.Tensor):
        if cfg.scaling.global_or_local == "global":
            x1 = X[:,:,0]
            x2 = X[:,:,1]
            x3 = X[:,:,2]
            max_1, max_2, max_3 = torch.max(x1), torch.max(x2), torch.max(x3)
            min_1, min_2, min_3 = torch.min(x1), torch.min(x2), torch.min(x3)

            if cfg.scaling.per_channel:    
                self.maxs = torch.tensor([max_1, max_2, max_3])
                self.mins = torch.tensor([min_1, min_2, min_3])
            else:
                self.maxs = torch.tensor([max_1, max_2, max_3]).max()
                self.mins = torch.tensor([min_1, min_2, min_3]).min()
        if cfg.scaling.global_or_local == "local":
            logger.info("Local minmax; skipping fit")

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        X = X.float()
        transformed_X = torch.empty_like(X)

        if cfg.scaling.global_or_local == "global":
            if cfg.scaling.per_channel:
                transformed_X = torch.stack([
                    (X[:,:,0] - self.mins[0]) / (self.maxs[0] - self.mins[0]),
                    (X[:,:,1] - self.mins[1]) / (self.maxs[1] - self.mins[1]),
                    (X[:,:,2] - self.mins[2]) / (self.maxs[2] - self.mins[2])
                ], dim=-1)
            else:
                transformed_X = (X - self.mins) / (self.maxs - self.mins)

        if cfg.scaling.global_or_local == "local":
            transformed_list = []
            for idx in range(X.shape[0]):
                x = X[idx]
                if cfg.scaling.per_channel:
                    transformed_x = torch.stack([
                        (x[:,0] - torch.min(x[:,0])) / (torch.max(x[:,0]) - torch.min(x[:,0])),
                        (x[:,1] - torch.min(x[:,1])) / (torch.max(x[:,1]) - torch.min(x[:,1])),
                        (x[:,2] - torch.min(x[:,2])) / (torch.max(x[:,2]) - torch.min(x[:,2]))
                    ], dim=-1)
                else:
                    transformed_x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
                transformed_list.append(transformed_x)
            transformed_X = torch.stack(transformed_list, dim=0)

        return transformed_X

        

class StandardScaler(Scaler):
    def __init__(self):
        pass

    def fit(self, X: torch.Tensor):
        """
        Fits the scaler to the data for Standard scaling.

        Args:
            X (np.ndarray): The data to fit the scaler on.
        """
        if cfg.scaling.global_or_local == "global":
            x1 = X[:,:,0]
            x2 = X[:,:,1]
            x3 = X[:,:,2]
            mean_1, mean_2, mean_3 = torch.mean(x1), torch.mean(x2), torch.mean(x3)
            std_1, std_2, std_3 = torch.std(x1), torch.std(x2), torch.std(x3)

            if cfg.scaling.per_channel:    
                self.means = torch.tensor([mean_1, mean_2, mean_3])
                self.stds = torch.tensor([std_1, std_2, std_3])
            else:
                self.means = torch.tensor([mean_1, mean_2, mean_3]).mean()
                self.stds = torch.tensor([std_1, std_2, std_3]).mean()
        if cfg.scaling.global_or_local == "local":
            logger.info("Local standard; skipping fit")

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Transforms the data using the Standard scaling approach.

        Args:
            X (torch.Tensor): The data to be transformed.

        Returns:
            torch.Tensor: The transformed data.
        """
        transformed_X = torch.empty_like(X)
        if cfg.scaling.global_or_local == "global":
            if cfg.scaling.per_channel:
                transformed_X = torch.stack([
                    (X[:,:,0] - self.means[0]) / self.stds[0],
                    (X[:,:,1] - self.means[1]) / self.stds[1],
                    (X[:,:,2] - self.means[2]) / self.stds[2]
                ], dim=-1)
            else:
                transformed_X = (X - self.means) / self.stds
        if cfg.scaling.global_or_local == "local":
            for idx, x in enumerate(X):
                if cfg.scaling.per_channel:
                    transformed_X[idx,:,0] = (x[:,0] - torch.mean(x[:,0])) / torch.std(x[:,0])
                    transformed_X[idx,:,1] = (x[:,1] - torch.mean(x[:,1])) / torch.std(x[:,1])
                    transformed_X[idx,:,2] = (x[:,2] - torch.mean(x[:,2])) / torch.std(x[:,2])
                else:
                    transformed_X[idx] = (x - np.mean(x)) / np.std(x)
        return transformed_X

class LogScaler(Scaler):
    def __init__(self):
        pass

    def fit(self, X: torch.Tensor):
        logger.info("Log scaler; skipping fit")

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Transforms the data using logarithmic scaling.

        Args:
            X (torch.Tensor): The data to be transformed.

        Returns:
            torch.Tensor: The transformed data.
        """
        logger.info("Data has been log scaled")
        return torch.log(X)
