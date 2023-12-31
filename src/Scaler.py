# Your existing imports and setup here
from global_config import logger, cfg
import numpy as np
import tensorflow as tf
import pickle 
import os
from typing import Any, Union

class Scaler:
    
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
        if self.scaler_type == "robust":
            return RobustScaler()
        if self.scaler_type == "log":
            return LogScaler()

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
        scaler_path = os.path.join(scaler_path, f"{self.scaler_type}_{self.global_or_local}.pkl")
        self.scaler = pickle.load(open(scaler_path, "rb"))
        logger.info("Scaler loaded.")

    def save_scaler(self):
        """
        Saves the fitted scaler to a file.
        """
        logger.info("Saving fitted scaler.")
        scaler_path = os.path.join(scaler_path, f"{self.scaler_type}_{self.global_or_local}.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Scaler saved to {scaler_path}.")

    def transform(self, X: Any) -> Any:
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

    def fit(self, X: np.ndarray):
        """
        Fits the scaler to the data for MinMax scaling.

        Args:
            X (np.ndarray): The data to fit the scaler on.
        """
        if cfg.scaling.global_or_local == "global":
            x1 = X[:,:,0]
            x2 = X[:,:,1]
            x3 = X[:,:,2]
            max_1, max_2, max_3 = np.max(x1), np.max(x2), np.max(x3)
            min_1, min_2, min_3 = np.min(x1), np.min(x2), np.min(x3)

            if cfg.scaling.per_channel:    
                self.maxs = np.array([max_1, max_2, max_3])
                self.mins = np.array([min_1, min_2, min_3])
            else:
                self.maxs = np.array([max_1, max_2, max_3]).max()
                self.mins = np.array([min_1, min_2, min_3]).min()
        if cfg.scaling.global_or_local == "local":
            logger.info("Local minmax; skipping fit")

    def transform(self, X: np.ndarray) -> tf.Tensor:
        """
        Transforms the data using the MinMax scaling approach.

        Args:
            X (np.ndarray): The data to be transformed.

        Returns:
            tf.Tensor: The transformed data.
        """
        X = tf.cast(X, tf.float32)
        transformed_X = tf.identity(X)  # Create a new tensor with the same values as X

        if cfg.scaling.global_or_local == "global":
            if cfg.scaling.per_channel:
                transformed_X = tf.stack([
                    (X[:,:,0] - self.mins[0]) / (self.maxs[0] - self.mins[0]),
                    (X[:,:,1] - self.mins[1]) / (self.maxs[1] - self.mins[1]),
                    (X[:,:,2] - self.mins[2]) / (self.maxs[2] - self.mins[2])
                ], axis=-1)
            else:
                transformed_X = (X - self.mins) / (self.maxs - self.mins)

        if cfg.scaling.global_or_local == "local":
            transformed_list = []
            for idx in tf.range(tf.shape(X)[0]):
                x = X[idx]
                if cfg.scaling.per_channel:
                    transformed_x = tf.stack([
                        (x[:,0] - tf.reduce_min(x[:,0])) / (tf.reduce_max(x[:,0]) - tf.reduce_min(x[:,0])),
                        (x[:,1] - tf.reduce_min(x[:,1])) / (tf.reduce_max(x[:,1]) - tf.reduce_min(x[:,1])),
                        (x[:,2] - tf.reduce_min(x[:,2])) / (tf.reduce_max(x[:,2]) - tf.reduce_min(x[:,2]))
                    ], axis=-1)
                else:
                    transformed_x = (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x))
                transformed_list.append(transformed_x)
            transformed_X = tf.stack(transformed_list, axis=0)

        return transformed_X

class StandardScaler(Scaler):
    def __init__(self):
        pass

    def fit(self, X: np.ndarray):
        """
        Fits the scaler to the data for Standard scaling.

        Args:
            X (np.ndarray): The data to fit the scaler on.
        """
        if cfg.scaling.global_or_local == "global":
            x1 = X[:,:,0]
            x2 = X[:,:,1]
            x3 = X[:,:,2]
            mean_1, mean_2, mean_3 = np.mean(x1), np.mean(x2), np.mean(x3)
            std_1, std_2, std_3 = np.std(x1), np.std(x2), np.std(x3)

            if cfg.scaling.per_channel:    
                self.means = np.array([mean_1, mean_2, mean_3])
                self.stds = np.array([std_1, std_2, std_3])
            else:
                self.means = np.array([mean_1, mean_2, mean_3]).mean()
                self.stds = np.array([std_1, std_2, std_3]).mean()
        if cfg.scaling.global_or_local == "local":
            logger.info("Local standard; skipping fit")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transforms the data using the Standard scaling approach.

        Args:
            X (np.ndarray): The data to be transformed.

        Returns:
            np.ndarray: The transformed data.
        """
        X = X.astype(float)
        if cfg.scaling.global_or_local == "global":
            if cfg.scaling.per_channel:
                X[:,:,0] = (X[:,:,0] - self.means[0]) / self.stds[0]
                X[:,:,1] = (X[:,:,1] - self.means[1]) / self.stds[1]
                X[:,:,2] = (X[:,:,2] - self.means[2]) / self.stds[2]
            else:
                X = (X - self.means) / self.stds
        if cfg.scaling.global_or_local == "local":
            for idx, x in enumerate(X):
                if cfg.scaling.per_channel:
                    X[idx,:,0] = (x[:,0] - np.mean(x[:,0])) / np.std(x[:,0])
                    X[idx,:,1] = (x[:,1] - np.mean(x[:,1])) / np.std(x[:,1])
                    X[idx,:,2] = (x[:,2] - np.mean(x[:,2])) / np.std(x[:,2])
                else:
                    X[idx] = (x - np.mean(x)) / np.std(x)
        logger.info("Data has been standard scaled")
        return X

class RobustScaler(Scaler):
    def __init__(self):
        pass


    def fit(self, X: np.ndarray):
        """
        Fits the scaler to the data for Robust scaling.

        Args:
            X (np.ndarray): The data to fit the scaler on.
        """
        if cfg.scaling.global_or_local == "global":
            x1 = X[:,0]
            x2 = X[:,1]
            x3 = X[:,2]
            median_1, median_2, median_3 = np.median(x1), np.median(x2), np.median(x3)
            iqr_1, iqr_2, iqr_3 = np.subtract(*np.percentile(x1, [75, 25])), np.subtract(*np.percentile(x2, [75, 25])), np.subtract(*np.percentile(x3, [75, 25]))

            if cfg.scaling.per_channel:    
                self.medians = np.array([median_1, median_2, median_3])
                self.iqrs = np.array([iqr_1, iqr_2, iqr_3])
            else:
                self.medians = np.median(np.array([median_1, median_2, median_3]))
                self.iqrs = np.median(np.array([iqr_1, iqr_2, iqr_3]))
        if cfg.scaling.global_or_local == "local":
            logger.info("Local robust; skipping fit")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transforms the data using the Robust scaling approach.

        Args:
            X (np.ndarray): The data to be transformed.

        Returns:
            np.ndarray: The transformed data.
        """
        X = X.astype(float)
        if cfg.scaling.global_or_local == "global":
            if self.per_channel:
                X[:,:,0] = (X[:,:,0] - self.medians[0]) / self.iqrs[0]
                X[:,:,1] = (X[:,:,1] - self.medians[1]) / self.iqrs[1]
                X[:,:,2] = (X[:,:,2] - self.medians[2]) / self.iqrs[2]
            else:
                X = (X - self.medians) / self.iqrs
        if cfg.scaling.global_or_local == "local":
            for idx, x in enumerate(X):
                if cfg.scaling.per_channel:
                    X[idx,:,0] = (x[:,0] - np.median(x[:,0])) / np.subtract(*np.percentile(x[:,0], [75, 25]))
                    X[idx,:,1] = (x[:,1] - np.median(x[:,1])) / np.subtract(*np.percentile(x[:,1], [75, 25]))
                    X[idx,:,2] = (x[:,2] - np.median(x[:,2])) / np.subtract(*np.percentile(x[:,2], [75, 25]))
                else:
                    X[idx] = (x - np.median(x)) / np.subtract(*np.percentile(x, [75, 25]))
        logger.info("Data has been robust scaled")
        return X

class LogScaler(Scaler):
    def __init__(self):
        pass

    def fit(self, X: np.ndarray):
        logger.info("Log scaler; skipping fit")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transforms the data using logarithmic scaling.

        Args:
            X (np.ndarray): The data to be transformed.

        Returns:
            np.ndarray: The transformed data.
        """
        logger.info("Data has been log scaled")
        return np.log(X)
