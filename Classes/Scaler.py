# Your existing imports and setup here
from global_config import logger, cfg
import numpy as np


class Scaler:
    
    def __init__(self):
        self.global_or_local = cfg.scaling.global_or_local
        self.per_channel = cfg.scaling.per_channel
        self.scaler_type = cfg.scaling.scaler_type
        self.parameter_validation()
        self.scaler = self.get_scaler()

        
    def get_scaler(self):
        if self.scaler_type == "minmax":
            return MinMaxScaler()
        if self.scaler_type == "standard":
            return StandardScaler()
        if self.scaler_type == "robust":
            return RobustScaler()
        if self.scaler_type == "log":
            return LogScaler()

    def parameter_validation(self):
        # Validate scaler_type
        if self.scaler_type not in (valid_scalers:=["minmax", "standard", "robust", "log"]):
            raise Exception(f"Invalid scaler type ({self.scaler_type}). Must be one of {str(valid_scalers)}")
        # Validate per_channel
        if not isinstance(self.per_channel, bool):
            raise Exception(f"Invalid type for per_channel ({type(self.per_channel).__name__}). Must be a boolean.")
        # Validate global_or_local
        if self.global_or_local not in (valid_global_or_local:=["local", "global"]):
            raise Exception(f"Invalid value for global_or_local ({self.global_or_local}). Must be one of {str(valid_global_or_local)}")

    def fit(self, X):
        logger.info("Fitting sclaer")
        self.scaler.fit(X)
        logger.info("Scaler fitted")

    def transform(self, X):
        logger.info("Transforming data")
        return self.scaler.transform(X)


class MinMaxScaler(Scaler):
    def __init__(self):
        pass

    def fit(self, X):
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

    def transform(self, X):
        if cfg.scaling.global_or_local == "global":
            if cfg.scaling.per_channel:
                X[:,:,0] = (X[:,:,0] - self.mins[0]) / (self.maxs[0] - self.mins[0])
                X[:,:,1] = (X[:,:,1] - self.mins[1]) / (self.maxs[1] - self.mins[1])
                X[:,:,2] = (X[:,:,2] - self.mins[2]) / (self.maxs[2] - self.mins[2])
            else:
                X = (X - self.mins) / (self.maxs - self.mins)
        if cfg.scaling.global_or_local == "local":
            for x in X:
                if cfg.scaling.per_channel:
                    x[:,0] = (x[:,0] - np.min(x[:,0])) / (np.max(x[:,0]) - np.min(x[:,0]))
                    x[:,1] = (x[:,1] - np.min(x[:,1])) / (np.max(x[:,1]) - np.min(x[:,1]))
                    x[:,2] = (x[:,2] - np.min(x[:,2])) / (np.max(x[:,2]) - np.min(x[:,2]))
                else:
                    x = (x - np.min(x)) / (np.max(x) - np.min(x))
        return X

class StandardScaler(Scaler):
    def __init__(self):
        pass

    def fit(self, X):
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
        if cfg.scalingelf.global_or_local == "local":
            logger.info("Local standard; skipping fit")

    def transform(self, X):
        if cfg.scaling.global_or_local == "global":
            if cfg.scaling.per_channel:
                X[:,:,0] = (X[:,:,0] - self.means[0]) / self.stds[0]
                X[:,:,1] = (X[:,:,1] - self.means[1]) / self.stds[1]
                X[:,:,2] = (X[:,:,2] - self.means[2]) / self.stds[2]
            else:
                X = (X - self.means) / self.stds
        if cfg.scaling.global_or_local == "local":
            for x in X:
                if cfg.scaling.per_channel:
                    x[:,0] = (x[:,0] - np.mean(x[:,0])) / np.std(x[:,0])
                    x[:,1] = (x[:,1] - np.mean(x[:,1])) / np.std(x[:,1])
                    x[:,2] = (x[:,2] - np.mean(x[:,2])) / np.std(x[:,2])
                else:
                    x = (x - np.mean(x)) / np.std(x)
        return X

class RobustScaler(Scaler):
    def __init__(self):
        pass


    def fit(self, X):
        if cfg.scaling.global_or_local == "global":
            x1 = X[:,:,0]
            x2 = X[:,:,1]
            x3 = X[:,:,2]
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

    def transform(self, X):
        if cfg.scaling.global_or_local == "global":
            if self.per_channel:
                X[:,:,0] = (X[:,:,0] - self.medians[0]) / self.iqrs[0]
                X[:,:,1] = (X[:,:,1] - self.medians[1]) / self.iqrs[1]
                X[:,:,2] = (X[:,:,2] - self.medians[2]) / self.iqrs[2]
            else:
                X = (X - self.medians) / self.iqrs
        if cfg.scaling.global_or_local == "local":
            for x in X:
                if cfg.scaling.per_channel:
                    x[:,0] = (x[:,0] - np.median(x[:,0])) / np.subtract(*np.percentile(x[:,0], [75, 25]))
                    x[:,1] = (x[:,1] - np.median(x[:,1])) / np.subtract(*np.percentile(x[:,1], [75, 25]))
                    x[:,2] = (x[:,2] - np.median(x[:,2])) / np.subtract(*np.percentile(x[:,2], [75, 25]))
                else:
                    x = (x - np.median(x)) / np.subtract(*np.percentile(x, [75, 25]))
        return X

class LogScaler(Scaler):
    def __init__(self):
        pass

    def fit(self, X):
        logger.info("Log scaler; skipping fit")
        return X

    def transform(self, X):
        return np.log(X)
