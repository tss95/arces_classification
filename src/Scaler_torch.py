from typing import Any, Dict, Tuple, Union
import torch
import numpy as np
from global_config import logger

EPS = 1e-8


def _to_tensor(x: Union[np.ndarray, torch.Tensor]) -> Tuple[torch.Tensor, bool]:
    """Ensure tensor input; return tensor and flag indicating if original was numpy."""
    if isinstance(x, torch.Tensor):
        return x, False
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x), True
    raise TypeError(f"Unsupported input type for scaling: {type(x)}")


class Scaler:
    """
    Thin wrapper that owns the concrete scaler implementation and handles config/state.
    Works on tensors of shape (batch, channels, timesteps) or numpy arrays with the same layout.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.scaler_type = cfg.scaling.scaler_type.lower()
        self.global_or_local = cfg.scaling.global_or_local
        self.per_channel = cfg.scaling.per_channel
        self._validate()
        self.impl = self._build_impl()

    def _build_impl(self):
        if self.scaler_type == "minmax":
            return MinMaxScaler(self.global_or_local, self.per_channel)
        if self.scaler_type == "standard":
            return StandardScaler(self.global_or_local, self.per_channel)
        raise NotImplementedError("Only minmax and standard scalers are currently implemented")

    def _validate(self):
        if self.scaler_type not in ["minmax", "standard"]:
            raise ValueError(f"Invalid scaler type ({self.scaler_type}). Must be one of ['minmax', 'standard'].")
        if not isinstance(self.per_channel, bool):
            raise ValueError(f"Invalid type for per_channel ({type(self.per_channel).__name__}). Must be a boolean.")
        if self.global_or_local not in ["local", "global"]:
            raise ValueError(f"Invalid value for global_or_local ({self.global_or_local}). Must be 'local' or 'global'.")

    @property
    def requires_fit(self) -> bool:
        return self.impl.requires_fit

    def fit_loader(self, dataloader):
        """Fit using a dataloader that yields either tensors or (tensor, labels, ids)."""
        if not self.requires_fit:
            logger.info("Scaler does not require fitting; skipping.")
            return
        logger.info("Fitting scaler on dataloader.")
        self.impl.fit_loader(dataloader)
        logger.info("Scaler fitted.")

    def transform(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        x_tensor, was_numpy = _to_tensor(X)
        x_tensor = x_tensor.float()
        transformed = self.impl.transform(x_tensor)
        if was_numpy:
            return transformed.numpy()
        return transformed

    def state_dict(self) -> Dict[str, Any]:
        return {
            "scaler_type": self.scaler_type,
            "global_or_local": self.global_or_local,
            "per_channel": self.per_channel,
            "state": self.impl.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        # Validate compatibility
        expected = (self.scaler_type, self.global_or_local, self.per_channel)
        incoming = (
            state.get("scaler_type"),
            state.get("global_or_local"),
            state.get("per_channel"),
        )
        if expected != incoming:
            logger.warning(f"Scaler config mismatch (expected {expected}, got {incoming}); loading state anyway.")
        self.impl.load_state_dict(state.get("state", {}))


class MinMaxScaler:
    def __init__(self, global_or_local: str, per_channel: bool):
        self.global_or_local = global_or_local
        self.per_channel = per_channel
        self.mins = None
        self.maxs = None

    @property
    def requires_fit(self) -> bool:
        return self.global_or_local == "global"

    def fit_loader(self, dataloader):
        assert self.requires_fit, "fit_loader should only be called when global scaling is requested."
        global_min, global_max = None, None
        for batch in dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.float()
            if self.per_channel:
                batch_min = x.amin(dim=(0, 2))
                batch_max = x.amax(dim=(0, 2))
            else:
                batch_min = x.amin()
                batch_max = x.amax()
            global_min = batch_min if global_min is None else torch.minimum(global_min, batch_min)
            global_max = batch_max if global_max is None else torch.maximum(global_max, batch_max)
        self.mins = global_min
        self.maxs = global_max

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.global_or_local == "local":
            if self.per_channel:
                mins = x.amin(dim=2, keepdim=True)
                maxs = x.amax(dim=2, keepdim=True)
            else:
                mins = x.amin(dim=(1, 2), keepdim=True)
                maxs = x.amax(dim=(1, 2), keepdim=True)
        else:
            if self.mins is None or self.maxs is None:
                raise RuntimeError("Scaler has not been fitted yet.")
            mins = self.mins.to(x.device)
            maxs = self.maxs.to(x.device)
            if self.per_channel:
                mins = mins.view(1, -1, 1)
                maxs = maxs.view(1, -1, 1)
        denom = (maxs - mins).clamp(min=EPS)
        return (x - mins) / denom

    def state_dict(self) -> Dict[str, Any]:
        if self.mins is None or self.maxs is None:
            return {}
        return {"mins": self.mins.cpu(), "maxs": self.maxs.cpu()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        self.mins = state.get("mins")
        self.maxs = state.get("maxs")


class StandardScaler:
    def __init__(self, global_or_local: str, per_channel: bool):
        self.global_or_local = global_or_local
        self.per_channel = per_channel
        self.means = None
        self.stds = None

    @property
    def requires_fit(self) -> bool:
        return self.global_or_local == "global"

    def fit_loader(self, dataloader):
        assert self.requires_fit, "fit_loader should only be called when global scaling is requested."
        running_sum = None
        running_sumsq = None
        count = 0
        for batch in dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.float()
            if self.per_channel:
                batch_sum = x.sum(dim=(0, 2))
                batch_sumsq = (x ** 2).sum(dim=(0, 2))
                batch_count = x.shape[0] * x.shape[2]
            else:
                batch_sum = x.sum()
                batch_sumsq = (x ** 2).sum()
                batch_count = x.numel()
            running_sum = batch_sum if running_sum is None else running_sum + batch_sum
            running_sumsq = batch_sumsq if running_sumsq is None else running_sumsq + batch_sumsq
            count += batch_count

        means = running_sum / count
        vars_ = running_sumsq / count - means ** 2
        stds = torch.sqrt(vars_.clamp(min=EPS))
        self.means = means
        self.stds = stds

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.global_or_local == "local":
            if self.per_channel:
                means = x.mean(dim=2, keepdim=True)
                stds = x.std(dim=2, keepdim=True).clamp(min=EPS)
            else:
                means = x.mean(dim=(1, 2), keepdim=True)
                stds = x.std(dim=(1, 2), keepdim=True).clamp(min=EPS)
        else:
            if self.means is None or self.stds is None:
                raise RuntimeError("Scaler has not been fitted yet.")
            means = self.means.to(x.device)
            stds = self.stds.to(x.device).clamp(min=EPS)
            if self.per_channel:
                means = means.view(1, -1, 1)
                stds = stds.view(1, -1, 1)
        return (x - means) / stds

    def state_dict(self) -> Dict[str, Any]:
        if self.means is None or self.stds is None:
            return {}
        return {"means": self.means.cpu(), "stds": self.stds.cpu()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        self.means = state.get("means")
        self.stds = state.get("stds")
