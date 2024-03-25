import torch
import numpy as np

import random

class BandpassFilterTransform:
    def __init__(self, lower_bounds, upper_bounds, sampling_rate=100, prob=0.5, default_low=2, default_high=8):
        """
        Initializes the bandpass filter transform with options for random selection 
        from predefined lists of lower and upper bounds, and default bounds.
        
        Parameters:
        - lower_bounds: List of possible lower bounds for the filter.
        - upper_bounds: List of possible upper bounds for the filter.
        - sampling_rate: Sampling rate of the data.
        - prob: Probability of applying the augmentation.
        - default_low: Default lower bound of the filter if augmentation isn't applied.
        - default_high: Default upper bound of the filter if augmentation isn't applied.
        """
        assert len(lower_bounds) == len(upper_bounds), "Lower and upper bounds lists must be of the same length."
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.default_low = default_low
        self.default_high = default_high
        self.sampling_rate = sampling_rate
        self.prob = prob

    def __call__(self, x):
        # Decide whether to apply augmentation or use default bounds
        if torch.rand(1).item() > self.prob:
            low_freq = self.default_low
            high_freq = self.default_high
        else:
            # Randomly select from the predefined bounds
            index = random.randint(0, len(self.lower_bounds) - 1)
            low_freq = self.lower_bounds[index]
            high_freq = self.upper_bounds[index]

        # Apply bandpass filter with selected bounds
        return self.bandpass_filter(x, low_freq, high_freq, self.sampling_rate)
    
    def bandpass_filter(self, x, low_freq, high_freq, sampling_rate):
        batch_size, n_channels, timesteps = x.shape
        fft_signal = torch.fft.rfft(x, dim=2)
        freqs = torch.fft.rfftfreq(timesteps, d=1/sampling_rate)

        # Generate mask for the desired frequency band
        mask = (freqs >= low_freq) & (freqs <= high_freq)
        fft_signal_filtered = fft_signal * mask.unsqueeze(0).unsqueeze(1)

        # Inverse FFT to transform back to time domain
        signal_filtered = torch.fft.irfft(fft_signal_filtered, dim=2, n=timesteps)

        return signal_filtered.type(x.dtype)
    
class AddNoiseTransform:
    def __init__(self, prob, per_channel_scaling=True):
        """
        Initializes the AddNoiseTransform.
        
        Parameters:
        - prob: Probability of adding noise to a given sample in the batch.
        - per_channel_scaling: If True, scales noise by the standard deviation of each channel individually.
                               Otherwise, uses the global standard deviation across all channels.
        """
        self.prob = prob
        self.per_channel_scaling = per_channel_scaling

    def __call__(self, x):
        batch_size, n_channels, timesteps = x.shape
        num_to_select = round(batch_size * self.prob)
        indices = torch.randperm(batch_size)[:num_to_select]
        selected_x = x[indices]

        if self.per_channel_scaling:
            std_per_channel = torch.std(selected_x, dim=2)
            noise = torch.randn_like(selected_x)
            scaled_noise = noise * std_per_channel.unsqueeze(-1)
        else:
            std_global = torch.std(selected_x, dim=[1, 2])
            noise = torch.randn_like(selected_x)
            scaled_noise = noise * std_global.unsqueeze(-1).unsqueeze(-1)

        scaled_noise = scaled_noise.to(x.dtype)
        x.index_add_(0, indices, scaled_noise)
        return x
    
class AddGapTransform:
    def __init__(self, prob, max_size):
        """
        Initializes the AddGapTransform.

        Parameters:
        - prob: Probability of adding a gap to a given sample in the batch.
        - max_size: Maximum size of the gap as a fraction of the total timesteps.
        """
        self.prob = prob
        self.max_size = max_size

    def __call__(self, x):
        batch_size, n_channels, timesteps = x.shape
        num_to_select = int(round(batch_size * self.prob))
        selected_indices = torch.randperm(batch_size)[:num_to_select]
        gap_starts = torch.randint(0, timesteps - int(self.max_size * timesteps), (num_to_select,))

        updated_x = x.clone()
        for i, index in enumerate(selected_indices):
            gap_start = gap_starts[i]
            gap_end = gap_start + torch.randint(0, int(self.max_size * timesteps), (1,))
            channel = torch.randint(0, n_channels, (1,))

            gap_mask = torch.zeros(timesteps, dtype=torch.bool)
            gap_mask[gap_start:gap_end] = True
            updated_x[index, channel, gap_mask] = updated_x[index, channel, ~gap_mask].mean()

        return updated_x
    
    
class TaperTransform:
    def __init__(self, prob, alpha=0.04):
        """
        Initializes the TaperTransform.

        Parameters:
        - prob: Probability of applying the taper to a given sample in the batch.
        - alpha: Shape parameter of the Tukey window, controlling the tapering.
                 A value of 0 results in a rectangular window, and 1 results in a Hann window.
        """
        self.prob = prob
        self.alpha = alpha

    def __call__(self, x):
        batch_size, n_channels, timesteps = x.shape
        num_to_select = int(round(batch_size * self.prob))
        indices = torch.randperm(batch_size)[:num_to_select]

        w = self.tukey(timesteps, self.alpha, device=x.device).type(x.dtype)
        w = w.unsqueeze(0).unsqueeze(1)  # Adjust to shape (1, 1, timesteps) for broadcasting

        for index in indices:
            x[index] *= w  # Apply taper window across the timesteps

        return x

    def tukey(self, M, alpha=0.5, device='cuda:0'):
        # Create the Tukey window in PyTorch
        if alpha <= 0:
            return torch.ones(M, device=device)
        elif alpha >= 1:
            return torch.hann_window(M, device=device)

        x = torch.linspace(0.0, 1.0, M, device=device)
        w = torch.ones(x.shape, device=device)

        # First condition 0 <= x < alpha/2
        first_condition = x < alpha / 2
        w = torch.where(first_condition, 0.5 * (1 + torch.cos(2 * np.pi / alpha * (x - alpha / 2))), w)

        # Second condition 1 - alpha / 2 <= x <= 1
        second_condition = x >= (1 - alpha / 2)
        w = torch.where(second_condition, 0.5 * (1 + torch.cos(2 * np.pi / alpha * (x - 1 + alpha / 2))), w)

        return w
    
class ZeroChannelTransform:
    def __init__(self, prob):
        """
        Initializes the ZeroChannelTransform.
        
        Parameters:
        - prob: Probability of zeroing out a channel for a given sample in the batch.
        """
        self.prob = prob

    def __call__(self, x):
        batch_size, n_channels, timesteps = x.shape
        num_to_select = round(batch_size * self.prob)
        selected_indices = torch.randperm(batch_size)[:num_to_select]
        channels_to_zero = torch.randint(0, n_channels, (num_to_select,))

        for i, batch_index in enumerate(selected_indices):
            channel = channels_to_zero[i]
            # Zero out the selected channel
            x[batch_index, channel, :] = 0  # Zero out across timesteps

        return x

    
