from global_config import logger, cfg
from scipy.signal.windows import tukey
import numpy as np

class augment_torch:
    import torch

    def __init__(self):
        pass

    def augment_pipeline(self, x, y):
        # Add Noise
        if cfg.augment.add_noise:
            x = self.add_noise_torch(x, cfg.augment.add_noise_kwargs.prob)
        
        # Taper
        if cfg.augment.taper:
            x = self.taper_torch(x, cfg.augment.taper_kwargs.prob, cfg.augment.taper_kwargs.alpha)
            
        # Add Gap
        if cfg.augment.add_gap:
            x = self.add_gap_torch(x, cfg.augment.add_gap_kwargs.prob, cfg.augment.add_gap_kwargs.max_size)
            
        # Zero Channel
        if cfg.augment.zero_channel:
            x = self.zero_channel_torch(x, cfg.augment.zero_channel_kwargs.prob)
            
        return x, y

    def add_noise_torch(self, x, prob):
        batch_size = x.shape[0]
        num_to_select = round(batch_size * prob)
        indices = torch.randperm(batch_size)[:num_to_select]
        selected_x = x[indices]

        if cfg.scaling.per_channel:
            std_per_channel = torch.std(selected_x, dim=1)
            noise = torch.randn_like(selected_x)
            scaled_noise = noise * std_per_channel.unsqueeze(-1)
        else:
            std_global = torch.std(selected_x, dim=[1, 2])
            noise = torch.randn_like(selected_x)
            scaled_noise = noise * std_global.unsqueeze(-1).unsqueeze(-1)

        # Cast scaled_noise to the same dtype as x
        scaled_noise = scaled_noise.to(x.dtype)

        x.index_add_(0, indices, scaled_noise)
        return x

    def add_gap_torch(self, x, prob, max_size: float):
        batch_size, timesteps, n_channels = x.shape

        # Number of batches to select for introducing gaps
        num_to_select = int(round(batch_size * prob))
        selected_events = torch.randperm(batch_size)[:num_to_select]

        # Randomly select a starting points for the gaps
        gap_starts = torch.randint(0, timesteps-int(max_size*timesteps), (num_to_select,))

        # Create a tensor to hold the updated values
        updated_values = x.clone()

        for i in range(num_to_select):
            event = selected_events[i]
            gap_start = gap_starts[i]
            gap_end = gap_start + torch.randint(0, int(max_size*timesteps), (1,))

            # Randomly select a channel for each event
            selected_channel = torch.randint(0, n_channels, (1,))

            # Calculate the average value of the channel for each event
            avg_val = x[event, :, selected_channel].mean()

            # Create a tensor of indices for the gap
            gap_indices = torch.arange(timesteps)
            gap_mask = (gap_indices >= gap_start) & (gap_indices < gap_end)

            # Replace the values within the gap with the average value
            updated_values[event, gap_mask, selected_channel] = avg_val

        return updated_values
    
    def taper_torch(self, x, prob, alpha=0.04):
        batch_size = x.shape[0]
        num_to_select = int(round(batch_size * prob))
        indices = torch.randperm(batch_size)[:num_to_select]

        w = self.tukey_torch(x.shape[1], alpha, device=x.device)
        w=w.type(x.dtype)
        w = w.unsqueeze(-1)
        selected_x = x[indices]
        
        updated_x = selected_x * w
        x[indices] = updated_x
        return x
    
    def zero_channel_torch(self, x, prob):
        dtype = x.dtype  # Capture the data type of the input tensor
        batch_size, timesteps, n_channels = x.shape

        # Calculate the number of events to zero out
        num_to_select = round(batch_size * prob)

        # Randomly select batch indices to zero out
        selected_indices = torch.randperm(batch_size)[:num_to_select]

        # Generate a random channel for each selected batch
        random_channels = torch.randint(0, n_channels, (num_to_select,))

        # Create a copy of the original tensor
        x_copy = x.clone()

        for i in range(num_to_select):
            x_i = x_copy[selected_indices[i], :, :]
            channel_avg = x_i[:,random_channels[i]].mean(dim=-1, keepdim=True)
            x_copy[selected_indices[i], :, random_channels[i]] = channel_avg

        return x_copy

    def tukey_torch(self, M, alpha=0.5, device='cuda:0'):
        # Create the Tukey window in PyTorch
        # M is the number of points in the output window
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

class augment_tf:
    import tensorflow as tf

    def __init__(self):
        pass


    def augment_pipeline(self, x, y):
        # Add Noise
        if cfg.augment.add_noise:
            x = self.add_noise_tf(x, cfg.augment.add_noise_kwargs.prob)
        
        # Taper
        if cfg.augment.taper:
            x = self.taper_tf(x, cfg.augment.taper_kwargs.prob, cfg.augment.taper_kwargs.alpha)
            
        # Add Gap
        if cfg.augment.add_gap:
            x = self.add_gap_tf(x, cfg.augment.add_gap_kwargs.prob, cfg.augment.add_gap_kwargs.max_size)
            
        # Zero Channel
        if cfg.augment.zero_channel:
            x = self.zero_channel_tf(x, cfg.augment.zero_channel_kwargs.prob)
            
        return x, y


    def add_noise_tf(self, x, prob):
        batch_size = tf.shape(x)[0]
        num_to_select = tf.cast(tf.round(tf.cast(batch_size, tf.float32) * prob), tf.int32)
        indices = tf.random.shuffle(tf.range(batch_size))[:num_to_select]
        selected_x = tf.gather(x, indices)
        selected_x = tf.convert_to_tensor(selected_x)  # Convert to dense tensor if needed

        if cfg.scaling.per_channel:
            std_per_channel = tf.math.reduce_std(selected_x, axis=1)
            noise = tf.random.normal(tf.shape(selected_x), dtype=tf.float32)
            scaled_noise = tf.multiply(tf.cast(noise, tf.float32), tf.cast(tf.expand_dims(std_per_channel, axis=-1), tf.float32))
        else:
            std_global = tf.math.reduce_std(selected_x, axis=[1, 2])
            noise = tf.random.normal(tf.shape(selected_x), dtype=tf.float32)
            scaled_noise = tf.multiply(tf.cast(noise, tf.float32), tf.cast(tf.expand_dims(tf.expand_dims(std_global, axis=-1), axis=-1), tf.float32))

        # Cast scaled_noise to the same dtype as x
        scaled_noise = tf.cast(scaled_noise, x.dtype)
        
        x = tf.tensor_scatter_nd_add(x, tf.expand_dims(indices, axis=-1), scaled_noise)
        return x




    def taper_tf(self, x, prob, alpha=0.04):
        batch_size = tf.shape(x)[0]
        num_to_select = tf.cast(tf.round(tf.cast(batch_size, tf.float32) * prob), tf.int32)
        indices = tf.random.shuffle(tf.range(batch_size))[:num_to_select]
        w = self.tukey_tf(tf.shape(x)[1], alpha) 
        # Cast to ensure data type consistency
        w = tf.cast(w, dtype=x.dtype)
        
        w = tf.expand_dims(w, axis=-1)
        selected_x = tf.gather(x, indices)

        # This line should now work without data type issues
        updated_x = selected_x * w
        
        x = tf.tensor_scatter_nd_update(x, tf.expand_dims(indices, axis=-1), updated_x)
        return x







    def add_gap_tf(self, x, prob, max_size: float):
        logger.warning("add_gap_tf is not implemented yet")
        return x
        """
        batch_size, timesteps, n_channels = tf.shape(x)
    |
        # Number of batches to select for introducing gaps
        num_to_select = tf.cast(tf.round(batch_size * prob), tf.int32)
        selected_events = tf.random.shuffle(tf.range(batch_size))[:num_to_select]

        # Convert timesteps to float for multiplication with max_size
        timesteps_float = tf.cast(timesteps, tf.float32)

        # Calculate the maximum gap size in float context, then round up and convert to int
        max_gap_size = tf.cast(tf.math.ceil(max_size * timesteps_float), tf.int32)

        # Calculate the upper limit for gap start points
        gap_start_limit = timesteps - max_gap_size

        # Generate random gap start points
        gap_starts = tf.random.uniform((num_to_select,), 0, gap_start_limit, dtype=tf.int32)

        for i in tf.range(num_to_select):
            event = selected_events[i]
            gap_start = gap_starts[i]
            gap_end = gap_start + tf.random.uniform([], 0, max_gap_size, dtype=tf.int32)

            # Replace the values within the gap with the average value
            avg_val = tf.reduce_mean(x[event, :, :], keepdims=True)

            # Create indices for the gap
            gap_indices = tf.range(gap_start, gap_end)
            indices = tf.stack([tf.repeat(event, gap_end-gap_start), gap_indices], axis=1)

            # Create updates for the gap
            updates = tf.repeat(avg_val, gap_end-gap_start)

            x = tf.tensor_scatter_nd_update(x, indices, updates)

        return x
        """

    def zero_channel_tf(self, x, prob):
        dtype = x.dtype  # Capture the data type of the input tensor
        batch_size, timesteps, n_channels = tf.shape(x)

        # Calculate the number of events to zero out
        num_to_select = tf.cast(tf.round(tf.cast(batch_size, tf.float32) * prob), tf.int32)

        # Randomly select batch indices to zero out
        selected_indices = tf.random.shuffle(tf.range(batch_size))[:num_to_select]

        # Generate a random channel for each selected batch
        random_channels = tf.random.uniform(shape=[num_to_select], minval=0, maxval=n_channels, dtype=tf.int32)

        # Create a copy of the original tensor
        x_copy = tf.identity(x)

        for i in range(num_to_select.numpy()):
                x_i = tf.identity(x_copy[selected_indices[i], :, :])
                channel_avg = tf.reduce_mean(x_i[:, random_channels[i]], axis=-1, keepdims=True)
                channel_avg = tf.repeat(channel_avg, repeats=timesteps)  # Repeat the average to match the shape of the indices tensor
                x_copy = tf.tensor_scatter_nd_update(x_copy, [[selected_indices[i], j, random_channels[i]] for j in range(timesteps)], channel_avg)

        return x_copy


    def tukey_tf(self, M, alpha=0.5):
        # Create the Tukey window in TensorFlow
        # M is the number of points in the output window
        if alpha <= 0:
            return tf.ones(M)
        elif alpha >= 1:
            return tf.signal.hann_window(M)
        
        x = tf.linspace(0.0, 1.0, M)
        w = tf.ones(x.shape)
        
        # First condition 0 <= x < alpha/2
        first_condition = x < alpha / 2
        w = tf.where(first_condition, 0.5 * (1 + tf.cos(2 * np.pi / alpha * (x - alpha / 2))), w)
        
        # Second condition 1 - alpha / 2 <= x <= 1
        second_condition = x >= (1 - alpha / 2)
        w = tf.where(second_condition, 0.5 * (1 + tf.cos(2 * np.pi / alpha * (x - 1 + alpha / 2))), w)
        
        return w

def test_tukey():
    M = 10
    alpha = 0.5
    
    w_tf = augment_tf.tukey_tf(M, alpha)
    w_torch = augment_torch.tukey_torch(M, alpha)

    assert np.allclose(w_tf.numpy(), w_torch.cpu().numpy(), atol=1e-6)

def test_add_noise():
    x_tf = tf.random.normal([1, 10000, 3])
    x_torch = torch.from_numpy(np.copy(x_tf.numpy()))
    prob = 0.5

    y_tf = add_noise_tf(x_tf, prob)
    y_torch = add_noise_torch(x_torch, prob)

    assert np.allclose(y_tf.numpy(), y_torch.numpy(), atol=1e-3)

def test_taper():
    x_tf = tf.random.normal([1, 10000, 3])
    x_torch = torch.from_numpy(np.copy(x_tf.numpy())).to('cuda:0')
    prob = 0.5
    alpha = 0.04

    y_tf = taper_tf(x_tf, prob, alpha)
    y_torch = taper_torch(x_torch, prob, alpha)

    assert np.allclose(y_tf.numpy(), y_torch.cpu().numpy(), atol=1e-6)


def test_zero_channel():
    # Generate a random tensor in PyTorch
    x_torch = torch.rand((3, 6000, 3))

    # Convert the PyTorch tensor to a TensorFlow tensor
    x_tf = tf.convert_to_tensor(x_torch.numpy())

    # Call both functions
    output_torch = zero_channel_torch(x_torch, prob=1)
    output_tf = zero_channel_tf(x_tf, prob=1)

    # Convert the output of the PyTorch function to a TensorFlow tensor
    output_torch_tf = tf.convert_to_tensor(output_torch.numpy())

    plot_results(x_tf, output_torch_tf, output_tf, "Zero Channel test")

def test_gap():
    # Generate a random tensor in PyTorch
    x_torch = torch.rand((3, 6000, 3))

    # Convert the PyTorch tensor to a TensorFlow tensor
    x_tf = tf.convert_to_tensor(x_torch.numpy())

    # Call both functions
    output_torch = add_gap_torch(x_torch, prob=1, max_size=0.1)
    output_tf = add_gap_tf(x_tf, prob=1, max_size=0.1)

    # Convert the output of the PyTorch function to a TensorFlow tensor
    output_torch_tf = tf.convert_to_tensor(output_torch.numpy())

    plot_results(x_tf, output_torch_tf, output_tf, "Gap test")


def plot_results(x_tf, output_torch_tf, output_tf, title):
    # Create a figure for all events and channels
    fig, axs = plt.subplots(x_tf.shape[0]*2, x_tf.shape[2], figsize=(15, 10))

    # Loop over each event
    for i in range(x_tf.shape[0]):
        # Loop over each channel
        for j in range(x_tf.shape[2]):
            # Plot the output of the Torch function for this channel
            axs[2*i, j].plot(np.squeeze(output_torch_tf[i, :, j]))
            axs[2*i, j].set_title(f'Output Torch - Event {i+1} - Channel {j+1}')

            # Plot the output of the TensorFlow function for this channel
            axs[2*i+1, j].plot(np.squeeze(output_tf[i, :, j]))
            axs[2*i+1, j].set_title(f'Output TF - Event {i+1} - Channel {j+1}')

    # Save the plot to the specified path
    plt.savefig(os.path.join(cfg.paths.plots_folder, f"{title}.png"))

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    test_add_noise()
    test_tukey()
    test_taper()
    test_gap()
    test_zero_channel()