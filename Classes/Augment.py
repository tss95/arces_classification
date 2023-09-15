from global_config import logger, cfg
import tensorflow as tf
from scipy.signal.windows import tukey
import numpy as np

def augment_pipeline(x, y):
    # Add Noise
    if cfg.augment.add_noise:
        x = add_noise_tf(x, cfg.augment.add_noise_kwargs.prob)
    
    # Taper
    if cfg.augment.taper:
        x = taper_tf(x, cfg.augment.taper_kwargs.prob, cfg.augment.taper_kwargs.alpha)
        
    # Add Gap
    if cfg.augment.add_gap:
        x = add_gap_tf(x, cfg.augment.add_gap_kwargs.prob, cfg.augment.add_gap_kwargs.max_size)
        
    # Zero Channel
    if cfg.augment.zero_channel:
        x = zero_channel_tf(x, cfg.augment.zero_channel_kwargs.prob)
        
    return x, y


def add_noise_tf(x, prob):
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

def taper_tf(x, prob, alpha=0.04):
    batch_size = tf.shape(x)[0]
    num_to_select = tf.cast(tf.round(tf.cast(batch_size, tf.float32) * prob), tf.int32)
    indices = tf.random.shuffle(tf.range(batch_size))[:num_to_select]
    w = tukey_tf(tf.shape(x)[1], alpha) 
    # Cast to ensure data type consistency
    w = tf.cast(w, dtype=x.dtype)
    
    w = tf.expand_dims(w, axis=-1)
    selected_x = tf.gather(x, indices)

    # This line should now work without data type issues
    updated_x = selected_x * w
    
    x = tf.tensor_scatter_nd_update(x, tf.expand_dims(indices, axis=-1), updated_x)
    return x

def add_gap_tf(x, prob, max_size):
    dtype = x.dtype  # Capture the data type of the input tensor
    batch_size, timesteps, n_channels = tf.shape(x)
    num_to_select = tf.cast(tf.round(tf.cast(batch_size, tf.float32) * prob), tf.int32)
    indices = tf.random.shuffle(tf.range(batch_size))[:num_to_select]
    
    gap_start_max = tf.cast((1 - max_size) * tf.cast(timesteps, tf.float32), tf.int32)
    gap_start = tf.random.uniform(shape=[], minval=0, maxval=gap_start_max, dtype=tf.int32)
    gap_end = tf.random.uniform(shape=[], minval=gap_start, maxval=gap_start + tf.cast(max_size * tf.cast(timesteps, tf.float32), tf.int32), dtype=tf.int32)

    # Randomly select one channel
    channel = tf.random.uniform(shape=[], minval=0, maxval=n_channels, dtype=tf.int32)
    
    # Create zeros tensor for the gap
    zeros = tf.zeros([batch_size, gap_end - gap_start, n_channels], dtype=dtype)
    
    # Create a mask to apply the gap
    mask = tf.concat([tf.ones([batch_size, gap_start, n_channels], dtype=dtype), zeros, tf.ones([batch_size, timesteps - gap_end, n_channels], dtype=dtype)], axis=1)
    
    # Apply the mask only to the selected batch indices and channel
    mask = tf.tensor_scatter_nd_update(mask, tf.expand_dims(indices, axis=-1), tf.ones([num_to_select, timesteps, n_channels], dtype=dtype))
    mask = mask * tf.one_hot(channel, n_channels, dtype=dtype)
    
    return x * mask


def zero_channel_tf(x, prob):
    dtype = x.dtype  # Capture the data type of the input tensor
    batch_size, timesteps, n_channels = tf.shape(x)

    # Calculate the number of events to zero out
    num_to_select = tf.cast(tf.round(tf.cast(batch_size, tf.float32) * prob), tf.int32)

    # Randomly select batch indices to zero out
    selected_indices = tf.random.shuffle(tf.range(batch_size))[:num_to_select]

    # Generate a random channel for each selected batch
    random_channels = tf.random.uniform(shape=[num_to_select], minval=0, maxval=n_channels, dtype=tf.int32)

    # Create a mask of ones initially
    mask = tf.ones([batch_size, timesteps, n_channels], dtype=dtype)

    # Create indices for tensor_scatter_nd_update
    scatter_indices = tf.stack([
        tf.tile(selected_indices[:, tf.newaxis, tf.newaxis], [1, timesteps, 1]),
        tf.tile(tf.range(timesteps)[tf.newaxis, :, tf.newaxis], [num_to_select, 1, 1]),
        tf.tile(random_channels[:, tf.newaxis, tf.newaxis], [1, timesteps, 1])
    ], axis=-1)

    # Create updates (zeros)
    updates = tf.zeros([tf.shape(scatter_indices)[0], tf.shape(scatter_indices)[1], 1], dtype=dtype)

    # Update the mask to have zeros at the selected positions
    mask = tf.tensor_scatter_nd_update(mask, scatter_indices, updates)

    return x * mask




def tukey_tf(M, alpha=0.5):
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