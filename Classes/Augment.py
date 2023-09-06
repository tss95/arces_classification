from global_config import logger, cfg
import numpy as np
from scipy.signal.windows import tukey, gaussian, triang
import tensorflow as tf
import math

def augment_pipeline(x, y):
    batch_size = tf.shape(x)[0]
    batch_size_float = tf.cast(batch_size, tf.float32)  # Cast to float for multiplication
    
    # Zero Channel Augmentation
    if cfg.augment.zero_channel:
        prob_zero_channel_tf = tf.constant(cfg.augment.zero_channel_kwargs.prob, dtype=tf.float32)
        num_to_select_zero_channel = tf.cast(tf.round(batch_size_float * prob_zero_channel_tf), tf.int32)
        indices_zero_channel = tf.random.shuffle(tf.range(batch_size))[:num_to_select_zero_channel]
        x = zero_channel_tf(x, indices_zero_channel)
        
    # Add Gap Augmentation
    if cfg.augment.add_gap:
        prob_add_gap_tf = tf.constant(cfg.augment.add_gap_kwargs.prob, dtype=tf.float32)
        num_to_select_add_gap = tf.cast(tf.round(batch_size_float * prob_add_gap_tf), tf.int32)
        indices_add_gap = tf.random.shuffle(tf.range(batch_size))[:num_to_select_add_gap]
        x = add_gap_tf(x, indices_add_gap, cfg.augment.add_gap_kwargs.max_size)
        
    # Add Noise Augmentation
    if cfg.augment.add_noise:
        prob_add_noise_tf = tf.constant(cfg.augment.add_noise_kwargs.prob, dtype=tf.float32)
        num_to_select_add_noise = tf.cast(tf.round(batch_size_float * prob_add_noise_tf), tf.int32)
        indices_add_noise = tf.random.shuffle(tf.range(batch_size))[:num_to_select_add_noise]
        x = add_noise_tf(x, indices_add_noise, cfg.augment.add_noise_kwargs.noise_std)
        
    # Taper Augmentation
    if cfg.augment.taper:
        prob_taper_tf = tf.constant(cfg.augment.taper_kwargs.prob, dtype=tf.float32)
        num_to_select_taper = tf.cast(tf.round(batch_size_float * prob_taper_tf), tf.int32)
        indices_taper = tf.random.shuffle(tf.range(batch_size))[:num_to_select_taper]
        x = taper_tf(x, indices_taper, cfg.augment.taper_kwargs.alpha)
        
    return x, y


def zero_channel_tf(x, indices):
    batch_size, timesteps, channels = tf.shape(x)
    zeros = tf.zeros([batch_size, timesteps, channels], dtype=x.dtype)
    
    # Create a mask of shape [batch_size] where selected indices are set to True
    mask = tf.scatter_nd(tf.reshape(indices, [-1, 1]), tf.ones_like(indices, dtype=tf.bool), [batch_size])
    
    mask_expanded = tf.reshape(mask, [-1, 1, 1])  # shape: [batch_size, 1, 1]
    
    # Only zero out the channel for samples where mask is True
    updated_x = tf.where(mask_expanded, zeros, x)
    
    return updated_x

def add_gap_tf(x, indices, max_size):
    batch_size, timesteps, channels = tf.shape(x)
    max_gap = tf.cast((1 - max_size) * tf.cast(timesteps, tf.float32), tf.int32)
    gap_start = tf.random.uniform([], 0, max_gap, dtype=tf.int32)
    gap_end = tf.random.uniform([], minval=gap_start, maxval=gap_start + tf.cast(max_size * tf.cast(timesteps, tf.float32), tf.int32), dtype=tf.int32)
    gap_indices = tf.range(gap_start, gap_end)
    gap_length = tf.shape(gap_indices)[0]

    # Create a full index tensor for the update operation
    batch_indices = tf.repeat(indices, gap_length * channels)
    time_indices = tf.tile(tf.repeat(gap_indices, channels), [tf.size(indices)])
    channel_indices = tf.tile(tf.range(channels), [tf.size(indices) * gap_length])

    full_indices = tf.stack([batch_indices, time_indices, channel_indices], axis=1)
    updates = tf.zeros([tf.size(full_indices) // 3], dtype=x.dtype)
    
    x = tf.tensor_scatter_nd_update(x, full_indices, updates)
    return x

def add_noise_tf(x, indices, noise_std):
    selected_x = tf.gather(x, indices)
    noise_shape = tf.shape(selected_x)
    noise = tf.random.normal(shape=noise_shape, mean=0., stddev=noise_std, dtype=x.dtype)
    updated_x = selected_x + noise
    x = tf.tensor_scatter_nd_update(x, tf.expand_dims(indices, axis=-1), updated_x)
    return x

def taper_tf(x, indices, alpha=0.04):
    l = tf.shape(x)[1]  # Number of timesteps for selected samples
    t = tf.linspace(0., 1., l)
    alpha_tf = tf.constant(alpha, dtype=tf.float32)
    pi_tf = tf.constant(math.pi, dtype=tf.float32)
    w = tf.where(
        t < alpha_tf,
        0.5 * (1 + tf.cos(-pi_tf / alpha_tf * (t - alpha_tf))),
        tf.where(
            t <= (1 - alpha_tf),
            1.0,
            0.5 * (1 + tf.cos(pi_tf / alpha_tf * (t - 1 + alpha_tf)))
        )
    )
    
    w = tf.expand_dims(w, axis=0)  # Add batch dimension
    w = tf.expand_dims(w, axis=2)  # Add channel dimension
    
    selected_x = tf.gather(x, indices)
    updated_x = selected_x * w
    x = tf.tensor_scatter_nd_update(x, tf.expand_dims(indices, axis=-1), updated_x)
    
    return x
