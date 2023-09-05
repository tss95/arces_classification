from global_config import logger, cfg
import numpy as np
from scipy.signal.windows import tukey, gaussian, triang
import tensorflow as tf
import math

# Make TF pipeline
def augment_pipeline(dataset):
    if cfg.augment.zero_channel:
        prob_zero_channel = cfg.augment.zero_channel_kwargs.prob
        dataset = dataset.map(lambda x, y: (tf.cond(tf.random.uniform([]) < prob_zero_channel, 
                                                    lambda: zero_channel_tf(x, channel=cfg.augment.zero_channel_channel), 
                                                    lambda: x), y))
    if cfg.augment.add_gap:
        prob_add_gap = cfg.augment.add_gap_kwargs.prob
        dataset = dataset.map(lambda x, y: (tf.cond(tf.random.uniform([]) < prob_add_gap, 
                                                    lambda: add_gap_tf(x, max_size=cfg.augment.add_gap_kwargs.max_size), 
                                                    lambda: x), y))
    if cfg.augment.add_noise:
        prob_add_noise = cfg.augment.add_noise_kwargs.prob
        dataset = dataset.map(lambda x, y: (tf.cond(tf.random.uniform([]) < prob_add_noise, 
                                                    lambda: add_noise_tf(x, noise_std=cfg.augment.add_noise_kwargs.noise_std), 
                                                    lambda: x), y))
    if cfg.augment.taper:
        prob_taper = cfg.augment.taper_kwargs.prob
        dataset = dataset.map(lambda x, y: (tf.cond(tf.random.uniform([]) < prob_taper, 
                                                    lambda: taper_tf(x, alpha=cfg.augment.taper_kwargs.alpha), 
                                                    lambda: x), y))
    return dataset

# Set all values in a specific channel to zero (TensorFlow version)
def zero_channel_tf(x, channel):
    zeros = tf.zeros_like(x[..., channel])
    x = tf.tensor_scatter_nd_update(x, [[..., channel]], zeros)
    return x

# Add a gap of zeros to the waveform at a random position (TensorFlow version)
def add_gap_tf(x, max_size):
    l = tf.shape(x)[0]
    gap_start = tf.random.uniform([], 0, (1 - max_size) * tf.cast(l, tf.float32), dtype=tf.int32)
    gap_end = tf.random.uniform([], gap_start, gap_start + max_size * tf.cast(l, tf.float32), dtype=tf.int32)
    gap_indices = tf.range(gap_start, gap_end)
    updates = tf.zeros([gap_end - gap_start], dtype=x.dtype)
    x = tf.tensor_scatter_nd_update(x, [gap_indices], updates)
    return x

# Add Gaussian noise to the waveform (TensorFlow version)
def add_noise_tf(x, noise_std):
    noise = tf.random.normal(shape=tf.shape(x), mean=0., stddev=noise_std)
    return x + noise

# Apply a Tukey window to taper the waveform (TensorFlow version)
def taper_tf(x, alpha=0.04):
    l = tf.shape(x)[0]
    t = tf.linspace(0., 1., l)
    w = tf.where(
        t < alpha,
        0.5 * (1 + tf.cos(tf.constant(-math.pi) / alpha * (t - alpha))),
        tf.where(
            t <= (1 - alpha),
            1.0,
            0.5 * (1 + tf.cos(tf.constant(math.pi) / alpha * (t - 1 + alpha)))
        )
    )
    return x * w[:, tf.newaxis]
