import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from global_config import cfg, model_cfg

"""
Adapted from:
https://github.com/state-spaces/s4/blob/main/models/s4/s4d.py

"""


class S4DKernel(Model):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        # Generate dt
        H = d_model
        log_dt = np.random.rand(H) * (
            np.log(dt_max) - np.log(dt_min)
        ) + np.log(dt_min)

        C = np.random.randn(H, N // 2).astype(np.complex64)
        self.C = tf.Variable(tf.complex(C.real, C.imag))
        self.log_dt = tf.Variable(log_dt, dtype=tf.complex64)

        log_A_real = np.log(0.5 * np.ones((H, N//2)))
        A_imag = np.pi * np.repeat(np.arange(N//2)[None, :], H, axis=0)
        self.log_A_real = tf.Variable(log_A_real)
        self.A_imag = tf.Variable(A_imag, dtype=tf.complex64)

    def call(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        # Materialize parameters
        dt = tf.exp(self.log_dt) # (H)
        C = self.C # (H N)
        A = -tf.exp(tf.cast(self.log_A_real, tf.complex64)) + 1j * self.A_imag # (H N)

        # Vandermonde multiplication
        dtA = A * tf.expand_dims(dt, -1)  # (H N)
        K = dtA[..., None] * tf.cast(tf.range(L), dtA.dtype) # (H N L)
        C = C * (tf.exp(dtA)-1.) / A
        C = C[..., tf.newaxis]  # Add an extra dimension to C
        K = 2 * tf.reduce_sum(C * tf.exp(K), axis=1)
        K = tf.math.real(K)

        return K

class S4D(Model):
    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        self.D = tf.Variable(tf.random.normal((self.h,)))

        # SSM Kernel
        self.kernel = S4Kernel(self.h, N=self.n, dt_min=model_cfg.dt_min, dt_max=model_cfg.dt_max, **kernel_args)

        # Pointwise
        self.activation = layers.Activation('gelu')
        self.dropout = layers.Dropout(dropout) if dropout > 0.0 else tf.identity

        # position-wise output transform to mix features
        self.output_linear = tf.keras.Sequential([
            layers.Conv1D(2*self.h, kernel_size=1),
            layers.Activation('tanh'),
            layers.Dense(self.h)
        ])

    def call(self, u):
        """ Input and output shape (B, H, L) """
        if not self.transposed: u = tf.transpose(u, perm=[0, 2, 1])
        L = tf.shape(u)[-1]

        # Compute SSM Kernel
        k = self.kernel(L) # (H L)

        # Convolution
        k_f = tf.signal.rfft(k, fft_length=[2*L]) # (H L)
        k_f = k_f[None, ...]
        u_f = tf.signal.rfft(u, fft_length=[2*L]) # (B H L)
        y = tf.signal.irfft(u_f*k_f, fft_length=[2*L]) # (B H L)
        y = y[..., :L]

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D[..., None]

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed: y = tf.transpose(y, perm=[0, 2, 1])
        return y, None # Return a dummy state to satisfy this repo's interface, but this can be modified
    

if __name__ == "__main__":
    # Instantiate the model
    model = S4(d_model=model_cfg.d_model, d_state=model_cfg.d_state, dropout=model_cfg.dropout, transposed=model_cfg.transposed)

    # Create a random input tensor
    input_tensor = tf.random.uniform((1, 3, 9601))

    # Call the model
    output, _ = model(input_tensor)

    # Print the output shape
    print("Output shape:", output.shape)