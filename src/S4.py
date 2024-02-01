import tensorflow as tf
from global_config import cfg, model_cfg

from tensorflow.keras.layers import Dense, Conv1D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
from tensorflow.keras import activations
from functools import partial
import numpy as np

def LinearActivation(
        d_output, bias=True,
        transposed=False,
        activation=None,
        activate=False, # Apply activation as part of this module
        **kwargs,
    ):
    """Returns a linear keras.Layer with control over axes order, initialization, and activation."""

    # Construct core module
    linear_cls = partial(Conv1D, kernel_size=1, padding='same') if transposed else Dense
    if activation is not None and activation == 'glu': d_output *= 2
    linear = linear_cls(d_output, use_bias=bias, **kwargs)

    if activate and activation is not None:
        activation = Activation(activation, dim=-2 if transposed else -1)
        linear = Sequential([linear, activation])
    return linear

def Activation(activation=None, dim=-1):
    if activation in [None, 'linear']:
        return tf.keras.layers.Lambda(lambda x: x)
    elif activation == 'glu':
        return tf.keras.layers.Lambda(lambda x: tf.keras.activations.glu(x, axis=dim))
    else:
        return tf.keras.layers.Activation(tf.keras.activations.get(activation))




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

kernel_registry = {
    's4d': S4DKernel
}

class FFTConv(tf.keras.layers.Layer):
    def __init__(self, d_model, l_max=None, channels=1, swap_channels=False, bidirectional=False, activation='gelu', transposed=True, dropout=0.0, tie_dropout=False, drop_kernel=0.0, mode='dplr', kernel=None, **kernel_args):
        super(FFTConv, self).__init__()
        self.d_model = d_model
        self.L = self.l_max = l_max
        self.bidirectional = bidirectional
        self.channels = channels
        self.transposed = transposed
        self.swap_channels = swap_channels

        if activation is not None and activation.startswith('glu'):
            channels *= 2
        self.activation = Activation(activation, dim=1 if self.transposed else -1)

        self.D = self.add_weight(shape=(channels, self.d_model), initializer='random_normal')

        if self.bidirectional:
            channels *= 2

        # Inner convolution kernel
        if mode is not None:
            assert kernel is None, "Pass either mode or kernel but not both"
            # TODO THIS NEEDS TO BE ADRESSED. Deviates significantly from the original code
            #kernel_cls = S4DKernel
            #self.kernel = kernel_cls(d_model=self.d_model, l_max=self.l_max, channels=channels, **kernel_args)
            self.kernel = S4DKernel(d_model, N=model_cfg.d_state, dt_min=model_cfg,dt_min, dt_max=model_cfg.dt_max, **kernel_args)

        dropout_fn = DropoutNd if tie_dropout else tf.keras.layers.Dropout
        self.drop = dropout_fn(dropout) if dropout > 0.0 else tf.keras.layers.Lambda(lambda x: x)
        self.drop_kernel = tf.keras.layers.Dropout(drop_kernel) if drop_kernel > 0.0 else tf.keras.layers.Lambda(lambda x: x)

    def call(self, x, state=None, rate=1.0, **kwargs):
        # Always work with (B D L) dimension in this module
        if not self.transposed: 
            x = tf.transpose(x, perm=[0, 2, 1])
        L = tf.shape(x)[-1]

        # Compute SS Kernel
        l_kernel = L if self.L is None else min(L, round(self.L / rate))
        k, k_state =  self.kernel(L=l_kernel, rate=rate, state=state) # (C H L) (B C H L)

        # Convolution
        if self.bidirectional:
            k0, k1 = tf.split(k, num_or_size_splits=2, axis=0)
            k = tf.pad(k0, [[0, 0], [0, L]]) + tf.pad(tf.reverse(k1, axis=[-1]), [[L, 0], [0, 0]])

        # Kernel dropout
        k = self.drop_kernel(k)

        # FFT convolution
        k_f = tf.signal.rfft(k, fft_length=[l_kernel+L]) # (C H L)
        x_f = tf.signal.rfft(x, fft_length=[l_kernel+L]) # (B H L)
        y_f = tf.einsum('bhl,chl->bchl', x_f, k_f)
        y = tf.signal.irfft(y_f, fft_length=[l_kernel+L]) # (B C H L)
        y = y[..., :L]

        # Compute D term in state space equation - essentially a skip connection
        y = y + tf.einsum('bhl,ch->bchl', x, self.D)

        # Compute state update
        if state is not None:
            assert not self.bidirectional, "Bidirectional not supported with state forwarding"
            y = y + k_state
            next_state = self.kernel.forward_state(x, state)
        else:
            next_state = None

        # Reshape to flatten channels
        if self.swap_channels:
            y = tf.reshape(y, [tf.shape(y)[0], -1, tf.shape(y)[-1]])
        else:
            y = tf.reshape(y, [tf.shape(y)[0], -1, tf.shape(y)[-1]])

        y = self.drop(y)

        if not self.transposed: 
            y = tf.transpose(y, perm=[0, 2, 1])
        y = self.activation(y)

        return y, next_state


class S4Block(tf.keras.layers.Layer):
    def __init__(self, d_model, bottleneck=None, gate=None, gate_act=None, mult_act=None, final_act='glu', dropout=0.0, tie_dropout=False, **layer_args):
        super(S4Block, self).__init__()

        self.d_model = d_model
        self.bottleneck = bottleneck
        self.gate = gate

        if bottleneck is not None:
            self.d_model = self.d_model // bottleneck
            self.input_linear = LinearActivation(self.d_model, 
                                                 activation = None, 
                                                 transposed = False,
                                                 activate = False)

        if gate is not None:
            self.input_gate = LinearActivation(self.d_model * gate, 
                                               activation = activations.get(gate_act), 
                                               transposed = False, 
                                               activate = True)
            if self.layer.d_output != self.d_model * gate:
                self.output_gate = LinearActivation(self.d_model * gate, 
                                                    activation = None, 
                                                    transposed = False, 
                                                    activate = False)

        # TODO: Define the inner layer (FFTConv)

        self.mult_activation = activations.get(mult_act)
        dropout_fn = DropoutNd if tie_dropout else tf.keras.layers.Dropout
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else tf.keras.layers.Lambda(lambda x: x)

        self.output_linear = LinearActivation(self.d_model, 
                                              activation=activations.get(final_act), 
                                              transposed=False, 
                                              activate = True)


    def build(self, input_shape):
        # TODO: Create the layers that have parameters
        pass

    def call(self, inputs):
        # TODO: Define the forward pass
        pass

    
class DropoutNd(tf.keras.layers.Layer):
    def __init__(self, p: float = 0.5, tie=True, transposed=True, **kwargs):
        super(DropoutNd, self).__init__(**kwargs)
        self.p = p
        self.tie = tie
        self.transposed = transposed

    def call(self, inputs, training=None):
        if 0. < self.p < 1.:
            noise_shape = self._get_noise_shape(inputs)
            def dropped_inputs():
                return tf.nn.dropout(inputs, noise_shape=noise_shape, rate=self.p)
            return tf.keras.backend.in_train_phase(dropped_inputs, inputs, training=training)
        return inputs

    def _get_noise_shape(self, inputs):
        input_shape = tf.shape(inputs)
        noise_shape = input_shape
        if self.tie:
            noise_shape = input_shape[:2] + (1,) * (len(input_shape) - 2)
        if self.transposed:
            noise_shape = tf.concat([input_shape[:1], input_shape[-1:], input_shape[1:-1]], axis=0)
        return noise_shape