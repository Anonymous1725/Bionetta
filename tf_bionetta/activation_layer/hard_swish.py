from __future__ import annotations

import tensorflow as tf


class HardSwish(tf.keras.layers.Layer):
    """
    Class implementing the Hard Swish activation function, given by:

    `f(x) = x * max(0, min(1, x + 3) / 6)`
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the Hard Swish layer.
        """

        super(HardSwish, self).__init__(*args, **kwargs)
    
    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Computes the forward pass of the Hard Sigmoid layer.
        """

        return x * tf.nn.relu6(x + 3) / 6