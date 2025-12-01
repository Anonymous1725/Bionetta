from __future__ import annotations

import tensorflow as tf


class ReLU6(tf.keras.layers.Layer):
    """
    Class implementing the ReLU6 activation function, given by:

    `f(x) = max(0, min(6, x))`
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the ReLU6 layer.
        """

        super(ReLU6, self).__init__(*args, **kwargs)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Computes the forward pass of the ReLU6 layer.
        """

        return tf.nn.relu6(x)