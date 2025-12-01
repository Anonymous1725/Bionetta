from __future__ import annotations

import tensorflow as tf


class HardSigmoid(tf.keras.layers.Layer):
    """
    Class implementing Hard Sigmoid activation function, given by:

    `f(x) = max(0, min(1, x + 3) / 6)`
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the Hard Sigmoid layer.
        """

        super(HardSigmoid, self).__init__(*args, **kwargs)
    
    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Computes the forward pass of the Hard Sigmoid layer.
        """

        return tf.nn.relu6(x + 3) / 6