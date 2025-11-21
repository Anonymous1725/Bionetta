"""
A small package for proper LeakyReLU initialization (shifted ReLU, where
the alpha value is the power of two).
"""

import tensorflow as tf


def ShiftReLU(shift: int) -> tf.keras.layers.Layer:
    """
    Shifted ReLU activation function.

    Args:
        shift: The shift value.

    Returns:
        A shifted ReLU activation function.
    """

    assert shift > 0, "The shift value must be greater than zero."

    return tf.keras.layers.LeakyReLU(alpha=1/(2**shift))
