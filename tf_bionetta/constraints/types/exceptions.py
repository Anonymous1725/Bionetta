"""
Class for handling exceptions related to layer constraints in TensorFlow Bionetta.
"""

from typing import Callable, Any

import tensorflow as tf


class UnsupportedActivation(Exception):
    """
    Custom error when the layer activation is not supported for constraint calculation.
    """
    
    def __init__(
        self, 
        activation: Callable[[Any], Any] | tf.keras.layers.Layer
    ) -> None:
        """
        Initializes the UnsupportedActivation error.

        Args:
            activation (`tf.keras.layers.Layer` or `Callable`): The unsupported activation.
        """
        
        self.activation = activation
        super().__init__(f"Unsupported activation function: {activation}. Needs estimation.")


class InvalidLeakyReLUAlpha(UnsupportedActivation):
    """
    Custom error when the alpha value for the Leaky ReLU activation function is invalid: that is,
    it is either negative or not the power of two.
    """
    
    def __init__(self, alpha: float) -> None:
        """
        Initializes the InvalidLeakyReLUAlpha error.

        Args:
            alpha (`float`): The invalid alpha value.
        """
        
        self.alpha = alpha
        super().__init__(tf.keras.layers.LeakyReLU(alpha=alpha))
