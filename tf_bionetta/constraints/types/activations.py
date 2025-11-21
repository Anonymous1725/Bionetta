"""
Package with the specification of the constraints costs
for supported operations in the model.
"""

from __future__ import annotations

from typing import Callable
from enum import IntEnum
from math import log2

import tensorflow as tf

from tf_bionetta.activation_layer.hard_sigmoid import HardSigmoid
from tf_bionetta.activation_layer.hard_swish import HardSwish
from tf_bionetta.activation_layer.relu6 import ReLU6
from tf_bionetta.specs.backend_enums import (
    ProvingBackend, 
    Groth16, 
    UltraGroth
)

class ActivationOps(IntEnum):
    """
    Enum for the Bionetta-supported activation in the model. Each
    operation has a corresponding cost in terms of constraints.
    """

    # Supported non-linear operations natively
    RELU = 0
    SHIFT_LEAKY_RELU = 1
    HARD_SIGMOID = 2
    RELU6 = 3
    SQRT = 4
    LINEAR = 5
    HARD_SWISH = 6
    
    # Unsupported non-linear operations
    UNSUPPORTED_LEAKY_RELU = 7
    NOT_SUPPORTED = 8
    
    _ESTIMATION_COST: int = 512 # Default cost for an unknown non-linearity
    
    
    def cost(
        self, 
        backend: ProvingBackend,
    ) -> int:
        """
        Get the cost of the non-linear operation in terms of constraints.

        Args:
            backend (`ProvingBackend.ULTRAGROTH`): The backend to use for the cost estimation.

        Returns:
            int: The cost of the non-linear operation.
        """
        
        # NOTE: Costs below are specified manually based on the 
        # current Circom implementation
        if isinstance(backend, Groth16):
            return {
                ActivationOps.RELU: 255,
                ActivationOps.SHIFT_LEAKY_RELU: 256,
                ActivationOps.HARD_SIGMOID: 511,
                ActivationOps.RELU6: 262,
                ActivationOps.SQRT: 1024,
                ActivationOps.LINEAR: 0, # Linear activation has no cost
                ActivationOps.HARD_SWISH: 523,
                ActivationOps.UNSUPPORTED_LEAKY_RELU: ActivationOps._ESTIMATION_COST,
                ActivationOps.NOT_SUPPORTED: ActivationOps._ESTIMATION_COST,
            }.get(self.value, ActivationOps._ESTIMATION_COST)
            
        if isinstance(backend, UltraGroth):
            limb_size = backend.limb_size
            
            return {
                ActivationOps.RELU: 255 // limb_size, # ReLU cost decreases by the limb size
                ActivationOps.SHIFT_LEAKY_RELU: 256,
                ActivationOps.HARD_SIGMOID: 511,
                ActivationOps.RELU6: 510,
                ActivationOps.SQRT: 1024,
                ActivationOps.LINEAR: 0, # Linear activation has no cost
                ActivationOps.HARD_SWISH: 523,
                ActivationOps.UNSUPPORTED_LEAKY_RELU: ActivationOps._ESTIMATION_COST,
                ActivationOps.NOT_SUPPORTED: ActivationOps._ESTIMATION_COST,
            }.get(self.value, ActivationOps._ESTIMATION_COST)
            
        raise ValueError(f"Unsupported backend: {backend}")
            
    
    @classmethod
    def from_keras(cls, activation: tf.keras.layers.Layer | Callable) -> ActivationOps:
        """
        Get the corresponding non-linear operation from a Keras layer.

        Args:
            activation (`tf.keras.layers.Layer` or `Callable`): A Keras activation.

        Returns:
            ActivationOps: The corresponding non-linear operation.
        """

        if isinstance(activation, tf.keras.layers.Layer):
            if isinstance(activation, tf.keras.layers.ReLU):
                return cls.RELU
            if isinstance(activation, tf.keras.layers.LeakyReLU):
                if not valiate_leaky_relu_alpha(activation.alpha):
                    return cls.UNSUPPORTED_LEAKY_RELU
                return cls.SHIFT_LEAKY_RELU
            if isinstance(activation, ReLU6):
                return cls.RELU6
            if isinstance(activation, HardSigmoid):
                return cls.HARD_SIGMOID
            if isinstance(activation, HardSwish):
                return cls.HARD_SWISH

        # Assume that activation is `Callable` instead
        if activation == tf.keras.activations.relu:
            return cls.RELU
        if activation == tf.keras.activations.linear:
            return cls.LINEAR
        # if activation == tf.keras.activations.hard_sigmoid:
        #     return cls.HARD_SIGMOID
        # if activation == tf.keras.activations.relu6:
        #     return cls.RELU6
        # if activation == tf.keras.activations.hard_swish:
        #     return cls.HARD_SWISH
        
        # Otherwise, assume that the activation is not supported
        return cls.NOT_SUPPORTED


def valiate_leaky_relu_alpha(alpha: float) -> bool:
    """
    Validate the alpha value for the Leaky ReLU activation function.

    Args:
        alpha (`float`): The alpha value.

    Returns:
        bool: Whether the alpha value is valid.
    """

    if alpha < 0:
        return False  # Alpha must be greater than 0

    log_alpha = log2(alpha)
    return log_alpha.is_integer()

