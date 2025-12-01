"""
Class that stores the complexity of a layer in terms of the number of operations it uses.
"""

from typing import List, Tuple

import tensorflow as tf

from tf_bionetta.constraints.types.activations import ActivationOps
from tf_bionetta.constraints.types.exceptions import (
    UnsupportedActivation,
    InvalidLeakyReLUAlpha
)
from tf_bionetta.specs.backend_enums import ProvingBackend


class LayerComplexity:
    """
    Class that stores the complexity of a layer in terms of the number of 
    operations it uses.
    
    Namely, we store:
        - The number of non-linear operations
        - The number of signal multiplications
    """
    
    
    def __init__(
        self, 
        mul_ops: int = 0,
        non_linear_ops: List[Tuple[tf.keras.layers.Layer | ActivationOps, int]] = None, 
    ) -> None:
        """
        Initializes the LayerComplexity object.
        
        Args:
            - non_linear_ops (`List[Tuple[tf.keras.layers.Layer | ActivationOps, int]]`): 
                A list of tuples where each tuple contains an activation and 
                the number of times it is applied in the layer.
            - mul_ops (`int`): The number of signal multiplications in the layer.
        """
        
        self.non_linear_ops = non_linear_ops
        self.mul_ops = mul_ops
        
    
    def compute_constraints(
        self, 
        backend: ProvingBackend,
    ) -> Tuple[int, Exception | None]:
        """
        Computes the constraints for the layer based on the backend provided.
       
        It MUST set constraints_estimated if the constraints cannot be
        calculated precisely.

        Args:
            - backend (`ProvingBackend`): The backend to use for computing the constraints.

        Output:
            - The number of constraints for the layer.
            - An exception if the constraints cannot be calculated.
        """
        
        constaints, exception = self.mul_ops, None
        
        for activation, count in self.non_linear_ops:
            activation_op = None
            if not isinstance(activation, ActivationOps):
                # If the activation is a layer, we need to get its activation operation
                activation_op = ActivationOps.from_keras(activation)
            else:
                activation_op = activation
            
            # Raise errors if the activation is not supported or incorrect
            if activation_op == ActivationOps.NOT_SUPPORTED:
                exception = UnsupportedActivation(activation)
            if activation_op == ActivationOps.UNSUPPORTED_LEAKY_RELU:
                exception = InvalidLeakyReLUAlpha(alpha=activation.alpha)
            
            # Otherwise, we can compute the cost of the activation
            cost_per_activation = activation_op.cost(backend)
            constaints += count * cost_per_activation

        self.constraints_estimated = exception is not None
        return constaints, exception
