from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf

from tf_bionetta.constraints.types.activations import ActivationOps
from tf_bionetta.constraints.types.layer_complexity import LayerComplexity
from tf_bionetta.layers.interface import BionettaLayer
from tf_bionetta.activation_layer.hard_swish import HardSwish

class HardSwish(BionettaLayer, HardSwish):
    """
    Class implementing the Hard Swish activation function, given by:

    `f(x) = x * max(0, min(1, x + 3) / 6)`
    """

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Builds the layer. This method is called once the input shape is known.
        For HardSwish, this function simply computes the number of constraints
        """

        # Compute the complexity of the layer based on the input shape
        # (for the further constraints calculation)
        self.complexity = self.compute_complexity(input_shape)

    def compute_complexity(
        self, 
        input_shape: tf.TensorShape
    ) -> LayerComplexity:
        """
        Calculates the complexity of the layer.
        
        Args:
            _: (`tf.TensorShape`): The shape of the input tensor. Not used in this layer.
        
        Returns:
            LayerComplexity: The estimated complexity of the layer.
        """
        num_elements = np.prod([d for d in input_shape if d is not None])
        return LayerComplexity(
            mul_ops=0,
            non_linear_ops=[
                (self, num_elements)
            ],
        )

    def calculate_constraints(
        self, 
        input_shape: tf.TensorShape
    ) -> LayerComplexity:
        """
        Calculate the number of constraints for the layer.
        
        Args:
            - input_shape (`tf.TensorShape`): The shape of the input tensor.
        
        Returns:
            - `LayerComplexity`: An object containing the number of multiplications
              and non-linear operations in the layer.
        """

        input_shape = tf.TensorShape(input_shape)
        # We find the product of all components except for the batch dimension
        input_neurons = tf.reduce_prod(input_shape[1:])

        return LayerComplexity(
            mul_ops=0,
            non_linear_ops=[
                (ActivationOps.HARD_SWISH, input_neurons)
            ],
        )
