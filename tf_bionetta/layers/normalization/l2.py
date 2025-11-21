"""
Package for unit normalization layer.
"""

from __future__ import annotations

from typing import Dict, Any

import tensorflow as tf

from tf_bionetta.layers.interface import BionettaLayer
from tf_bionetta.constraints.types.layer_complexity import LayerComplexity
from tf_bionetta.constraints.types.activations import ActivationOps


class L2UnitNormalizationLayer(BionettaLayer):
    """
    Layer for L2 unit normalization. A common layer in recognition tasks.
    """

    def __init__(self, radius: float = 1.0, axis: int = -1, **kwargs) -> None:
        """
        Initializes the L2 Normalization Layer. Optinally, the radius of the output
        vectors can be set.

        :param radius: The radius of the output vectors.
        :param axis: The axis along which the normalization is performed.
        :param kwargs: Additional arguments.
        """

        super(L2UnitNormalizationLayer, self).__init__(**kwargs)
        self.axis = axis
        self.radius = radius


    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Builds the layer. This method is called once the input shape is known.
        """

        input_shape = tf.TensorShape(input_shape)
        assert len(input_shape) == 2, "Currently, the L2UnitNormalizationLayer only supports the flat inputs."
        
        # Compute the complexity of the layer based on the input shape
        # (for the further constraints calculation)
        self.complexity = self.compute_complexity(input_shape)


    def compute_complexity(
        self, 
        input_shape: tf.TensorShape
    ) -> LayerComplexity:
        """
        Calculate the complexity of the layer: number of multiplications
        and non-linear operations, based on the input shape.
        Args:
            - input_shape (`tf.TensorShape`): The shape of the input tensor.
        Output:
            - `LayerComplexity`: An object containing the number of multiplications
              and non-linear operations in the layer.
        """

        input_neurons = input_shape[1]
        
        # Here, we need input_neurons multiplications to compute the L2 norm,
        # and then another input_neurons multiplications to scale each 
        # input component by the inverse of the L2 norm.
        return LayerComplexity(
            mul_ops=2*input_neurons,
            non_linear_ops=[(ActivationOps.SQRT, 1)]
        )


    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Forward pass.
        """

        return self.radius * tf.math.l2_normalize(inputs, axis=self.axis)


    def get_config(self) -> Dict[str, Any]:
        """
        Returns the configuration of the layer.
        """

        config = super(L2UnitNormalizationLayer, self).get_config()
        config.update({"axis": self.axis, "radius": self.radius})

        return config


    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> L2UnitNormalizationLayer:
        """
        Creates a layer from the configuration.
        """

        return cls(**config)
