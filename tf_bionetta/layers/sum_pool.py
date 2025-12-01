"""
Layer for the Sum Pooling operation.
"""

import tensorflow as tf

from tf_bionetta.layers.interface import BionettaLayer
from tf_bionetta.constraints.types.layer_complexity import LayerComplexity


class SumPool(BionettaLayer):
    """
    Class implementing the Sum Pooling layer, which sums all the elements in the
    input tensor, in contrast to the commonly used `AveragePooling2D`. This
    ensures no precision increase due to the multiplication by `1/n` where `n`
    is the number of elements in the tensor.
    """

    def __init__(self):
        super(SumPool, self).__init__()


    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Builds the layer. This method is called once the input shape is known.
        """

        self.complexity = self.compute_complexity(input_shape)


    def compute_complexity(self, _: tf.TensorShape) -> LayerComplexity:
        """
        Calculate the number of constraints for the layer.
        
        Args:
            - input_shape (`tf.TensorShape`): The shape of the input tensor.
            
        Returns:
            - `LayerComplexity`: An object containing the number of multiplications
              and non-linear operations in the layer.
        """

        return LayerComplexity(
            mul_ops=0,
            non_linear_ops=None
        )


    def call(self, x):
        return tf.reduce_sum(x)
