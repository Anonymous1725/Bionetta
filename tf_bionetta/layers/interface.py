"""
File for the interface of the Bionetta-compatible layers
"""

import tensorflow as tf

from tf_bionetta.constraints.types.layer_complexity import LayerComplexity


class BionettaLayer(tf.keras.layers.Layer):
    """
    Base class for all the Bionetta-compatible layers.
    """

    def __init__(self, *args, **kwargs):
        super(BionettaLayer, self).__init__(*args, **kwargs)


    def compute_complexity(
        self, 
        input_shape: tf.TensorShape,
    ) -> LayerComplexity:
        """
        Calculates the complexity of the layer: number of multiplications
        and non-linear operations, based on the input shape.

        Args:
            - input_shape (`tf.TensorShape`): The shape of the input tensor.

        Output:
            - A `LayerComplexity` object containing the complexity of the layer.
        """

        raise NotImplementedError("Calculate constraints method not implemented.")
