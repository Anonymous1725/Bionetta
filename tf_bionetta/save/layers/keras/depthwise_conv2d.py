"""
Class for interpeting the DepthwiseConv2D layer
"""

from __future__ import annotations

from typing import Dict, Any

import tensorflow as tf
import numpy as np

from tf_bionetta.save.layers.activations.convert import activation_to_dictionary
from tf_bionetta.save.layers.interface import SaveableLayer


class SaveableDepthwiseConv2D(SaveableLayer):
    """
    Class implementing the DepthwiseConv2D interpretation.
    """

    def __init__(self, layer: tf.keras.layers.DepthwiseConv2D) -> None:
        """
        Initializes the DepthwiseConv2D layer.

        Args:
            - layer (`tf.keras.layers.Layer`): The layer to be interpreted.
        """

        assert isinstance(
            layer, tf.keras.layers.DepthwiseConv2D
        ), "Only DepthwiseConv2D layers are supported"
        super().__init__(layer)

    def to_dictionary(self) -> Dict[str, Any]:
        """
        Converts the layer to a dictionary that can be saved to a JSON file.
        """

        layer = self._layer

        # NOTE: We always have the volume before and after the DepthwiseConv2D
        _, width_in, height_in, channels_in = layer.input_shape
        assert width_in == height_in, "Only square volumes are supported"

        _, width_out, height_out, channels_out = layer.output_shape
        assert width_out == height_out, "Only square volumes are supported"

        stride = layer.strides[0]
        assert layer.strides[0] == layer.strides[1], "Only square kernels are supported"

        kernel_size = layer.kernel_size[0]

        stride = layer.strides[0]
        assert layer.strides[0] == layer.strides[1], "Only square kernels are supported"

        dictionary = {
            "type": "DepthwiseConv2D",
            "use_bias": layer.use_bias,
            "name": layer.name,
            "length_in": width_in,
            "channels_in": channels_in,
            "length_out": width_out,
            "channels_out": channels_out,
            "filter": kernel_size,
            "stride": stride,
            "input": "prev",
        }
    
        activation = layer.activation
        if activation is not None and activation != tf.keras.activations.linear: 
            dictionary["post_activation"] = activation_to_dictionary(activation)

        return dictionary

    def to_weights(self) -> Dict[str, np.ndarray]:
        """
        Returns the weights of the layer in the form of dictionary.
        """

        return super().to_weights()
