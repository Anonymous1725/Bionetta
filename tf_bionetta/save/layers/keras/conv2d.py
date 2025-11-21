"""
Class for interpeting the Conv2D layer
"""

from __future__ import annotations

from typing import Dict, Any

import tensorflow as tf
import numpy as np

from tf_bionetta.save.layers.activations.convert import activation_to_dictionary
from tf_bionetta.save.layers.interface import SaveableLayer


class SaveableConv2D(SaveableLayer):
    """
    Class implementing the Conv2D interpretation.
    """

    def __init__(self, layer: tf.keras.layers.Conv2D) -> None:
        """
        Initializes the Conv2D layer.

        Args:
            - layer (`tf.keras.layers.Layer`): The layer to be interpreted.
        """

        assert isinstance(
            layer, tf.keras.layers.Conv2D
        ), "Only Conv2D layers are supported"
        super().__init__(layer)

    def to_dictionary(self) -> Dict[str, Any]:
        """
        Converts the layer to a dictionary that can be saved to a JSON file.
        """

        layer = self._layer

        # NOTE: We always have the volume before and after the Conv2D
        # Getting input shape parameters
        _, width_in, height_in, channels_in = layer.input_shape
        assert width_in == height_in, "Only square volumes are supported"

        # Getting output shape parameters
        _, width_out, height_out, channels_out = layer.output_shape
        assert width_out == height_out, "Only square volumes are supported"

        # Getting the kernel size
        kernel_size = layer.kernel_size[0]
        assert layer.kernel_size[0] == layer.kernel_size[1], "Only square kernels are supported"

        stride = layer.strides[0]
        assert layer.strides[0] == layer.strides[1], "Only square kernels are supported"

        if '/' in layer.input.name and layer.input.name.split('/')[1].split(':')[0] == 'add':
            name = f'tensor_{layer.input.name}'
        else:
            name = layer.input.name

        dictionary = {
            "type": "Conv2D",
            "use_bias": layer.use_bias,
            "name": layer.name,
            "length_in": width_in,
            "channels_in": channels_in,
            "length_out": width_out,
            "channels_out": channels_out,
            "filter": kernel_size,
            "stride": stride,
            "input": name,
        }

        activation = layer.activation
        if activation is not None and activation != tf.keras.activations.linear: 
            dictionary["post_activation"] = activation_to_dictionary(activation)

        return dictionary

    def to_weights(self) -> Dict[str, np.ndarray]:
        """
        Saves the weights of the layer to a dictionary.
        """

        return super().to_weights()
