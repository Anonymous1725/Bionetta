"""
Class for interpeting the keras Add layer
"""

from __future__ import annotations

from typing import Dict, Any

import tensorflow as tf
import numpy as np

from tf_bionetta.save.layers.interface import SaveableLayer


class SaveableZeroPadding2D(SaveableLayer):
    """
    Class implementing the Keras Add layer interpretation.
    """

    def __init__(self, layer: tf.keras.layers.ZeroPadding2D) -> None:
        """
        Initializes the Keras Add layer.

        Args:
            - layer (`tf.keras.layers.Add`): The layer to be interpreted.
        """

        assert isinstance(
            layer, tf.keras.layers.ZeroPadding2D
        ), "keras.layers.Add layer must be specified"
        super().__init__(layer)


    def to_dictionary(self) -> Dict[str, Any]:
        """
        Converts the layer to a dictionary that can be saved to a JSON file.
        """

        layer = self._layer

        # Channels is equal
        _, width_in, height_in, _ = layer.input_shape
        assert width_in == height_in, "Only square volumes are supported"

        _, width_out, height_out, _ = layer.output_shape
        assert width_out == height_out, "Only square volumes are supported"
        
        (top_pad, bottom_pad), (left_pad, right_pad) = layer.padding

        return {
            "type": "ZeroPadding2D",
            "name": layer.name,
            "input_shape": layer.input_shape[1:],
            "top_pad": top_pad,
            "bottom_pad": bottom_pad,
            "left_pad": left_pad,
            "right_pad": right_pad,
            "input": "prev",
        }


    def to_weights(self) -> Dict[str, np.ndarray]:
        """
        Saves the weights of the layer to the dictionary.
        """

        return super().to_weights()
