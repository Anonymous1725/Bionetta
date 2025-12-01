"""
Class for interpeting the GlobalAveragePooling2D layer
"""

from __future__ import annotations

from typing import Dict, Any

import tensorflow as tf
import numpy as np

from tf_bionetta.save.layers.interface import SaveableLayer


class SaveableGlobalAveragePooling2D(SaveableLayer):
    """
    Class implementing the GlobalAveragePooling2D interpretation.
    """

    def __init__(self, layer: tf.keras.layers.GlobalAveragePooling2D) -> None:
        """
        Initializes the EDLightConv2D layer.

        Args:
            - layer (`tf.keras.layers.Layer`): The layer to be interpreted.
        """

        assert isinstance(
            layer, tf.keras.layers.GlobalAveragePooling2D
        ), "Only GlobalAveragePooling2D layers are supported"
        super().__init__(layer)

    def to_dictionary(self) -> Dict[str, Any]:
        """
        Converts the layer to a dictionary that can be saved to a JSON file.
        """

        layer = self._layer

        # NOTE: We always have the volume before the GlobalAveragePooling2D
        _, width_in, height_in, channels_in = layer.input_shape
        assert width_in == height_in, "Only square volumes are supported"

        return {
            "type": "GlobalAveragePooling2D",
            "name": layer.name,
            "length": width_in,
            "channels": channels_in,
            "input": "prev",
        }

    def to_weights(self) -> Dict[str, np.ndarray]:
        """
        Converts the layer to a dictionary that can be saved to a JSON file.
        """

        return super().to_weights()
