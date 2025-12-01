"""
Class for interpeting the AveragePooling2D layer
"""

from __future__ import annotations

from typing import Dict, Any

import tensorflow as tf
import numpy as np

from tf_bionetta.save.layers.interface import SaveableLayer


class SaveableAveragePooling2D(SaveableLayer):
    """
    Class implementing the AveragePooling2D interpretation.
    """

    def __init__(self, layer: tf.keras.layers.AveragePooling2D) -> None:
        """
        Initializes the AveragePooling2D layer.

        Args:
            - layer (`tf.keras.layers.Layer`): The layer to be interpreted.
        """

        assert isinstance(
            layer, tf.keras.layers.AveragePooling2D
        ), "Only AveragePooling2D layers are supported"
        super().__init__(layer)

    def to_dictionary(self) -> Dict[str, Any]:
        """
        Converts the layer to a dictionary that can be saved to a JSON file.
        """

        layer = self._layer

        return {
            "type": "AveragePooling2D",
            "name": layer.name,
            "func": "Avg",
            "input": "prev",
            "out_shape": layer.output_shape[1:],
        }

    def to_weights(self) -> Dict[str, np.ndarray]:
        """
        Saves the weights of the layer.
        """

        return super().to_weights()
