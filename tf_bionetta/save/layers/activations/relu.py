"""
Class for interpeting the ReLU activation layer
"""

from __future__ import annotations

from typing import Dict, Any

import tensorflow as tf
import numpy as np

from tf_bionetta.save.layers.interface import SaveableLayer


class SaveableReLU(SaveableLayer):
    """
    Class implementing the ReLU activation interpretation.
    """

    def __init__(self, layer: tf.keras.layers.ReLU) -> None:
        """
        Initializes the ReLU activation layer.

        Args:
            - layer (`tf.keras.layers.ReLU`): The layer to be interpreted.
        """

        assert isinstance(
            layer, tf.keras.layers.ReLU
        ), "Only ReLU layer can be specified"
        super().__init__(layer)

    def to_dictionary(self) -> Dict[str, Any]:
        """
        Converts the layer to a dictionary that can be saved to a JSON file.
        """

        return {
            "type": "ReLU",
            "name": self._layer.name,
            "input_shape": self._layer.input_shape[1:],
            "input": "prev",
        }

    def to_weights(self) -> Dict[str, np.ndarray]:
        """
        Converts the layer to a dictionary that can be saved to a JSON file.
        """

        return super().to_weights()
