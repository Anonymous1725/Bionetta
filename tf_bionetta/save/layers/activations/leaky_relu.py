"""
Class for interpeting the LeakyReLU activation layer
"""

from __future__ import annotations

from typing import Dict, Any

import tensorflow as tf
import numpy as np

from tf_bionetta.save.layers.interface import SaveableLayer


class SaveableLeakyReLU(SaveableLayer):
    """
    Class implementing the LeakyReLU activation interpretation.
    """

    def __init__(self, layer: tf.keras.layers.LeakyReLU) -> None:
        """
        Initializes the LeakyReLU activation layer.

        Args:
            - layer (`tf.keras.layers.LeakyReLU`): The layer to be interpreted.
        """

        assert isinstance(
            layer, tf.keras.layers.LeakyReLU
        ), "Only LeakyReLU layers are supported"
        super().__init__(layer)

    @staticmethod
    def _calculate_shift(alpha: float) -> int:
        """
        Given the alpha parameter of the LeakyReLU activation function,
        calculates the shift parameter for the LeakyReLU activation function.

        Arguments:
            - alpha (float) - alpha parameter of the LeakyReLU activation function

        Output:
            - shift (int) - shift parameter
        """

        assert alpha > 0, "Alpha parameter must be positive"

        log2_alpha = np.log2(alpha)
        assert (
            log2_alpha.is_integer()
        ), "Alpha parameter must be the power of two, otherwise the circuit is not compilable"

        return -round(log2_alpha)

    def to_dictionary(self) -> Dict[str, Any]:
        """
        Converts the layer to a dictionary that can be saved to a JSON file.
        """

        layer = self._layer

        return {
            "type": "LeakyReLU",
            "name": layer.name,
            "shift": SaveableLeakyReLU._calculate_shift(layer.alpha),
            "input_shape": layer.input_shape[1:],
            "input": "prev",
        }

    def to_weights(self) -> Dict[str, np.ndarray]:
        """
        Saves the weights of the layer to the dictionary.
        """
        return super().to_weights()
