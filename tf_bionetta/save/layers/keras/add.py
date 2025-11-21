"""
Class for interpeting the keras Add layer
"""

from __future__ import annotations

from typing import Dict, Any

import tensorflow as tf
import numpy as np

from tf_bionetta.save.layers.interface import SaveableLayer


class SaveableAdd(SaveableLayer):
    """
    Class implementing the Keras Add layer interpretation.
    """

    def __init__(self, layer: tf.keras.layers.Add) -> None:
        """
        Initializes the Keras Add layer.

        Args:
            - layer (`tf.keras.layers.Add`): The layer to be interpreted.
        """

        assert isinstance(
            layer, tf.keras.layers.Add
        ), "keras.layers.Add layer must be specified"
        super().__init__(layer)


    def to_dictionary(self) -> Dict[str, Any]:
        """
        Converts the layer to a dictionary that can be saved to a JSON file.
        """

        layer = self._layer

        # NOTE: We always have the same shape for the Add layer
        # Print all attributes of layer
        assert len(layer.input_shape) >= 2, "Add layer must have at least two inputs"

        input_layers = []
        for input_layer in layer.input:
            # Example of name: block_4_add/add:0
            if input_layer.name.split('/')[1].split(':')[0] == 'add':
                input_layers.append(f"tensor_{input_layer.name}")
            else:
                input_layers.append(input_layer.name)

        return {
            "type": "TensorAdd",
            "name": f"tensor_{layer.name}",
            "input_shape": layer.input_shape[0][1:],
            "input": input_layers,
        }


    def to_weights(self) -> Dict[str, np.ndarray]:
        """
        Saves the weights of the layer to the dictionary.
        """

        return super().to_weights()
