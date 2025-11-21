"""
Class for interpeting the Conv2D layer
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple

import tensorflow as tf
import numpy as np

from tf_bionetta.save.layers.activations.convert import activation_to_dictionary
from tf_bionetta.save.layers.interface import SaveableLayer


class SaveableDense(SaveableLayer):
    """
    Class implementing the Dense interpretation.
    """
    
    def __init__(
        self, 
        layer: tf.keras.layers.Dense,
        input_shape: Optional[Tuple[int, ...]] = None,
    ) -> None:
        """
        Initializes the Dense layer.

        Args:
            - layer (`tf.keras.layers.Layer`): The layer to be interpreted.
            - input_shape (Tuple[int, ...], optional): The shape of the input to
              the layer. If None, the input shape is inferred from the layer.
        """

        assert isinstance(
            layer, tf.keras.layers.Dense
        ), "Only Dense layers are supported"
        super().__init__(layer)
        self._custom_input_shape = input_shape
        
        
    def to_dictionary(self) -> Dict[str, Any]:
        """
        Converts the layer to a dictionary that can be saved to a JSON file.
        """

        layer = self._layer

        # NOTE: We always have a flat input before the Dense layer
        input_neurons = layer.input_shape[1:] if self._custom_input_shape is None else self._custom_input_shape
        output_neurons = layer.units

        dictionary = {
            "type": "Dense",
            "name": layer.name,
            "input_neurons": input_neurons,
            "output_neurons": output_neurons,
            "input": "prev",
        }

        activation = layer.activation
        if activation is not None and activation != tf.keras.activations.linear: 
            dictionary["post_activation"] = activation_to_dictionary(activation)

        return dictionary

    def to_weights(self) -> Dict[str, np.ndarray]:
        """
        Converts the layer to a dictionary that can be saved to a JSON file.
        """

        return super().to_weights()
