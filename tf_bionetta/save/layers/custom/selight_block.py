"""
Class for interpeting the SELightBlock layer
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np

from tf_bionetta.save.layers.interface import SaveableLayer
from tf_bionetta.layers.se.light import SELightBlock
from tf_bionetta.save.layers.activations.convert import activation_to_dictionary


class SaveableSELightBlock(SaveableLayer):
    """
    Class implementing the SELightBlock interpretation.
    """

    def __init__(self, layer: SELightBlock) -> None:
        """
        Initializes the saveable SELightBlock layer.

        Args:
            - layer (`SELightBlock`): The layer to be interpreted.
        """

        assert isinstance(layer, SELightBlock), "Only SELightBlock layers are supported"
        super().__init__(layer)

    def to_dictionary(self) -> Dict[str, Any]:
        """
        Converts the layer to a dictionary that can be saved to a JSON file.
        """

        layer = self._layer

        # NOTE: We always have the volume before the SELightBlock
        _, width_in, height_in, channels_in = layer.input_shape

        # TODO: How to specify the activation function? Also precision parameters
        return {
            "type": "SELightBlock",
            "name": layer.name,
            "width": width_in,
            "height": height_in,
            "channels": channels_in,
            "hidden_size": layer.hidden_units,
            "activation": activation_to_dictionary(layer.activation),
            "input": "prev",
        }

    def to_weights(self) -> Dict[str, np.ndarray]:
        """
        Converts the layer to a dictionary that can be saved to a JSON file.
        """

        return super().to_weights()
