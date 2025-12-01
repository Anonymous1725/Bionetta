"""
Class for interpeting the Hard Sigmoid activation layer
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np

from tf_bionetta.save.layers.interface import SaveableLayer
from tf_bionetta.layers.hard_swish import HardSwish


class SaveableHardSwish(SaveableLayer):
    """
    Class implementing the hard sigmoid activation interpretation.
    """

    def __init__(self, layer: HardSwish) -> None:
        """
        Initializes the HardSwish activation layer

        Args:
            - layer (`HardSwish`): The layer to be interpreted.
        """

        assert isinstance(
            layer, HardSwish
        ), "Must use HardSwish only for interpretation"
        super().__init__(layer)

    def to_dictionary(self) -> Dict[str, Any]:
        """
        Converts the layer to a dictionary that can be saved to a JSON file.
        """

        layer = self._layer

        # NOTE: We always have the same shape (,input_neurons) for the HardSwish activation layer
        return {
            "type": "HardSwish",
            "name": layer.name,
            "input_shape": layer.input_shape[1:],
            "input": "prev",
        }

    def to_weights(self) -> Dict[str, np.ndarray]:
        """
        Converts the layer to a dictionary that can be saved to a JSON file.
        """

        return super().to_weights()
