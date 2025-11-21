"""
Class for interpeting the Hard Sigmoid activation layer
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np

from tf_bionetta.save.layers.interface import SaveableLayer
from tf_bionetta.layers.relu6 import ReLU6


class SaveableReLU6(SaveableLayer):
    """
    Class implementing the hard sigmoid activation interpretation.
    """

    def __init__(self, layer: ReLU6) -> None:
        """
        Initializes the ReLU6 activation layer

        Args:
            - layer (`ReLU6`): The layer to be interpreted.
        """

        assert isinstance(
            layer, ReLU6
        ), "Must use ReLU6 only for interpretation"
        super().__init__(layer)

    def to_dictionary(self) -> Dict[str, Any]:
        """
        Converts the layer to a dictionary that can be saved to a JSON file.
        """

        layer = self._layer

        # NOTE: We always have the same shape (,input_neurons) for the ReLU6 activation layer
        return {
            "type": "ReLU6",
            "name": layer.name,
            "input_shape": layer.input_shape[1:],
            "input": "prev",
        }

    def to_weights(self) -> Dict[str, np.ndarray]:
        """
        Converts the layer to a dictionary that can be saved to a JSON file.
        """

        return super().to_weights()
