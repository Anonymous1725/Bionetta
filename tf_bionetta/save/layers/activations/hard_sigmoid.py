"""
Class for interpeting the Hard Sigmoid activation layer
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np

from tf_bionetta.save.layers.interface import SaveableLayer
from tf_bionetta.layers.hard_sigmoid import HardSigmoid


class SaveableHardSigmoid(SaveableLayer):
    """
    Class implementing the hard sigmoid activation interpretation.
    """

    def __init__(self, layer: HardSigmoid) -> None:
        """
        Initializes the hard sigmoid activation layer, defined as:

        `h-sigmoid(x) = ReLU6(x+3)/6`

        Args:
            - layer (`HardSigmoid`): The layer to be interpreted.
        """

        assert isinstance(
            layer, HardSigmoid
        ), "Must use HardSigmoid only for interpretation"
        super().__init__(layer)

    def to_dictionary(self) -> Dict[str, Any]:
        """
        Converts the layer to a dictionary that can be saved to a JSON file.
        """

        layer = self._layer

        # NOTE: We always have the same shape (,input_neurons) for the HardSigmoid activation layer
        return {
            "type": "HardSigmoid",
            "name": layer.name,
            "input_shape": layer.input_shape[1:],
            "input": "prev",
        }

    def to_weights(self) -> Dict[str, np.ndarray]:
        """
        Converts the layer to a dictionary that can be saved to a JSON file.
        """

        return super().to_weights()
