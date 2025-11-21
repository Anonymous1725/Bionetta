"""
Class for interpeting the L2 Unit Normalization activation layer
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np

from tf_bionetta.save.layers.interface import SaveableLayer
from tf_bionetta.layers.normalization.l2 import L2UnitNormalizationLayer


class SaveableL2UnitNormalization(SaveableLayer):
    """
    Class implementing the L2 unit normalization activation interpretation.
    """

    def __init__(self, layer: L2UnitNormalizationLayer) -> None:
        """
        Initializes the L2 unit normalization activation layer.

        Args:
            - layer (`L2UnitNormalizationLayer`): The layer to be interpreted.
        """

        assert isinstance(
            layer, L2UnitNormalizationLayer
        ), "Must use L2UnitNormalizationLayer only for interpretation"
        super().__init__(layer)

    def to_dictionary(self) -> Dict[str, Any]:
        """
        Converts the layer to a dictionary that can be saved to a JSON file.
        """

        layer = self._layer

        # NOTE: We always have the same shape (,input_neurons) for the L2UnitNormalizationLayer
        return {
            "type": "L2UnitNormalization",
            "name": layer.name,
            "input_shape": layer.input_shape[1],
            "input": "prev",
        }

    def to_weights(self) -> Dict[str, np.ndarray]:
        """
        Converts the layer to a dictionary that can be saved to a JSON file.
        """

        return super().to_weights()
