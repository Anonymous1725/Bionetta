"""
Package containing the final `VerifyDist` layer that is
used for Circom code compliation interpretation.
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np
import tensorflow as tf

from tf_bionetta.save.layers.interface import SaveableLayer


class SaveableVerifyDist(SaveableLayer):
    """
    Class implementing the VerifyDist interpretation.
    """

    def __init__(self, layer: tf.keras.layers.Layer) -> None:
        """
        Initializes the VerifyDist layer.

        Args:
            - layer (`tf.keras.layers.Layer`): The layer to be interpreted.
        """

        pass  # No need to do anything here.

    def to_dictionary(self, output_size: int) -> Dict[str, Any]:
        """
        Converts the layer to a dictionary that can be saved to a JSON file.
        """

        return {
            "type": "VerifyDist",
            "name": "verify_dist",
            "length_in": output_size,
            "input": "prev",
        }

    def to_weights(self) -> Dict[str, np.ndarray]:
        """
        Converts the layer to a dictionary that can be saved to a JSON file.
        """

        return {}
