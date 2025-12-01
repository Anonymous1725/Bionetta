"""
Interface that all layers should implement to be used in the
architecture specification
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

import tensorflow as tf
import numpy as np


class SaveableLayer(ABC):
    """
    Interface that all layers should implement to be used in the
    architecture specification
    """

    @abstractmethod
    def __init__(self, layer: tf.keras.layers.Layer) -> None:
        """
        Initializes the layer.

        Args:
            - layer (`tf.keras.layers.Layer`): The layer to be interpreted.
        """

        self._layer = layer


    @abstractmethod
    def to_dictionary(self) -> Dict[str, Any] | None:
        """
        Converts the layer to a dictionary that can be saved to a JSON file. If the layer
        does not any specification parameters (e.g., they are not needed such as the Dropout layer),
        the function can return None.
        """

        pass


    @abstractmethod
    def to_weights(self) -> Dict[str, np.ndarray]:
        """
        Convers the layer to a dictionary that contains the weights of the layer.
        """

        weights: Dict[str, np.ndarray] = {}

        for weight in self._layer.weights:
            weights[weight.name] = weight.numpy().tolist()

        return weights
