"""
Class for interpeting the Conv2D layer
"""

from __future__ import annotations

from typing import Dict, Any 

import tensorflow as tf
import numpy as np

from tf_bionetta.save.layers.interface import SaveableLayer


class SaveableFlatten(SaveableLayer):
    """
    Class implementing the Flatten layer interpretation.
    """
    
    def __init__(self, layer: tf.keras.layers.Dense) -> None:
        """
        Initializes the Flatten layer.
        
        Args:
            - layer (`tf.keras.layers.Layer`): The layer to be interpreted.
        """
        
        assert isinstance(layer, tf.keras.layers.Flatten), "Only Flatten layers are supported"
        super().__init__(layer)
        
        
    def to_dictionary(self) -> Dict[str, Any]:
        """
        Converts the layer to a dictionary that can be saved to a JSON file.
        """
        
        layer = self._layer
        
        _, *input_shape = layer.input_shape
        _, output_neurons = layer.output_shape
        
        return {
            'type': 'Flatten',
            'name': layer.name,
            'input_shape': input_shape,
            'output_neurons': output_neurons,
            'input': 'prev',
        }
        
        
    def to_weights(self) -> Dict[str, np.ndarray]:
        """
        Converts the layer to a dictionary that can be saved to a JSON file.
        """
        
        return super().to_weights()
        