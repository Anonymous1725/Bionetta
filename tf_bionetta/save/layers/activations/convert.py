"""
Package for interpreting the activation layer
"""

from __future__ import annotations

from typing import Dict, Any

import tensorflow as tf

import tf_bionetta as tfb

from tf_bionetta.save.layers.activations.leaky_relu import SaveableLeakyReLU


def activation_to_dictionary(layer: tf.keras.layers.Layer) -> Dict[str, Any]:
    """
    Converts the activation layer to the saveable layer.

    Args:
        - layer (`tf.keras.layers.Layer`): The activation layer to be converted.

    Returns:
        - saveable_layer (`SaveableLayer`): The saveable layer.
    """

    if isinstance(layer, tf.keras.layers.ReLU) or layer == tf.keras.activations.relu:
        return {
            "name": "ReLU",
        }
    elif isinstance(layer, tf.keras.layers.LeakyReLU):
        return {
            "name": "LeakyReLU",
            "shift": SaveableLeakyReLU._calculate_shift(layer.alpha),
        }
    elif isinstance(layer, tfb.layers.ReLU6):
        return {
            "name": "ReLU6",
        }
    elif isinstance(layer, tfb.layers.HardSwish):
        return {
            "name": "HardSwish",
        }
    elif isinstance(layer, tfb.layers.HardSigmoid):
        return {
            "name": "HardSigmoid",
        }
    elif layer == tf.keras.activations.hard_sigmoid:
        raise ValueError(f"Used tf.keras.activations.hard_sigmoid is not the same(for tf version 2.13.1) as Bionetta Framework one, so try tfb.layers.HardSigmoid() instead")
    
    raise ValueError(f"Unsupported activation layer: {layer.__class__.__name__}")
