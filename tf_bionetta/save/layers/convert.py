"""
Module for converting the Keras API Layer to the SaveableLayer
"""

from typing import Optional

import tensorflow as tf

from tf_bionetta.save.layers.interface import SaveableLayer

# Import all saveable layers
from tf_bionetta.layers import *

# Import all custom layers
from tf_bionetta.save.layers.keras import *
from tf_bionetta.save.layers.custom import *
from tf_bionetta.save.layers.activations import *


def is_uninterpretable_layer(layer: tf.keras.layers.Layer) -> bool:
    """
    Checks if the layer is uninterpretable. That is, when the layer does not
    need to be neither saved nor specified in the architecture.
    """
    
    return (
        isinstance(layer, (tf.keras.layers.Dropout,
                            tf.keras.layers.Flatten))
        or isinstance(layer, tf.keras.layers.InputLayer)
    )


def to_saveable_layer(
    layer: tf.keras.layers.Layer,
    previous_layer: Optional[tf.keras.layers.Layer] = None
) -> SaveableLayer:
    """
    Converts the Keras API Layer to the SaveableLayer. 
    
    Args:
        - layer (`tf.keras.layers.Layer`): The layer to be converted.
        - previous_layer (`tf.keras.layers.Layer`): The previous layer in the model.
    """

    # Keras API layers
    if isinstance(layer, tf.keras.layers.Conv2D):
        return SaveableConv2D(layer)
    elif isinstance(layer, tf.keras.layers.DepthwiseConv2D):
        return SaveableDepthwiseConv2D(layer)
    elif isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
        return SaveableGlobalAveragePooling2D(layer)
    elif isinstance(layer, tf.keras.layers.AveragePooling2D):
        return SaveableAveragePooling2D(layer)
    elif isinstance(layer, tf.keras.layers.Dense):
        # If the previous layer is a Flatten layer, we need to pass the input shape
        if previous_layer is not None and isinstance(previous_layer, tf.keras.layers.Flatten):
            return SaveableDense(layer, input_shape=previous_layer.input_shape[1:])
        
        return SaveableDense(layer)
    elif isinstance(layer, tf.keras.layers.BatchNormalization):
        return SaveableBatchNormalization(layer)
    elif isinstance(layer, tf.keras.layers.Add):
        return SaveableAdd(layer)
    elif isinstance(layer, tf.keras.layers.ZeroPadding2D):
        return SaveableZeroPadding2D(layer)
    elif isinstance(layer, tf.keras.layers.MaxPool2D):
        return SaveableMaxPool2D(layer)

    # Activation layers
    elif isinstance(layer, tf.keras.layers.ReLU):
        return SaveableReLU(layer)
    elif isinstance(layer, tf.keras.layers.LeakyReLU):
        return SaveableLeakyReLU(layer)
    elif isinstance(layer, L2UnitNormalizationLayer):
        return SaveableL2UnitNormalization(layer)
    elif isinstance(layer, HardSigmoid):
        return SaveableHardSigmoid(layer)
    elif isinstance(layer, HardSwish):
        return SaveableHardSwish(layer)
    elif isinstance(layer, ReLU6):
        return SaveableReLU6(layer)

    # Custom layers
    elif isinstance(layer, EDLight2DConv):
        return SaveableEDLightConv2D(layer)
    elif isinstance(layer, SEHeavyBlock):
        return SaveableSEHeavyBlock(layer)
    elif isinstance(layer, SELightBlock):
        return SaveableSELightBlock(layer)

    raise ValueError(f"Layer {layer} is not supported")
