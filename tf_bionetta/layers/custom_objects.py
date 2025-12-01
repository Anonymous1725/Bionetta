"""
A helper function to return a dictionary of custom objects for loading models.
"""

from typing import Dict

from tf_bionetta.layers.interface import BionettaLayer
from tf_bionetta.layers.conv.edheavy import EDHeavy2DConv
from tf_bionetta.layers.conv.edlight import EDLight2DConv
from tf_bionetta.layers.se.heavy import SEHeavyBlock
from tf_bionetta.layers.se.light import SELightBlock
from tf_bionetta.layers.hard_sigmoid import HardSigmoid
from tf_bionetta.layers.relu6 import ReLU6
from tf_bionetta.layers.hard_swish import HardSwish
from tf_bionetta.layers.normalization.class_projection import ClassProjectionLayer
from tf_bionetta.layers.normalization.l2 import L2UnitNormalizationLayer


def get_custom_objects() -> Dict[str, BionettaLayer]:
    """
    Returns a dictionary of custom objects for loading Bionetta-compatible
    models using Keras API
    """

    return {
        "EDHeavy2DConv": EDHeavy2DConv,
        "EDLight2DConv": EDLight2DConv,
        "SEHeavyBlock": SEHeavyBlock,
        "SELightBlock": SELightBlock,
        "HardSigmoid": HardSigmoid,
        "HardSwish": HardSwish,
        "ReLU6": ReLU6,
        "ClassProjectionLayer": ClassProjectionLayer,
        "L2UnitNormalizationLayer": L2UnitNormalizationLayer,
    }
