from typing import List, Tuple

import tensorflow as tf


def unpack_model_layers(model: tf.keras.Model) -> List[Tuple[tf.keras.layers.Layer, int]]:
    unpack_layers = []

    def _unpack(layer):
        if isinstance(layer, tf.keras.Model):
            for sublayer in layer.layers:
                _unpack(sublayer)
        else:
            unpack_layers.append(layer)
    
    _unpack(model)
    return unpack_layers