"""
Class for interpeting the BatchNormalization layer
"""

from __future__ import annotations

from typing import Dict, Any, Tuple

import tensorflow as tf
import numpy as np

from tf_bionetta.save.layers.interface import SaveableLayer


class SaveableBatchNormalization(SaveableLayer):
    """
    Class implementing the BatchNormalization interpretation.
    """

    def __init__(self, layer: tf.keras.layers.BatchNormalization) -> None:
        """
        Initializes the BatchNormalization layer.

        Args:
            - layer (`tf.keras.layers.Layer`): The layer to be interpreted.
        """

        assert isinstance(
            layer, tf.keras.layers.BatchNormalization
        ), "Only BatchNormalization layers are supported"
        super().__init__(layer)

    @staticmethod
    def postprocess_batch_normalization(
        bn_layer: tf.keras.layers.BatchNormalization,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Given the batch normalization layer, instead of saving
        four parameters: gamma, beta, moving mean, and moving variance,
        we only save linear transformation parameters: alpha and beta.
        Namely, since
        ```
        BN(x) = gamma * (x - moving_mean) / sqrt(moving_variance + epsilon) + beta
        ```
        We have that `BN(x) = alpha * x + beta`, where
        ```
        alpha = gamma / sqrt(moving_variance + epsilon)
        beta = beta - gamma * moving_mean / sqrt(moving_variance + epsilon)
        ```

        Arguments:
            - bn_layer (tf.keras.layers.BatchNormalization) - BatchNormalization layer

        Output:
            - alpha (tf.Tensor) - alpha parameter(s)
            - beta (tf.Tensor) - beta parameter(s)
        """

        gamma, beta, moving_mean, moving_variance = bn_layer.get_weights()
        epsilon = bn_layer.epsilon

        std = np.sqrt(moving_variance + epsilon)
        alpha = gamma / std
        beta = beta - gamma * moving_mean / std

        return alpha, beta

    def to_dictionary(self) -> Dict[str, Any]:
        """
        Converts the layer to a dictionary that can be saved to a JSON file.
        """

        layer = self._layer

        # NOTE: Check the previous layer for the shape. If it is a volume,
        # we have to specify the volume shape. If it is a dense layer, we have
        # to specify the number of neurons.
        return {
            "type": "BatchNormalization",
            "name": layer.name,
            "input_shape": layer.input_shape[1:],
            "input": "prev",
        }

    def to_weights(self) -> Dict[str, np.ndarray]:
        """
        Saves the weights of the layer to a dictionary.
        """

        alpha, beta = SaveableBatchNormalization.postprocess_batch_normalization(
            self._layer
        )
        return {"alpha": alpha.tolist(), "beta": beta.tolist()}
