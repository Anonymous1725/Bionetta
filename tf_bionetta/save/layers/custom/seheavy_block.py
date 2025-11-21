"""
Class for interpeting the SEHeavyBlock layer
"""

from __future__ import annotations

from typing import Dict, Any

import tensorflow as tf
import numpy as np

from tf_bionetta.save.layers.interface import SaveableLayer
from tf_bionetta.layers.se.heavy import SEHeavyBlock
from tf_bionetta.save.layers.keras.batch_normalization import (
    SaveableBatchNormalization,
)  # For postprocessing BN layers
from tf_bionetta.save.layers.activations.convert import activation_to_dictionary


class SaveableSEHeavyBlock(SaveableLayer):
    """
    Class implementing the SEHeavyBlock interpretation.
    """

    def __init__(self, layer: SEHeavyBlock) -> None:
        """
        Initializes the EDLightConv2D layer.

        Args:
            - layer (`tf.keras.layers.Layer`): The layer to be interpreted.
        """

        assert isinstance(layer, SEHeavyBlock), "Only SEHeavyBlocks are supported"
        super().__init__(layer)

    def to_dictionary(self) -> Dict[str, Any]:
        """
        Converts the layer to a dictionary that can be saved to a JSON file.
        """

        layer = self._layer

        # NOTE: We always have the volume before the SEHeavyBlock
        _, width_in, height_in, channels_in = layer.input_shape
        assert width_in == height_in, "Only square volumes are supported"

        return {
            "type": "SEHeavyBlock",
            "name": layer.name,
            "length": width_in,
            "channels": channels_in,
            "hidden_size": layer.hidden_units,
            "filter": layer.kernel_size,
            "activation": activation_to_dictionary(layer.activation),
            "hidden_units": layer.hidden_units,
            "input": "prev",
        }

    def to_weights(self) -> Dict[str, np.ndarray]:
        """
        Converts the layer to a dictionary that can be saved to a JSON file.
        """

        layer_weights = super().to_weights()

        # Prepare the layer
        layer = self._layer

        if layer.single_kernel:
            # We have a single kernel for all input slices
            bn_encoder = layer.se_light_blocks.encoder_decoder.hidden_layer_batch_norm
            bn_decoder = layer.se_light_blocks.encoder_decoder.decoder_layer_batch_norm
            assert isinstance(
                bn_encoder, tf.keras.layers.BatchNormalization
            ), "BN encoder is not a BatchNormalization layer"
            assert isinstance(
                bn_decoder, tf.keras.layers.BatchNormalization
            ), "BN decoder is not a BatchNormalization layer"

            # Process the batch normalization layers to get alpha and beta coefficients for each
            alpha_encoder, beta_encoder = (
                SaveableBatchNormalization.postprocess_batch_normalization(bn_encoder)
            )
            alpha_decoder, beta_decoder = (
                SaveableBatchNormalization.postprocess_batch_normalization(bn_decoder)
            )

            # Update the layer weights
            layer_weights.update(
                {
                    "batch_normalization_encoder_alpha": alpha_encoder.tolist(),
                    "batch_normalization_encoder_beta": beta_encoder.tolist(),
                    "batch_normalization_decoder_alpha": alpha_decoder.tolist(),
                    "batch_normalization_decoder_beta": beta_decoder.tolist(),
                }
            )

            return layer_weights

        # Now, we have to process the batch normalization layers
        for i in range(layer.grid_width):
            for j in range(layer.grid_height):
                # Get the batch normalization layers
                bn_encoder = layer.se_light_blocks[i][
                    j
                ].encoder_decoder.hidden_layer_batch_norm
                bn_decoder = layer.se_light_blocks[i][
                    j
                ].encoder_decoder.decoder_layer_batch_norm
                assert isinstance(
                    bn_encoder, tf.keras.layers.BatchNormalization
                ), "BN encoder is not a BatchNormalization layer"
                assert isinstance(
                    bn_decoder, tf.keras.layers.BatchNormalization
                ), "BN decoder is not a BatchNormalization layer"

                # Extract alpha and beta coefficients for each BN layer
                alpha_encoder, beta_encoder = (
                    SaveableBatchNormalization.postprocess_batch_normalization(
                        bn_encoder
                    )
                )
                alpha_decoder, beta_decoder = (
                    SaveableBatchNormalization.postprocess_batch_normalization(
                        bn_decoder
                    )
                )

                # Update the layer weights
                layer_weights.update(
                    {
                        f"batch_normalization_{i}_{j}_encoder_alpha": alpha_encoder.tolist(),
                        f"batch_normalization_{i}_{j}_encoder_beta": beta_encoder.tolist(),
                        f"batch_normalization_{i}_{j}_decoder_alpha": alpha_decoder.tolist(),
                        f"batch_normalization_{i}_{j}_decoder_beta": beta_decoder.tolist(),
                    }
                )

        return layer_weights
