"""
Class for interpeting the EDLightConv2D layer
"""

from __future__ import annotations

from typing import Dict, Any

import tensorflow as tf
import numpy as np

from tf_bionetta.save.layers.interface import SaveableLayer
from tf_bionetta.layers.conv.edlight import EDLight2DConv
from tf_bionetta.save.layers.keras.batch_normalization import (
    SaveableBatchNormalization,
)  # For postprocessing BN layers
from tf_bionetta.save.layers.activations.convert import activation_to_dictionary


class SaveableEDLightConv2D(SaveableLayer):
    """
    Class implementing the EDLightConv2D interpretation.
    """

    def __init__(self, layer: EDLight2DConv) -> None:
        """
        Initializes the EDLightConv2D layer.

        Args:
            - layer (`tf.keras.layers.Layer`): The layer to be interpreted.
        """

        assert isinstance(
            layer, EDLight2DConv
        ), "Only EDLightConv2D layers are supported"
        super().__init__(layer)

    def to_dictionary(self) -> Dict[str, Any]:
        """
        Converts the layer to a dictionary that can be saved to a JSON file.
        """

        layer = self._layer

        # NOTE: We always have the volume before and after the EDLight2DConv
        _, width_in, height_in, channels_in = layer.input_shape
        assert width_in == height_in, "Only square volumes are supported"

        return {
            "type": "EDLightConv2D",
            "name": layer.name,
            "length_in": width_in,  # Input size
            "channels_in": channels_in,  # Number of input channels
            "filter_in": layer.kernel_size,  # Kernel size
            "channels_out": layer.channels,  # Number of output channels
            "hidden_size": layer.hidden_units,  # Number of hidden units in the hidden layer of ED
            "filter_out": layer.kernel_output_size,
            "activation": activation_to_dictionary(layer.activation),
            "input": "prev",
        }

    def to_weights(self) -> Dict[str, np.ndarray]:
        """
        Converts the layer to a dictionary that can be saved to a JSON file.
        """

        # Prepare the layer
        layer = self._layer

        # First, we need to get weights from the parent class
        layer_weights = super().to_weights()

        # Next, we process batch normalization layers to include them in the weights separately
        if layer.single_kernel:
            # A single kernel consists of two batch normalization layers: one after the encoder and one after the decoder
            bn_encoder = layer.encoder_decoder.hidden_layer_batch_norm
            bn_decoder = layer.encoder_decoder.decoder_layer_batch_norm
            assert isinstance(
                bn_encoder, tf.keras.layers.BatchNormalization
            ), "Encoder BN must be a valid BatchNormalization layer"
            assert isinstance(
                bn_decoder, tf.keras.layers.BatchNormalization
            ), "Decoder BN must be a valid BatchNormalization layer"

            # Process the batch normalization layers to get alpha and beta coefficients
            alpha_encoder, beta_encoder = (
                SaveableBatchNormalization.postprocess_batch_normalization(bn_encoder)
            )
            alpha_decoder, beta_decoder = (
                SaveableBatchNormalization.postprocess_batch_normalization(bn_decoder)
            )

            # Update the weights dictionary
            layer_weights.update(
                {
                    "batch_normalization_encoder_alpha": alpha_encoder.tolist(),
                    "batch_normalization_encoder_beta": beta_encoder.tolist(),
                    "batch_normalization_decoder_alpha": alpha_decoder.tolist(),
                    "batch_normalization_decoder_beta": beta_decoder.tolist(),
                }
            )

            return layer_weights

        # If we have a grid of kernels, we need to process each of them separately
        for i in range(layer.grid_width):
            for j in range(layer.grid_height):
                # Extract the batch normalization layers
                bn_encoder = layer.encoder_decoders[i][j].hidden_layer_batch_norm
                bn_decoder = layer.encoder_decoders[i][j].decoder_layer_batch_norm
                assert isinstance(
                    bn_encoder, tf.keras.layers.BatchNormalization
                ), "Dima is stupid"
                assert isinstance(
                    bn_decoder, tf.keras.layers.BatchNormalization
                ), "Mark is stupid"

                # Get the alpha and beta coefficients
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

                # Update the dictionary
                layer_weights.update(
                    {
                        f"batch_normalization_{i}_{j}_encoder_alpha": alpha_encoder.tolist(),
                        f"batch_normalization_{i}_{j}_encoder_beta": beta_encoder.tolist(),
                        f"batch_normalization_{i}_{j}_decoder_alpha": alpha_decoder.tolist(),
                        f"batch_normalization_{i}_{j}_decoder_beta": beta_decoder.tolist(),
                    }
                )

        return layer_weights
