"""
File responsible for saving the model after the
training has been completed (or during the training
while saving the concrete epoch).
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np
from keras import Model

from tf_bionetta.layers.custom_objects import get_custom_objects
from tf_bionetta.layers.conv.edlight import EDLight2DConv
from tf_bionetta.layers.se.light import SELightBlock
from tf_bionetta.layers.se.heavy import SEHeavyBlock
from tf_bionetta.layers.normalization.l2 import L2UnitNormalizationLayer
from tf_bionetta.layers.hard_sigmoid import HardSigmoid
from tf_bionetta.optimizations.batch_folding import fold_batch_norm, identity_batch_norm
import tensorflow as tf


class BionettaModelOptimizer:
    """
    Class responsible for model optimizations such as BatchNormalization folding,
    folding the set of sequential linear layers into a single linear layer, etc.
    """

    SUPPORTED_BN_FOLDING_LAYERS = (tf.keras.layers.Conv2D, tf.keras.layers.Dense)
    DEFAULT_PRECISION = 20  # Default precision for the circuit parameters

    def __init__(self, model: tf.keras.models.Model) -> None:
        """
        Initializes the model saver.

        Arguments:
            - model (tf.keras.models.Model) - model to save
        """

        self.model = model

    def _bn_fold(
        self, model: tf.keras.models.Model, cloned_model: tf.keras.models.Model
    ) -> tf.keras.models.Model:
        """
        Folds BatchNormalization layers with the previous Conv2D,
        DepthwiseConv2D, and Dense layers.

        Arguments:
            - model (tf.keras.models.Model) - model to fold BatchNorm layers
        """

        assert len(model.layers) > 0, "The model must have at least one layer"

        new_layers: List[tf.keras.layers.Layer] = [
            model.layers[0]
        ]  # A list of new layers
        for i in range(1, len(model.layers)):
            if isinstance(model.layers[i], tf.keras.layers.BatchNormalization):
                folded_layer = fold_batch_norm(
                    cloned_model.layers[i - 1], model.layers[i - 1], model.layers[i]
                )
                folding_applied = folded_layer is not None
                if folding_applied:
                    # Remove the last layer and add the folded one
                    cloned_model.layers[i - 1] = folded_layer
                    cloned_model.layers[i] = identity_batch_norm(
                        input_shape=model.layers[i].input_shape
                    )
            else:
                new_layers.append(cloned_model.layers[i])

        return cloned_model

    def fold_batch_norms(self) -> Model:
        """
        Folds BatchNorm layers with the previous Conv2D layers.
        """

        # Copy the model
        tf.keras.utils.get_custom_objects().update(get_custom_objects())
        cloned_model = tf.keras.models.clone_model(self.model)
        cloned_model.set_weights(self.model.get_weights())

        return self._bn_fold(self.model, cloned_model)

    def _postprocess_bn(
        self, bn_layer: tf.keras.layers.BatchNormalization
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

    def save_circuit_params(
        self, path: Path, precision: int = DEFAULT_PRECISION
    ) -> None:
        """
        Forms the circuit parameters needed for further Circom code generation
        and Rust witness generation.

        Arguments:
            - path (Path) - path to save the parameters
        """

        if not isinstance(path, Path):
            path = Path(path)  # Convert to Path object

        circuit_params = {
            "inputs": [{"name": "image", "size": self.model.input_shape[1:]}],
            "consts": {"precision": precision},
            "layers": [],
        }

        for layer in self.model.layers:
            # Further strategy:
            # Check for type and manually add the layer to the json file,
            # depending on the type of the layer.

            if isinstance(layer, tf.keras.layers.InputLayer):
                # NOTE: Input Layer plays no role in the circuit
                continue

            if isinstance(layer, EDLight2DConv):
                # NOTE: We always have the volume before and after the EDLight2DConv
                _, width_in, height_in, channels_in = layer.input_shape
                assert width_in == height_in, "Only square volumes are supported"

                edconv_layer_params = {
                    "type": "EDLightConv2D",
                    "name": layer.name,
                    "length_in": width_in,  # Input size
                    "channels_in": channels_in,  # Number of input channels
                    "filter_in": layer.kernel_size,  # Kernel size
                    "channels_out": layer.channels,  # Number of output channels
                    "hidden_units": layer.hidden_units,  # Number of hidden units in the hidden layer of ED
                    "filter_out": layer.kernel_output_size,
                    "input": "prev",
                }

                circuit_params["layers"].append(edconv_layer_params)
                continue

            if isinstance(layer, tf.keras.layers.AveragePooling2D):
                avg_pooling_layer_params = {
                    "type": "AveragePooling2D",
                    "name": layer.name,
                    "func": "Avg",
                    "input": "prev",
                    "out_shape": layer.output_shape[1:],
                }

                circuit_params["layers"].append(avg_pooling_layer_params)
                continue

            if isinstance(layer, tf.keras.layers.Conv2D):
                # NOTE: We always have the volume before and after the Conv2D
                # Getting input shape parameters
                _, width_in, height_in, channels_in = layer.input_shape
                assert width_in == height_in, "Only square volumes are supported"

                # Getting output shape parameters
                _, width_out, height_out, channels_out = layer.output_shape
                assert width_out == height_out, "Only square volumes are supported"

                # Getting the kernel size
                kernel_size = layer.kernel_size[0]

                conv_layer_params = {
                    "type": "Conv2D",
                    "name": layer.name,
                    "length_in": width_in,
                    "channels_in": channels_in,
                    "length_out": width_out,
                    "channels_out": channels_out,
                    "filter": kernel_size,
                    "input": "prev",
                }

                circuit_params["layers"].append(conv_layer_params)
                continue

            if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
                # NOTE: We always have the volume before and after the DepthwiseConv2D
                _, width_in, height_in, channels_in = layer.input_shape
                assert width_in == height_in, "Only square volumes are supported"

                _, width_out, height_out, channels_out = layer.output_shape
                assert width_out == height_out, "Only square volumes are supported"

                kernel_size = layer.kernel_size[0]

                depthwise_conv_params = {
                    "type": "DepthwiseConv2D",
                    "name": layer.name,
                    "length_in": width_in,
                    "channels_in": channels_in,
                    "length_out": width_out,
                    "channels_out": channels_out,
                    "filter": kernel_size,
                    "input": "prev",
                }

                circuit_params["layers"].append(depthwise_conv_params)
                continue

            if isinstance(layer, SELightBlock):
                # NOTE: We always have the volume before the SELightBlock
                _, width_in, height_in, channels_in = layer.input_shape

                # TODO: How to specify the activation function? Also precision parameters
                se_layer_params = {
                    "type": "SELightBlock",
                    "name": layer.name,
                    "width": width_in,
                    "height": height_in,
                    "channels": channels_in,
                    "hidden_size": layer.hidden_units,
                    "input": "prev",
                }

                circuit_params["layers"].append(se_layer_params)
                continue

            if isinstance(layer, SEHeavyBlock):
                # NOTE: We always have the volume before the SEHeavyBlock
                _, width_in, height_in, channels_in = layer.input_shape
                assert width_in == height_in, "Only square volumes are supported"

                se_layer_params = {
                    "type": "SEHeavyBlock",
                    "name": layer.name,
                    "length": width_in,
                    "channels": channels_in,
                    "hidden_units": layer.hidden_units,
                    "filter": layer.kernel_size,
                    "input": "prev",
                }

                circuit_params["layers"].append(se_layer_params)
                continue

            if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
                # NOTE: We always have the volume before the GlobalAveragePooling2D
                _, width_in, height_in, channels_in = layer.input_shape
                assert width_in == height_in, "Only square volumes are supported"

                global_avg_pool_params = {
                    "type": "GlobalAveragePooling2D",
                    "name": layer.name,
                    "length": width_in,
                    "channels": channels_in,
                    "input": "prev",
                }

                circuit_params["layers"].append(global_avg_pool_params)
                continue

            if isinstance(layer, tf.keras.layers.BatchNormalization):
                # NOTE: Check the previous layer for the shape. If it is a volume,
                # we have to specify the volume shape. If it is a dense layer, we have
                # to specify the number of neurons.
                bn_params = {
                    "type": "BatchNormalization",
                    "name": layer.name,
                    "input_shape": layer.input_shape[1:],
                    "input": "prev",
                }

                circuit_params["layers"].append(bn_params)
                continue

            if isinstance(layer, tf.keras.layers.Dense):
                # NOTE: We always have a flat input before the Dense layer
                input_neurons = layer.input_shape[1:]
                output_neurons = layer.units

                dense_params = {
                    "type": "Dense",
                    "name": layer.name,
                    "input_neurons": input_neurons,
                    "output_neurons": output_neurons,
                    "input": "prev",
                }

                circuit_params["layers"].append(dense_params)

                if layer.activation is not None:
                    print(
                        f"WARNING: Activation function {layer.activation} inside the Dense Layer is not supported, write it as a separate layer"
                    )

                continue

            if isinstance(layer, tf.keras.layers.Add):
                # NOTE: We always have the same shape for the Add layer
                # Print all attributes of layer
                assert (
                    len(layer.input_shape) >= 2
                ), "Add layer must have at least two inputs"
                add_params = {
                    "type": "Add",
                    "name": layer.name,
                    "input_shape": layer.input_shape[0][1:],
                    "input": [input_layer.name for input_layer in layer.input],
                }

                circuit_params["layers"].append(add_params)
                continue

            # If the layer is an activation layer, we have to specify the activation function
            if isinstance(layer, tf.keras.layers.ReLU):
                activation_params = {
                    "type": "ReLU",
                    "name": layer.name,
                    "input_shape": layer.input_shape[1:],
                    "input": "prev",
                }

                circuit_params["layers"].append(activation_params)
                continue

            if isinstance(layer, tf.keras.layers.LeakyReLU):
                activation_params = {
                    "type": "LeakyReLU",
                    "name": layer.name,
                    "power": round(np.log2(layer.alpha)),
                    "input_shape": layer.input_shape[1:],
                    "input": "prev",
                }

                circuit_params["layers"].append(activation_params)
                continue

            if isinstance(layer, L2UnitNormalizationLayer):
                # NOTE: We always have the same shape (,input_neurons) for the L2UnitNormalizationLayer
                l2norm_params = {
                    "type": "L2UnitNormalization",
                    "name": layer.name,
                    "input_shape": layer.input_shape[1],
                    "input": "prev",
                }

                circuit_params["layers"].append(l2norm_params)
                continue

            if isinstance(layer, HardSigmoid):
                # NOTE: We always have the same shape for the HardSigmoid layer
                hard_sigmoid_params = {
                    "type": "HardSigmoid",
                    "name": layer.name,
                    "input_shape": layer.input_shape[1:],
                    "input": "prev",
                }

                circuit_params["layers"].append(hard_sigmoid_params)
                continue

            print(f"WARNING: Layer {type(layer)} is not supported")

        # Save the formed circuit parameters
        with open(path, "w") as f:
            json.dump(circuit_params, f, indent=4)

    def save_weights(self, path: Path) -> None:
        """
        Saves the model's weights

        Arguments:
            - model - model to save
        """

        weights_dict: Dict[str, tf.Tensor] = {}

        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                alpha, beta = self._postprocess_bn(layer)
                weights_dict[layer.name] = {
                    "alpha": alpha.tolist(),
                    "beta": beta.tolist(),
                }
                continue

            layer_weights: Dict[str, tf.Tensor] = {}

            for weight in layer.weights:
                layer_weights[weight.name] = weight.numpy().tolist()
            if layer_weights:
                weights_dict[layer.name] = layer_weights

        self.model.save(path.parent / "model.h5")
        with open(path, "w") as f:
            json.dump(weights_dict, f, indent=4)
