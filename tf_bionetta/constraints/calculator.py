"""
Some constants used for calculating the number of
constraints for the layer.
"""

from __future__ import annotations

import logging
import sys

from typing_extensions import TypeAlias
from typing import Dict, Tuple

# For drawing tables
from rich.console import Console
from rich.table import Table

import tensorflow as tf
import numpy as np

from tf_bionetta.utils import unpack_model_layers
from tf_bionetta.constraints.types.layer_complexity import LayerComplexity
from tf_bionetta.constraints.types.activations import ActivationOps
from tf_bionetta.constraints.types.severity import (
    severity_from_layer_constraints,
    severity_from_model_constraints,
)
from tf_bionetta.layers import HardSigmoid
from tf_bionetta.layers import HardSwish
from tf_bionetta.layers.interface import BionettaLayer
from tf_bionetta.logging.logger import MaybeLogger
from tf_bionetta.specs.backend_enums import(
    ProvingBackend,
    Groth16,
    UltraGroth
)

# Type alias for the layer cost. It consists of the number of constraints
# and a boolean flag indicating if the cost is an estimation.
LayerCost: TypeAlias = Tuple[int, bool]


class ModelConstraintsCalculator:
    """
    Class that calculates the total number of constraints for a Keras model.
    """

    def __init__(
        self,
        model: tf.keras.models.Model,
        backend: ProvingBackend,
        name: str = None,
        logger: logging.Logger | None = None
    ) -> None:
        """
        Initialize the calculator for the selected Keras model.
        
        NOTE: Depending on the backend used (Groth16 or UltraGroth), the 
        number of constraints differs significantly.

        Args:
            model (`tf.keras.models.Model`): A Keras model.
            name (`str`, optional): The name of the model. Defaults to `None`.
            backend (`ProvingBackend`, optional): The proving backend to use. Defaults to `ProvingBackend.GROTH16`.
            logger (`logging.Logger | None`): A logger object.
        """

        assert isinstance(model, tf.keras.models.Model), "Input must be a Keras Model."

        self.model = model
        self.name = name if name is not None else model.name
        self._backend = backend
        self._logger = MaybeLogger(logger)
        self._unpack_layers = unpack_model_layers(self.model)
        self.layer_constraints = self._compute_layer_constraints()


    def _compute_layer_constraints(self) -> Dict[tf.keras.layers.Layer, LayerCost]:
        """
        Calculate the total sum of constraints across all layers in the Keras model.

        Returns:
            Dict[tf.keras.layers.Layer, Tuple[int, bool]]: A dictionary with layer constraints and a boolean flag indicating if the cost is an estimation.
        """

        # The result dictionary with layer constraints. As keys, we
        # have the layers and as values, we have (1) the number of
        # constraints and (2) a boolean flag indicating if the cost
        # is an estimation.
        result: Dict[tf.keras.layers.Layer, LayerCost] = {}
        for layer in self._unpack_layers:
            # If we have the custom layer (aka BionettaLayer),
            # we can get the constraints directly
            if isinstance(layer, BionettaLayer):
                complexity: LayerComplexity = layer.complexity
                constraints, exception = complexity.compute_constraints(self._backend)
                if exception is not None:
                    self._logger.error(
                        f"Exception '{exception}' occurred while estimating the cost of layer {layer.name}. Using estimation."
                    )
                    
                result[layer] = (constraints, complexity.constraints_estimated)
                continue

            # The special case is the fully connected layer, we check
            # this guy separately
            if self.is_fc_with_activation(layer):
                complexity = LayerComplexity(
                    mul_ops=0,
                    non_linear_ops=[
                        (layer.activation, layer.units)
                    ]
                )
                constraints, exception = complexity.compute_constraints(self._backend)
                if exception is not None:
                    self._logger.error(
                        f"Exception '{exception}' occurred while estimating the cost of layer {layer.name}. Using estimation."
                    )
                
                result[layer] = (constraints, complexity.constraints_estimated)
                continue

            # Check if the layer is the activation layer
            if self.is_activation_layer(layer):
                
                input_neurons = self.get_input_neurons_number(layer)
                complexity = LayerComplexity(
                    mul_ops=0,
                    non_linear_ops=[
                        (layer, input_neurons)
                    ]
                )
                constraints, exception = complexity.compute_constraints(self._backend)
                if exception is not None:
                    self._logger.error(
                        f"Exception '{exception}' occurred while estimating the cost of layer {layer.name}. Using estimation."
                    )

                result[layer] = (constraints, complexity.constraints_estimated)
                continue

            # Finally, check if the layer is free of constraints
            result[layer] = (0, self.is_free_layer(layer))

        return result


    @staticmethod
    def is_activation_layer(layer: tf.keras.layers.Layer) -> bool:
        return isinstance(
            layer,
            (
                tf.keras.layers.Activation,  # Standard activations
                tf.keras.layers.ReLU,
                tf.keras.layers.LeakyReLU,  # Advanced activations
                HardSigmoid,
                HardSwish,
                tf.keras.layers.ReLU,
                tf.keras.layers.ELU,
                tf.keras.layers.PReLU,
                tf.keras.layers.ThresholdedReLU,
                tf.keras.layers.Softmax,
            ),
        )


    @staticmethod
    def is_free_layer(layer: tf.keras.layers.Layer) -> bool:
        """
        Based on the layer type, determine if the layer is free of constraints.
        """

        if isinstance(
            layer,
            (
                tf.keras.layers.InputLayer,
                tf.keras.layers.Dropout,
                tf.keras.layers.BatchNormalization,
                tf.keras.layers.AveragePooling2D,
                tf.keras.layers.GlobalAveragePooling2D,
                tf.keras.layers.ZeroPadding2D,
                tf.keras.layers.Add,
                tf.keras.layers.Flatten,
            ),
        ):
            return False

        if isinstance(
            layer,
            (
                tf.keras.layers.Dense,
                tf.keras.layers.Conv2D,
                tf.keras.layers.DepthwiseConv2D,
                tf.keras.layers.SeparableConv2D,
            ),
        ):
            # Check if the activation is present
            if layer.activation is None:
                return False

            activation = layer.activation
            # If the activation is linear, the layer is free
            if activation == tf.keras.activations.linear:
                return False

            # If the activation is not linear, the layer is not free
            return True

        return True


    # Function to calculate number of input neurons/features for any activation layer
    def get_input_neurons_number(self, activation_layer: tf.keras.layers.Layer) -> int:
        # Find the index of the activation layer
        layer_index = self._unpack_layers.index(activation_layer)

        # Get the output shape of the previous layer (input to the activation)
        previous_layer = self._unpack_layers[layer_index - 1]
        # if tf.__version__.startswith('2.18'):

        if isinstance(previous_layer.input, list):  # For Add layer
            input_shape = previous_layer.input[0].shape.as_list()
            for i in range(1, len(previous_layer.input)):
                input_shape = max(input_shape, previous_layer.input[i].shape.as_list())
        else:
            input_shape = previous_layer.compute_output_shape(previous_layer.input.shape)

        # Exclude batch size (which is None) and calculate the total number of features
        neurons_number = np.prod(
            input_shape[1:]
        )  # Multiply all dimensions except batch size
        return neurons_number


    def total_constraints(self) -> int:
        """
        Calculate the total sum of constraints across all layers in the Keras model.

        :return: Total constraints as an integer.
        """
        return sum([cost for cost, _ in self.layer_constraints.values()])


    @staticmethod
    def is_fc_with_activation(layer):
        """
        Determine if a layer is a fully connected (Dense) layer with an activation function.

        :param layer: A layer object.
        :return: True if the layer is a Dense layer with an activation function, False otherwise.
        """

        return isinstance(layer, tf.keras.layers.Dense) and layer.activation is not None

    
    @staticmethod
    def _to_label(cost) -> str:
        layer_severity = severity_from_layer_constraints(cost)
        return f"[{layer_severity.rich_color()}]{cost}"


    def print_constraints_summary(self) -> None:
        """
        Print a summary of each layer with its constraints, similar to model.summary().
        """

        table = Table(title=f"{self.model.name} Constraints Summary")
        table.add_column("Layer (type)", justify="left")
        table.add_column("Constraints", justify="center", style="green")

        name, constraints = self._backend.initial_constraints()
        self.layer_constraints[name] = (constraints, False)  # False, because constraints not estimated
        table.add_row(name, str(ModelConstraintsCalculator._to_label(constraints)))

            
        for layer in self._unpack_layers:
            layer_name = f"{layer.name} ({layer.__class__.__name__})"

            cost, estimated = self.layer_constraints[layer]
            cost_label = self._to_label(cost)
            if estimated:
                cost_label += " [bold red](ESTIMATED)"

            table.add_row(layer_name, cost_label)

        # Add the total constraints
        total_constraints = self.total_constraints()
        model_severity = severity_from_model_constraints(total_constraints)
        table.add_section()
        table.add_row(
            "Total", f"[{model_severity.rich_color()}]{str(total_constraints)}"
        )

        # Print the table
        console = Console()
        console.print(table)
