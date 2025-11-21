"""
Implementation of Heavy Squeeze-and-Excitation Layer
"""

from __future__ import annotations

from typing import Any, Callable, Tuple, Annotated

import tensorflow as tf
from keras import regularizers, initializers
from keras.layers import InputSpec

from tf_bionetta.layers.interface import BionettaLayer
from tf_bionetta.layers.se.light import SELightBlock
from tf_bionetta.constraints.types.layer_complexity import LayerComplexity


class SEHeavyBlock(BionettaLayer):
    """
    Implementation of the Heavy Squeeze-and-Excitation Layer.
    """

    DEFAULT_ACTIVATION: Annotated[
        tf.keras.layers.Layer,
        "Default activation function used after the encoding step in the ED Layer",
    ] = tf.keras.layers.ReLU()

    def __init__(
        self,
        kernel_size: int,
        hidden_units: int = 10,
        activation: tf.keras.layers.Layer = DEFAULT_ACTIVATION,
        kernel_initializer: str | None = "glorot_normal",
        kernel_regularizer: regularizers.Regularizer | None = tf.keras.regularizers.l2(1e-4),
        single_kernel: bool = False,
        **kwargs,
    ) -> None:
        """
        Initializes the Squeeze-and-Excitation Layer.

        Args:
            - kernel_size (`int`): Size of the kernel (filter) to be applied.
            - hidden_units (`int`, optional): Number of hidden units in the ED Layer. Defaults to 10.
            - activation (`str`, optional): The activation function in the hidden layer. Defaults to 'relu'.
            - kernel_initializer (`str`, optional): The initializer for the kernel weights. Defaults to 'glorot_normal'.
            - kernel_regularizer (`regularizers.Regularizer`, optional): The regularizer for the kernel weights. Defaults to L2 with 1e-4.
            - single_kernel (`bool`, optional): Whether to use a single kernel for all input slices. Defaults to False.
        """

        super(SEHeavyBlock, self).__init__(**kwargs)

        if type(activation) == dict:
            activation = tf.keras.layers.deserialize(activation)

        assert isinstance(
            activation, (Callable, tf.keras.layers.Layer)
        ), "Activation must be a Keras Layer. If you specified a string, please use `tf.keras.layers` instead."

        self.kernel_size = kernel_size
        self.hidden_units = hidden_units
        self.activation = activation
        self.single_kernel = single_kernel

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        self.input_spec = InputSpec(ndim=4) # We expect the 4D input of shape (batch_size, height, width, channels)


    def build(self, input_shape: Tuple, is_test_mode: bool = False) -> None:
        """
        Builds the layer. This method is called once the input shape is known.
        """

        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError(
                "The last dimension of the inputs to a Squeeze-and-Excitation layer "
                "should be defined. Found None. "
                f"Full input shape received: {input_shape}"
            )

        # We need to:
        # 1. Calculate the grid size
        # 2. For each grid cell, initialize the Light SE Block
        (_, height, width, _) = input_shape

        assert (
            width % self.kernel_size == 0
        ), f"Width {width} is not divisible by kernel size {self.kernel_size}"
        assert (
            height % self.kernel_size == 0
        ), f"Height {height} is not divisible by kernel size {self.kernel_size}"

        self.grid_width = width // self.kernel_size
        self.grid_height = height // self.kernel_size

        # Initialize the light blocks
        # If we use a single kernel, we only need one light block
        # Otherwise, we need a light block for each grid cell
        if self.single_kernel:
            self.se_light_block = SELightBlock(
                hidden_units=self.hidden_units,
                activation=self.activation,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
            )

            # Force the sublayers to build by calling them with a dummy input.
            dummy_input = tf.zeros(input_shape)
            _ = self.se_light_block(dummy_input)

        else:
            self.se_light_blocks = [
                [None] * self.grid_height for _ in range(self.grid_width)
            ]
            for i in range(self.grid_width):
                for j in range(self.grid_height):
                    self.se_light_blocks[i][j] = SELightBlock(
                        hidden_units=self.hidden_units,
                        activation=self.activation,
                        kernel_initializer=self.kernel_initializer,
                        kernel_regularizer=self.kernel_regularizer,
                    )

            # Force the sublayers to build by calling them with a dummy input.
            if is_test_mode:
                dummy_input = tf.zeros(input_shape)
                for i in range(self.grid_width):
                    for j in range(self.grid_height):
                        _ = self.se_light_blocks[i][j](dummy_input)

        # Computes the complexity of the layer (for further constraints estimation)
        self.complexity = self.compute_complexity(input_shape)


    def compute_complexity(
        self, 
        input_shape: tf.TensorShape
    ) -> LayerComplexity:
        """
        Calculates the complexity of the layer based on the input shape.
        
        Args:
            - input_shape (`tf.TensorShape`): The shape of the input tensor.
        
        Output:
            - A `LayerComplexity` object containing the complexity of the layer.
        """

        input_shape = tf.TensorShape(input_shape)
        (_, _, _, channels) = input_shape

        grid_size = self.grid_width*self.grid_height
        return LayerComplexity(
            mul_ops=(self.kernel_size**2)*channels*grid_size,
            non_linear_ops=[
                (self.activation, grid_size*self.hidden_units)
            ],
        )
        

    def call(self, inputs: tf.Tensor):
        """
        Calls the layer. This method is called when the layer is used in a model.
        """

        assert len(inputs.shape) == 4, f"Expected 4D input, got {len(inputs.shape)}"

        # Prepare the output
        grid = []
        for i in range(self.grid_width):
            row = []
            for j in range(self.grid_height):
                patch = inputs[
                    :,
                    i * self.kernel_size : (i + 1) * self.kernel_size,
                    j * self.kernel_size : (j + 1) * self.kernel_size,
                    :,
                ]

                # Apply the light block
                if self.single_kernel:
                    se_output = self.se_light_block(patch)
                else:
                    se_output = self.se_light_blocks[i][j](patch)

                row.append(se_output)

            grid.append(tf.concat(row, axis=2))  # Concatenate the row along the width

        output = tf.concat(grid, axis=1)
        return output


    def compute_output_shape(self, input_shape: Tuple):
        """
        Computes the output shape of the layer.
        """

        input_shape = tf.TensorShape(input_shape)
        return input_shape  # The shape does not change


    def get_config(self):
        """
        Returns the configuration of the layer for serialization.
        """

        config = super(SEHeavyBlock, self).get_config()
        config.update(
            {
                "kernel_size": self.kernel_size,
                "hidden_units": self.hidden_units,
                "activation": self.activation,
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
                "single_kernel": self.single_kernel,
            }
        )
        return config


    @classmethod
    def from_config(cls, config):
        """
        Creates the layer from its configuration.
        """

        config["kernel_initializer"] = initializers.deserialize(
            config["kernel_initializer"]
        )
        config["kernel_regularizer"] = regularizers.deserialize(
            config["kernel_regularizer"]
        )
        return cls(**config)
