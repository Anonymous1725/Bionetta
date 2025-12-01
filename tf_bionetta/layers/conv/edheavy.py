"""
Package for implementing the Encoder-Decoder Heavy Convolution Layer
"""

from __future__ import annotations

from typing import Any, Callable, Tuple, List, Dict

import tensorflow as tf
from keras import regularizers, initializers
from keras.layers import InputSpec

from tf_bionetta.layers.interface import BionettaLayer, LayerComplexity
from tf_bionetta.layers.ed import EncoderDecoderLayer


class EDHeavy2DConv(BionettaLayer):
    """
    Encoder-Decoder Heavy Convolution Layer implementation. This layer is zk-cost-effective
    alternative to the regular convolutional layer. The main advantage of this layer is that
    it requires significantly less activation functions than the regular convolutional layer.
    Yet, the number of parameters is significantly reduced.

    It works by going filter by filter and applying the Encoder-Decoder layer to each filter.
    In contrast to the light version, this version works similarly to the regular convolutional
    layer: it uses a single filter (encoder+decoder parameters) for the single output channel.
    """

    DEFAULT_ACTIVATION = tf.keras.layers.ReLU()


    def __init__(
        self,
        kernel_size: int,
        channels: int,
        hidden_units: int = 10,
        kernel_output_size: int = 10,
        activation: Callable[[Any], Any] | tf.keras.layers.Layer = DEFAULT_ACTIVATION,
        kernel_initializer: str | None = "glorot_normal",
        kernel_regularizer: regularizers.Regularizer | None = tf.keras.regularizers.l2(1e-4),
        **kwargs,
    ) -> None:
        """
        2D Heavy Convolutional Layer with Encoder-Decoder Layer.

        Args:
            - kernel_size (`int`): Size of the kernel (filter) to be applied.
            - channels (int): Number of output channels (filters).
            - hidden_units (int, optional): Number of hidden units in the ED Layer. Defaults to 10.
            - kernel_output_size (int, optional): Output size of the kernel. Defaults to 10.
        """

        super(EDHeavy2DConv, self).__init__(**kwargs)

        if type(activation) == dict:
            activation = tf.keras.layers.deserialize(activation)

        assert isinstance(
            activation, (Callable, tf.keras.layers.Layer)
        ), "Activation must be a Keras Layer. If you specified a string, please use `tf.keras.layers` instead."

        self.kernel_size = kernel_size
        self.channels = channels
        self.hidden_units = hidden_units
        self.kernel_output_size = kernel_output_size
        self.activation = activation

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        self.input_spec = InputSpec(
            ndim=4
        )  # We expect the 4D input of shape (batch_size, height, width, channels)


    def build(self, input_shape: Tuple, is_test_mode: bool = False) -> None:
        """
        Builds the layer. This method is called once the input shape is known.
        """

        input_shape = tf.TensorShape(input_shape)
        (_, height, width, _) = input_shape  # Extract the input shape

        self.width = width
        self.height = height

        assert (
            width % self.kernel_size == 0
        ), f"Width {width} is not divisible by kernel size {self.kernel_size}"
        assert (
            height % self.kernel_size == 0
        ), f"Height {height} is not divisible by kernel size {self.kernel_size}"

        # Some helper size parameters
        self.grid_width = width // self.kernel_size
        self.grid_height = height // self.kernel_size
        self.ed_output_kernel_area = (
            self.kernel_output_size * self.kernel_output_size
        )  # Size of the output window volume

        # Initialize all the encoder-decoder layers
        self.encoder_decoders: List[EncoderDecoderLayer] = [None] * self.channels
        for c in range(self.channels):
            self.encoder_decoders[c] = EncoderDecoderLayer(
                self.ed_output_kernel_area,
                hidden_units=self.hidden_units,
                activation=self.activation,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"encoder_decoder_kernel_{c}",
            )
        if is_test_mode:
            dummy_input = tf.zeros(
                (1, self.kernel_size * self.kernel_size * self.channels)
            )
            for encoder_decoder in self.encoder_decoders:
                encoder_decoder(dummy_input)

        # Calculate the model's complexity for further 
        # constraints estimation
        self.complexity = self.compute_complexity(input_shape)


    def compute_complexity(
        self, 
        input_shape: tf.TensorShape
    ) -> LayerComplexity:
        """
        Calculate the complexity of the layer based on the input shape.
        
        Args:
            - input_shape (`tf.TensorShape`): The shape of the input tensor.
            
        Output:
            - A `LayerComplexity` object containing the complexity of the layer.
        """

        input_shape = tf.TensorShape(input_shape)
        (_, height, width, _) = input_shape
        grid_width = width // self.kernel_size
        grid_height = height // self.kernel_size

        return LayerComplexity(
            mul_ops=0,
            non_linear_ops=[
                (self.activation, grid_width * grid_height * self.channels * self.hidden_units)
            ],
        )

        
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Computes the output of the layer.
        """

        assert len(inputs.shape) == 4, f"Expected 4D input, got {len(inputs.shape)}"

        batch_size = tf.shape(inputs)[0]
        input_channels = tf.shape(inputs)[3]

        # Prepare the output
        stack = []

        for c in range(self.channels):
            grid = []
            for i in range(self.grid_width):
                row = []
                for j in range(self.grid_height):
                    # Extract the token
                    from_x: int = i * self.kernel_size
                    to_x: int = (i + 1) * self.kernel_size
                    from_y: int = j * self.kernel_size
                    to_y: int = (j + 1) * self.kernel_size
                    token = inputs[:, from_x:to_x, from_y:to_y, :]

                    # Flatten the token along all channels, except for the batch channel.
                    # Expected shape: (batch, width * height * channels)
                    # Reshape the token to the flattened shape
                    token = tf.reshape(
                        token,
                        [
                            batch_size,
                            self.kernel_size * self.kernel_size * input_channels,
                        ],
                    )

                    # Apply the encoder-decoder layer on the flattened token
                    token = self.encoder_decoders[c](token)

                    # Reshape the token to the out shape
                    token = tf.reshape(
                        token,
                        [batch_size, self.kernel_output_size, self.kernel_output_size],
                    )

                    # Insert the token into the output
                    row.append(token)

                grid.append(
                    tf.concat(row, axis=2)
                )  # Concatenate the row along the width

            channel_output = tf.concat(grid, axis=1)
            stack.append(channel_output)

        output = tf.stack(stack, axis=-1)
        return output


    def compute_output_shape(self, input_shape: Tuple) -> Tuple:
        """
        Computes the output shape of the layer.
        """

        input_shape = tf.TensorShape(input_shape)

        (batch_size, height, width, _) = input_shape  # Extract the input shape
        grid_width = width // self.kernel_size
        grid_height = height // self.kernel_size

        output_width = grid_width * self.kernel_output_size
        output_height = grid_height * self.kernel_output_size
        output_channels = self.channels

        return tf.TensorShape(
            [batch_size, output_height, output_width, output_channels]
        )


    def get_config(self) -> Dict:
        """
        Gets the configuration of the layer for serialization.
        """

        config = super().get_config()
        config.update(
            {
                "kernel_size": self.kernel_size,
                "channels": self.channels,
                "hidden_units": self.hidden_units,
                "kernel_output_size": self.kernel_output_size,
                "activation": self.activation,
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            }
        )

        return config


    @classmethod
    def from_config(cls, config: dict) -> EDHeavy2DConv:
        """
        Initializes the layer from the configuration.
        """

        config["kernel_initializer"] = initializers.deserialize(
            config["kernel_initializer"]
        )
        config["kernel_regularizer"] = regularizers.deserialize(
            config["kernel_regularizer"]
        )

        return cls(**config)
