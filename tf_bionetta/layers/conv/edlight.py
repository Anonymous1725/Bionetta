"""
Package for implementing the Encoder-Decoder Light Convolution Layer
"""

from typing import Any, Callable, Tuple, List

import tensorflow as tf
from keras import regularizers, initializers
from keras.layers import InputSpec

from tf_bionetta.layers.interface import BionettaLayer, LayerComplexity
from tf_bionetta.layers.ed import EncoderDecoderLayer


class EDLight2DConv(BionettaLayer):
    """
    Encoder-Decoder Light Convolution Layer implementation. This layer is zk-cost-effective
    alternative to the regular convolutional layer. The main advantage of this layer is that
    it requires significantly less activation functions than the regular convolutional layer.
    Yet, the number of parameters is significantly reduced.

    It works by going filter by filter and applying the Encoder-Decoder layer to each filter.
    """

    DEFAULT_ACTIVATION = tf.keras.layers.ReLU()


    def __init__(
        self,
        kernel_size: int,
        channels: int,
        hidden_units: int = 10,
        kernel_output_size: int = 10,
        activation: tf.keras.layers.Layer = DEFAULT_ACTIVATION,
        kernel_initializer: str | None = "glorot_normal",
        kernel_regularizer: regularizers.Regularizer | None = tf.keras.regularizers.l2(1e-4),
        use_residual: bool = False,
        single_kernel: bool = False,
        **kwargs,
    ) -> None:
        """
        2D Convolutional Layer with Encoder-Decoder Layer.

        Args:
            - kernel_size (`int`): Size of the kernel (filter) to be applied.
            - channels (int): Number of output channels (filters).
            - hidden_units (int, optional): Number of hidden units in the ED Layer. Defaults to 10.
            - kernel_output_size (int, optional): Output size of the kernel. Defaults to 10.
            - activation (`str`, optional): The activation function in the hidden layer. Defaults to 'relu'.
            - kernel_initializer (`str`, optional): The initializer for the kernel weights. Defaults to 'glorot_normal'.
            - kernel_regularizer (`regularizers.Regularizer`, optional): The regularizer for the kernel weights. Defaults to L2 with 1e-4.
            - use_residual (`bool`, optional): Whether to use residual connections. Defaults to False.
            - single_kernel (`bool`, optional): Whether to use a single kernel for all input slices. Defaults to False.
        """

        super(EDLight2DConv, self).__init__(**kwargs)

        if type(activation) == dict:
            activation = tf.keras.layers.deserialize(activation)

        assert isinstance(
            activation, (Callable, tf.keras.layers.Layer)
        ), "Activation must be a Keras Layer. If you specified a string, please use `tf.keras.layers` instead."

        if use_residual:
            assert (
                kernel_output_size == kernel_size
            ), "Residual connections require the kernel output size to be the same as the kernel size."

        self.kernel_size = kernel_size
        self.channels = channels
        self.hidden_units = hidden_units
        self.kernel_output_size = kernel_output_size
        self.activation = activation
        self.use_residual = use_residual
        self.single_kernel = single_kernel
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        self.input_spec = InputSpec(
            ndim=4
        ) # We expect the 4D input of shape (batch_size, height, width, channels)


    def build(
        self, 
        input_shape: Tuple, 
        test_mode: bool = False
    ) -> None:
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
        self.ed_output_volume_size = (
            self.kernel_output_size * self.kernel_output_size * self.channels
        ) # Size of the output window volume

        # Initialize all the encoder-decoder layers. If
        # we are using a single filter, we only need one layer.
        # Otherwise, we need a layer for each grid cell.
        if self.single_kernel:
            self.encoder_decoder = EncoderDecoderLayer(
                self.ed_output_volume_size,
                hidden_units=self.hidden_units,
                activation=self.activation,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="encoder_decoder_kernel",
            )

            # Force the sublayers to build by calling them with a dummy input.
            if test_mode:
                dummy_input = tf.zeros(
                    (1, self.kernel_size * self.kernel_size * self.channels)
                )
                _ = self.encoder_decoder(dummy_input)
        else:
            self.encoder_decoders: List[List[EncoderDecoderLayer]] = [
                [None] * self.grid_height for _ in range(self.grid_width)
            ]
            
            for i in range(self.grid_width):
                for j in range(self.grid_height):
                    self.encoder_decoders[i][j] = EncoderDecoderLayer(
                        self.ed_output_volume_size,
                        hidden_units=self.hidden_units,
                        activation=self.activation,
                        kernel_initializer=self.kernel_initializer,
                        kernel_regularizer=self.kernel_regularizer,
                        name=f"encoder_decoder_kernel_{i}_{j}",
                    )

            # Force the sublayers to build by calling them with a dummy input.
            if test_mode:
                dummy_input = tf.zeros(
                    (1, self.kernel_size * self.kernel_size * self.channels)
                )
                for i in range(self.grid_width):
                    for j in range(self.grid_height):
                        _ = self.encoder_decoders[i][j](dummy_input)

        # Calculate the complexity of the layer (for further constraints estimation)
        self.complexity = self.compute_complexity(input_shape)


    def compute_complexity(
        self, 
        input_shape: tf.TensorShape
    ) -> LayerComplexity:
        """
        Calculates the layer complexity of the layer based 
        on the input shape.
        
        Args:
            input_shape (tf.TensorShape): The shape of the input tensor.
            
        Returns:
            LayerComplexity: An object containing the number of multiplications
            and non-linear operations in the layer.
        """

        input_shape = tf.TensorShape(input_shape)
        (_, height, width, _) = input_shape
        grid_width = width // self.kernel_size
        grid_height = height // self.kernel_size

        return LayerComplexity(
            mul_ops=0,
            non_linear_ops=[
                (self.activation, grid_width * grid_height * self.hidden_units)
            ]
        )


    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Computes the output of the layer.
        """

        assert len(inputs.shape) == 4, f"Expected 4D input, got {len(inputs.shape)}"

        batch_size = tf.shape(inputs)[0]
        input_channels = tf.shape(inputs)[3]

        # Prepare the output
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
                    [batch_size, self.kernel_size * self.kernel_size * input_channels],
                )

                # Apply the encoder-decoder layer on the flattened token
                if self.single_kernel:
                    token = self.encoder_decoder(token)
                else:
                    token = self.encoder_decoders[i][j](token)

                # Reshape the token to the out shape
                token = tf.reshape(
                    token,
                    [
                        batch_size,
                        self.kernel_output_size,
                        self.kernel_output_size,
                        self.channels,
                    ],
                )

                # Insert the token into the output
                row.append(token)

            grid.append(tf.concat(row, axis=2))  # Concatenate the row along the width

        output = tf.concat(grid, axis=1)

        if self.use_residual:
            output = tf.keras.layers.Add()([output, inputs])

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


    def get_config(self):
        """
        Returns the configuration of the layer for serialization.
        """

        config = super(EDLight2DConv, self).get_config()
        config.update(
            {
                "kernel_size": self.kernel_size,
                "channels": self.channels,
                "hidden_units": self.hidden_units,
                "kernel_output_size": self.kernel_output_size,
                "activation": self.activation,
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
                "use_residual": self.use_residual,
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
