"""
Package for implementing the Encoder-Decoder Light Convolution Layer
"""

from typing import Tuple

import tensorflow as tf
from keras import regularizers
from keras import initializers
from keras.layers import InputSpec

from tf_bionetta.layers.interface import BionettaLayer, LayerComplexity


class EDConv2D(BionettaLayer):
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
        hidden_layer_size: int,
        hidden_layer_channels: int,
        output_layer_size: int,
        output_layer_channels: int,
        kernel_size: int = 3,
        activation: tf.keras.layers.Layer | None = DEFAULT_ACTIVATION,
        kernel_initializer: str | None = "glorot_normal",
        kernel_regularizer: regularizers.Regularizer | None = tf.keras.regularizers.l2(
            1e-4
        ),
        use_residual: bool = False,
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

        super(EDConv2D, self).__init__(**kwargs)

        self.hidden_layer_size = hidden_layer_size
        self.hidden_layer_channels = hidden_layer_channels
        self.output_layer_size = output_layer_size
        self.output_layer_channels = output_layer_channels
        self.kernel_size = kernel_size
        self.activation = activation

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.use_residual = use_residual

        # We expect the 4D input of shape (batch_size, height, width, channels)
        self.input_spec = InputSpec(ndim=4)

    def build(self, input_shape: Tuple) -> None:
        """
        Builds the layer. This method is called once the input shape is known.
        """

        input_shape = tf.TensorShape(input_shape)
        (_, height, width, input_channels) = input_shape  # Extract the input shape

        assert width == height, "Currently only square images are supported."
        assert (
            width % self.hidden_layer_size == 0
        ), f"Width {width} is not divisible by the hidden layer size {self.hidden_layer_size}"
        assert (
            self.output_layer_size % self.hidden_layer_size == 0
        ), f"Output layer size {self.output_layer_size} is not divisible by the hidden layer size {self.hidden_layer_size}"

        stride = width // self.hidden_layer_size

        if self.use_residual:
            assert (
                width == self.output_layer_size
            ), "Residual connections require the output size to be the same as the input size."
            assert (
                self.output_layer_channels == input_channels
            ), "Residual connections require the input and output channels to be the same."

        self.encoder_conv = tf.keras.layers.Conv2D(
            self.hidden_layer_channels,
            kernel_size=self.kernel_size,
            strides=stride,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            padding="same",
            name="encoder_conv",
        )
        self.encoder_bn = tf.keras.layers.BatchNormalization(momentum=0.95)

        upsampling_size = self.output_layer_size // self.hidden_layer_size
        self.upsampling = tf.keras.layers.UpSampling2D(
            size=upsampling_size, interpolation="nearest"
        )
        self.decoder_conv = tf.keras.layers.Conv2D(
            self.output_layer_channels,
            kernel_size=self.kernel_size,
            strides=1,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            padding="same",
            name="decoder_conv",
        )
        self.decoder_bn = tf.keras.layers.BatchNormalization()

        # Calculate the model's complexity (for further constraints estimation)
        self.complexity = self.compute_complexity(input_shape)


    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Computes the output of the layer.
        """

        assert len(inputs.shape) == 4, f"Expected 4D input, got {len(inputs.shape)}"

        x = self.encoder_conv(inputs)
        x = self.encoder_bn(x)
        x = self.activation(x)
        x = self.upsampling(x)
        output = self.decoder_conv(x)
        output = self.decoder_bn(output)

        if self.use_residual:
            output = tf.keras.layers.Add()([output, inputs])

        return output


    def compute_output_shape(self, input_shape: Tuple) -> Tuple:
        """
        Computes the output shape of the layer.
        """

        batch_size = input_shape[0]
        return tf.TensorShape(
            [
                batch_size,
                self.output_layer_size,
                self.output_layer_size,
                self.output_layer_channels,
            ]
        )

    def get_config(self):
        """
        Returns the configuration of the layer for serialization.
        """

        config = super(EDConv2D, self).get_config()
        config.update(
            {
                "hidden_layer_size": self.hidden_layer_size,
                "hidden_layer_channels": self.hidden_layer_channels,
                "output_layer_size": self.output_layer_size,
                "output_layer_channels": self.output_layer_channels,
                "kernel_size": self.kernel_size,
                "activation": self.activation,
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
                "use_residual": self.use_residual,
            }
        )

        return config


    def compute_complexity(
        self, 
        input_shape: tf.TensorShape
    ) -> LayerComplexity:
        """
        Calculate the number of constraints for the layer.
        
        Args:
            - input_shape (`tf.TensorShape`): The shape of the input tensor.
            
        Output:
            - `LayerComplexity`: An object containing the number of multiplications
              and non-linear operations in the layer.
        """

        input_shape = tf.TensorShape(input_shape)
        return LayerComplexity(
            mul_ops=0,
            non_linear_ops=[
                (self.activation, self.hidden_layer_channels*(self.hidden_layer_size**2))
            ],
        )
        

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
