"""
Package for implementing the Encoder-Decoder Light Convolution Layer
"""

from typing import Tuple

import tensorflow as tf
from keras import regularizers
from keras import initializers
from keras.layers import InputSpec

from tf_bionetta.layers.interface import BionettaLayer, LayerComplexity


class Gated2DConv(BionettaLayer):
    """
    Gated 2D Convolutional Layer implementation. This layer is zk-cost-effective
    alternative to the regular convolutional layer.
    """

    DEFAULT_ACTIVATION = tf.keras.layers.ReLU()

    def __init__(
        self,
        kernel_size: int,
        channels: int,
        activation: tf.keras.layers.Layer | None = DEFAULT_ACTIVATION,
        kernel_initializer: str | None = "glorot_normal",
        kernel_regularizer: regularizers.Regularizer | None = tf.keras.regularizers.l2(1e-4),
        squeeze_factor: int = False,
        image_activation: bool = False,
        **kwargs,
    ) -> None:
        """
        Initializes the Gated 2D Convolutional Layer.

        Args:
            - kernel_size (`int`): Size of the kernel (filter) to be applied.
            - channels (int): Number of output channels (filters).
            - activation (`str`, optional): The activation function in the hidden layer. Defaults to 'relu'.
            - kernel_initializer (`str`, optional): The initializer for the kernel weights. Defaults to 'glorot_normal'.
            - kernel_regularizer (`regularizers.Regularizer`, optional): The regularizer for the kernel weights. Defaults to L2 with 1e-4.
            - squeeze_factor (`int`, optional): The factor by which to squeeze the input tensor. Defaults to False.
        """

        super(Gated2DConv, self).__init__(**kwargs)

        self.kernel_size = kernel_size
        self.channels = channels
        self.activation = activation
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.squeeze_factor = squeeze_factor
        self.image_activation = image_activation

        self.input_spec = InputSpec(
            ndim=4
        )  # We expect the 4D input of shape (batch_size, height, width, channels)


    def build(self, input_shape: Tuple) -> None:
        """
        Builds the layer. This method is called once the input shape is known.
        """

        input_shape = tf.TensorShape(input_shape)
        (_, height, width, channels) = input_shape  # Extract the input shape

        # Assert that the squeeze factor is valid
        assert (
            self.channels % channels == 0
        ), "Output channels must be divisible by input channels."
        assert (
            height % self.squeeze_factor == 0
        ), "Height must be divisible by the squeeze factor."
        assert (
            width % self.squeeze_factor == 0
        ), "Width must be divisible by the squeeze factor."

        self.attention_conv = tf.keras.layers.DepthwiseConv2D(
            strides=self.squeeze_factor,
            kernel_size=self.kernel_size,
            depth_multiplier=self.channels // channels,
            padding="same",
        )
        self.avg_pool = tf.keras.layers.AveragePooling2D(
            strides=self.squeeze_factor, padding="same"
        )

        # Calculate the complexity of the layer (for further constraints estimation)
        self.complexity = self.compute_complexity(input_shape)


    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Computes the output of the layer.
        """

        assert len(inputs.shape) == 4, f"Expected 4D input, got {len(inputs.shape)}"

        # First, we apply the attention convolution
        # and conduct the global average pooling
        # to get self.channels attention maps
        x = self.attention_conv(
            inputs
        )  # Output shape: (batch_size, width/squeeze, height/squeeze, output_channels)
        x = tf.reduce_mean(
            x, axis=[1, 2], keepdims=True
        )  # Output shape: (batch_size, output_channels)
        x = self.activation(x)  # Output shape: (batch_size, output_channels)

        # Next, we average the input tensor
        # along the channel axis to get
        # the widthxheight attention map
        y = self.avg_pool(
            inputs
        )  # Output shape: (batch_size, height/squeeze, width/squeeze, input_channels)
        y = tf.reduce_mean(
            y, axis=3, keepdims=True
        )  # Output shape: (batch_size, height/squeeze, width/squeeze, 1)
        if self.image_activation:
            y = self.activation(
                y
            )  # Output shape: (batch_size, height/squeeze, width/squeeze, 1)

        # Now, we multiply y by each neuron in the attention map
        # to get the output tensor
        output = y * x
        return output


    def compute_output_shape(self, input_shape: Tuple) -> Tuple:
        """
        Computes the output shape of the layer.
        """

        input_shape = tf.TensorShape(input_shape)

        (batch_size, height, width, _) = input_shape  # Extract the input shape
        new_height = height // self.squeeze_factor
        new_width = width // self.squeeze_factor

        return tf.TensorShape([batch_size, new_height, new_width, self.channels])


    def get_config(self):
        """
        Returns the configuration of the layer for serialization.
        """

        config = super(Gated2DConv, self).get_config()
        config.update(
            {
                "kernel_size": self.kernel_size,
                "channels": self.channels,
                "activation": self.activation,
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
                "squeeze_factor": self.squeeze_factor,
                "image_activation": self.image_activation,
            }
        )

        return config


    def compute_complexity(
        self, 
        input_shape: tf.TensorShape
    ) -> LayerComplexity:
        """
        Computes the complexity of the layer: number of multiplications
        and non-linear operations, based on the input shape.
        
        Input:
            - input_shape (`tf.TensorShape`): The shape of the input tensor.
            
        Output:
            - `LayerComplexity`: An object containing the number of multiplications
              and non-linear operations in the layer.
        """

        input_shape = tf.TensorShape(input_shape)
        (_, height, width, _) = input_shape
        
        if self.image_activation:
            return LayerComplexity(
                mul_ops=0,
                non_linear_ops=[
                    (self.activation, width*height//(self.squeeze_factor**2))
                ],
            )
            
        return LayerComplexity(
            mul_ops=width*height*self.channels//(self.squeeze_factor**2),
            non_linear_ops=[
                (self.activation, self.channels)
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
