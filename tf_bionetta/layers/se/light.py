"""
Implementation of Squeeze-and-Excitation Layer
"""

from __future__ import annotations

from typing import Any, Callable, Tuple, Dict, Annotated

import tensorflow as tf
from keras import regularizers, initializers
from keras.layers import InputSpec

from tf_bionetta.layers.interface import BionettaLayer
from tf_bionetta.layers.ed import EncoderDecoderLayer
from tf_bionetta.constraints.types.layer_complexity import LayerComplexity


class SELightBlock(BionettaLayer):
    """
    Implementation of the regular Squeeze-and-Excitation Layer.
    The implementation is based on the following paper:

    https://arxiv.org/pdf/1709.01507
    """

    DEFAULT_ACTIVATION: Annotated[
        tf.keras.layers.Layer,
        "Default activation function used after the encoding step in the ED Layer",
    ] = tf.keras.layers.ReLU()


    def __init__(
        self,
        hidden_units: int = 10,
        activation: tf.keras.layers.Layer = DEFAULT_ACTIVATION,
        kernel_initializer: str | None = "glorot_normal",
        kernel_regularizer: regularizers.Regularizer | None = tf.keras.regularizers.l2(
            1e-4
        ),
        **kwargs,
    ) -> None:
        """
        Initializes the Squeeze-and-Excitation Layer.
        """

        super(SELightBlock, self).__init__(**kwargs)

        if type(activation) == dict:
            activation = tf.keras.layers.deserialize(activation)

        assert isinstance(
            activation, (Callable, tf.keras.layers.Layer)
        ), "Activation must be a Keras Layer. If you specified a string, please use `tf.keras.layers` instead."

        self.hidden_units = hidden_units
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
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError(
                "The last dimension of the inputs to a Squeeze-and-Excitation layer "
                "should be defined. Found None. "
                f"Full input shape received: {input_shape}"
            )

        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.encoder_decoder = EncoderDecoderLayer(
            units=last_dim,
            hidden_units=self.hidden_units,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
        )

        # Force the sublayers to build by calling them with a dummy input.
        if is_test_mode:
            dummy_input = tf.zeros((1, last_dim))
            _ = self.encoder_decoder(dummy_input)

        # Calculate the complexity of the layer (for further constraints estimation)
        self.complexity = self.compute_complexity(input_shape)


    def compute_complexity(
        self, 
        input_shape: tf.TensorShape
    ) -> LayerComplexity:
        """
        Calculates the complexity of the layer.
        
        Args:
            input_shape (tf.TensorShape): The shape of the input tensor.
        
        Returns:
            LayerComplexity: The estimated complexity of the layer.
        """

        input_shape = tf.TensorShape(input_shape)

        (_, height, width, channels) = input_shape
        return LayerComplexity(
            mul_ops=height*width*channels,
            non_linear_ops=[
                (self.activation, self.hidden_units)
            ]
        )


    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Calls the layer. This method is called when the layer is used in a model.
        """

        batch_size = tf.shape(inputs)[0]
        channels = tf.shape(inputs)[-1]

        x = self.global_avg_pool(inputs)
        x = self.encoder_decoder(x)
        x = tf.reshape(x, (batch_size, 1, 1, channels))
        output = inputs * x

        return output


    def compute_output_shape(self, input_shape: Tuple) -> Tuple:
        """
        Computes the output shape of the layer.
        """

        input_shape = tf.TensorShape(input_shape)
        return input_shape  # The shape does not change

    def get_config(self) -> Dict:
        """
        Returns the configuration of the layer for serialization.
        """

        config = super(SELightBlock, self).get_config()
        config.update(
            {
                "hidden_units": self.hidden_units,
                "activation": self.activation,
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict) -> SELightBlock:
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
