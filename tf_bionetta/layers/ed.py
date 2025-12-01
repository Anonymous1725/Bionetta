"""
Encoder Decoder Layer (EDLayer) implementation
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import tensorflow as tf
from keras import regularizers, initializers
from keras.layers import InputSpec

from tf_bionetta.layers.interface import BionettaLayer
from tf_bionetta.constraints.types.layer_complexity import LayerComplexity


class EncoderDecoderLayer(BionettaLayer):
    """
    Encoder Decoder Layer (EDLayer) implementation. This is a zk-cost-effective layer that
    is used as a substitute for the default dense layer. The main advantage
    of this layer is that requires significantly less activation functions
    than the regular dense layer.

    As an input, EDLayer requires input size and output size (similarly to
    the dense layer). In constrast to the dense layer, we also require to
    specify the number of hidden neurons and the activation function.
    """

    DEFAULT_HIDDEN_SIZE: int = 16  # Default number of hidden neurons
    DEFAULT_ACTIVATION: tf.keras.layers.Layer = (
        tf.keras.layers.ReLU()
    )  # Default activation function

    def __init__(
        self,
        units: int,
        hidden_units: int = DEFAULT_HIDDEN_SIZE,
        activation: Callable[[Any], Any] | tf.keras.layers.Layer = DEFAULT_ACTIVATION,
        kernel_initializer: str = "glorot_normal",
        kernel_regularizer: regularizers.Regularizer = tf.keras.regularizers.l2(1e-4),
        **kwargs,
    ) -> None:
        """
        Initializes the Encoder Decoder Layer.

        Args:
            - units (`int`): The number of neurons for the output
            - hidden_units (`int`, optional): The number of hidden neurons. Defaults to 10.
            - activation (`str`, optional): The activation function in the hidden layer. Defaults to 'relu'.
        """

        super(EncoderDecoderLayer, self).__init__(**kwargs)

        if type(activation) == dict:
            activation = tf.keras.layers.deserialize(activation)

        assert isinstance(
            activation, (Callable, tf.keras.layers.Layer)
        ), "Activation must be a Keras Layer. If you specified a string, please use `tf.keras.layers` instead."

        self.units = units  # Number of neurons in the output layer
        self.hidden_units = hidden_units
        # Just in case someone wants to put 'None' as an activation function
        self.activation = (
            EncoderDecoderLayer.DEFAULT_ACTIVATION if activation is None else activation
        )

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        # We expect the 2D input of shape (batch_size, input_size)
        self.input_spec = InputSpec(ndim=2)


    def build(self, input_shape: tf.TensorShape, test_mode: bool = False) -> None:
        """
        Builds the layer. This method is called once the input shape is known.
        The layer simply consists of three
        """

        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError(
                "The last dimension of the inputs to the Encoder-Decoder layer "
                "should be defined. Found None. "
                f"Full input shape received: {input_shape}"
            )

        self.hidden_layer = tf.keras.layers.Dense(
            self.hidden_units,
            activation=None,
            kernel_initializer="he_normal",
            kernel_regularizer=self.kernel_regularizer,
            name="hidden_layer",
        )
        self.hidden_layer_batch_norm = tf.keras.layers.BatchNormalization()
        self.decoder_layer = tf.keras.layers.Dense(
            self.units,
            activation=None,
            kernel_initializer="he_normal",
            kernel_regularizer=self.kernel_regularizer,
            name="decoder_layer",
        )
        self.decoder_layer_batch_norm = tf.keras.layers.BatchNormalization()

        # Force the sublayers to build by calling them with a dummy input.
        if test_mode:
            dummy_input = tf.zeros(input_shape)
            _ = self.hidden_layer(dummy_input)
            dummy_input = tf.zeros(self.hidden_layer.compute_output_shape(input_shape))
            _ = self.hidden_layer_batch_norm(dummy_input)
            dummy_input = tf.zeros(
                self.hidden_layer_batch_norm.compute_output_shape(dummy_input.shape)
            )
            _ = self.decoder_layer(dummy_input)
            dummy_input = tf.zeros(
                self.decoder_layer.compute_output_shape(dummy_input.shape)
            )
            _ = self.decoder_layer_batch_norm(dummy_input)

        self.built = True

        # Calculate the complexity of the layer (for further constraints estimation)
        self.complexity = self.compute_complexity(input_shape)


    def compute_complexity(
        self, 
        _: tf.TensorShape
    ) -> LayerComplexity:
        """
        Calculates the complexity of the layer.
        
        Args:
            _: (`tf.TensorShape`): The shape of the input tensor. Not used in this layer.
        
        Returns:
            LayerComplexity: The estimated complexity of the layer.
        """

        return LayerComplexity(
            mul_ops=0,
            non_linear_ops=[
                (self.activation, self.hidden_units)
            ],
        )


    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Calls the layer. This method is called when the layer is used in a model.
        """

        x = self.hidden_layer(inputs)
        x = self.hidden_layer_batch_norm(x)
        x = self.activation(x)
        x = self.decoder_layer(x)
        x = self.decoder_layer_batch_norm(x)

        return x


    def compute_output_shape(self, input_shape: Tuple) -> tf.TensorShape:
        """
        Returns the output shape of the layer
        """

        input_shape = tf.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tf.compat.dimension_value(input_shape[-1]) is None:
            raise ValueError(
                "The last dimension of the input shape of a Dense layer "
                "should be defined. Found None. "
                f"Received: input_shape={input_shape}"
            )

        return input_shape[:-1].concatenate(self.units)
