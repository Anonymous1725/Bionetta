"""
Module with the BioNetV3 embedding model.
"""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf

from tf_bionetta.layers import EDLight2DConv, SEHeavyBlock, L2UnitNormalizationLayer

SUPPORTED_IMG_SHAPE = (40, 40)
MODEL_NAME: str = "BioNetV1"


def BioNetV1(
    output_size: int,
    input_shape: Tuple[int, int, int] = (*SUPPORTED_IMG_SHAPE, 3),
) -> tf.keras.models.Model:
    """
    Embedding model is a target model that we need to train to make predictions.

    Arguments:
        - input_shape: Shape of the input tensor.
        - embedding_size: Size of the output embedding.
    """

    # Asserting the correct shape
    assert (
        input_shape[:2] == SUPPORTED_IMG_SHAPE
    ), f"Input shape {input_shape} is not supported."
    if len(input_shape) == 2:
        input_shape = input_shape + (1,)

    # Defining the model
    inputs = tf.keras.layers.Input(shape=input_shape)

    # First convolution to get multiple channels
    x = init_layer(inputs, 2, 16)  # Output: (20, 20, 16)

    # A series of bottleneck layers
    x = bottleneck(
        x,
        in_channels=16,
        out_channels=16,
        kernel=5,
        channels_squeeze=4,
        hidden_units=18,
        kernel_squeeze=2,
    )

    x = residual_bottleneck(
        x,
        in_channels=16,
        se_hidden_units=16,
        ed_hidden_units=16,
        expansion=4,
        kernel=4,
        squeeze=16,
    )
    x = bottleneck(
        x,
        in_channels=16,
        out_channels=48,
        kernel=4,
        hidden_units=12,
        channels_squeeze=4,
        kernel_squeeze=1,
    )

    x = residual_bottleneck(
        x,
        in_channels=48,
        expansion=8,
        kernel=4,
        se_hidden_units=32,
        ed_hidden_units=32,
        squeeze=32,
    )

    x = bottleneck(
        x,
        in_channels=48,
        out_channels=96,
        kernel=2,
        hidden_units=24,
        channels_squeeze=8,
        kernel_squeeze=2,
    )

    x = residual_bottleneck(
        x,
        in_channels=96,
        expansion=12,
        depthwise_expansion=True,
        kernel=2,
        se_hidden_units=128,
        ed_hidden_units=32,
        squeeze=64,
    )

    x = bottleneck(
        x,
        in_channels=96,
        out_channels=256,
        kernel=2,
        hidden_units=64,
        channels_squeeze=32,
        kernel_squeeze=1,
    )

    x = bottleneck(
        x,
        in_channels=256,
        out_channels=512,
        kernel=2,
        hidden_units=128,
        channels_squeeze=16,
        kernel_squeeze=1,
    )

    # Last stage of the model
    x = tf.keras.layers.GlobalAveragePooling2D()(x)  # (512,)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=1 / 128)(x)

    # Output embedding layer
    x = tf.keras.layers.Dense(
        output_size, activation=None, kernel_initializer="orthogonal"
    )(x)

    outputs = L2UnitNormalizationLayer()(x)

    return tf.keras.models.Model(inputs=inputs, outputs=outputs, name=MODEL_NAME)


def init_layer(
    x: tf.Tensor,
    stride: int,
    output_channels: int,
) -> tf.Tensor:
    """
    Layers that get executed at the beginning of the model.
    """

    x = tf.keras.layers.Conv2D(
        filters=output_channels,
        kernel_size=3,
        use_bias=False,
        strides=stride,
        kernel_initializer="he_normal",
        padding="same",
    )(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999)(x)
    return x


def residual_bottleneck(
    inputs: tf.Tensor,
    in_channels: int,
    se_hidden_units: int,
    ed_hidden_units: int,
    expansion: int,
    kernel: int,
    squeeze: int,
    depthwise_expansion: bool = False,
) -> tf.Tensor:
    """
    Bottleneck layer for the BioNetV3 model.

    Arguments:
        - x (tf.Tensor) - input tensor
        - filters (int) - number of filters for the convolutional layer
        - stride (int) - stride for the convolutional layer
        - kernel_size (int) - kernel size for the EDLightConv layer

    Returns:
        - tf.Tensor - output tensor
    """

    if in_channels * expansion % squeeze != 0:
        print(
            f"Warning: Squeeze factor {squeeze} is not a multiple of the input channels {in_channels*expansion}."
        )

    # 1. Expand the number of channels via convolution
    if depthwise_expansion:
        x = tf.keras.layers.DepthwiseConv2D(
            kernel_size=1,
            strides=1,
            depth_multiplier=expansion,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
        )(inputs)
    else:
        x = tf.keras.layers.Conv2D(
            in_channels * expansion,
            kernel_size=1,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
        )(inputs)
    x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999)(x)

    # 2. Apply the Squeeze-and-Excitation block on the expanded channels
    x = SEHeavyBlock(
        kernel_size=kernel,
        hidden_units=se_hidden_units,
        single_kernel=False,
        activation=tf.keras.layers.LeakyReLU(alpha=1 / 128),
        kernel_initializer="he_normal",
    )(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999)(x)

    # 3. Squeeze the channels via convolution
    x = tf.keras.layers.Conv2D(
        filters=in_channels * expansion // squeeze,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
    )(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999)(x)

    # 4. Apply the Encoder-Decoder layer on the squeezed channels
    x = EDLight2DConv(
        kernel_size=kernel,
        channels=in_channels,
        hidden_units=ed_hidden_units,
        kernel_output_size=kernel,
        single_kernel=False,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # 5. Use residual connection
    x = tf.keras.layers.Add()([x, inputs])
    return x


def bottleneck(
    inputs: tf.Tensor,
    in_channels: int,
    out_channels: int,
    kernel: int,
    hidden_units: int,
    channels_squeeze: int,
    kernel_squeeze: int,
) -> tf.Tensor:
    """
    Bottleneck layer for the BioNetV3 model.

    Arguments:
        - x (tf.Tensor) - input tensor
        - filters (int) - number of filters for the convolutional layer
        - stride (int) - stride for the convolutional layer
        - kernel_size (int) - kernel size for the EDLightConv layer

    Returns:
        - tf.Tensor - output tensor
    """

    # 1. Squeeze the number of channels via convolution
    x = tf.keras.layers.Conv2D(
        filters=in_channels // channels_squeeze,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
    )(inputs)
    x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999)(x)

    # 2. Apply the Encoder-Decoder layer on the squeezed channels
    x = EDLight2DConv(
        kernel_size=kernel,
        channels=out_channels,
        hidden_units=hidden_units,
        kernel_output_size=kernel // kernel_squeeze,
        single_kernel=False,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    return x
