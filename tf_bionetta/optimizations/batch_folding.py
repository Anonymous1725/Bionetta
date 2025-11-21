"""
Helper functions for Batch Normalization folding.
"""

import tensorflow as tf
import numpy as np

SUPPORTED_BN_FOLDING_LAYERS = (
    tf.keras.layers.Conv2D,
    tf.keras.layers.Dense,
    tf.keras.layers.DepthwiseConv2D,
)


def fold_batch_norm(
    layer: tf.keras.layers.Layer,
    cloned_layer: tf.keras.layers.Layer,
    bn_layer: tf.keras.layers.Layer,
) -> tf.keras.layers.Layer:
    """
    Folds BatchNorm layer with the previous Conv2D layer.

    Arguments:
        - layer (tf.keras.layers.Layer) - Conv2D layer
        - bn_layer (tf.keras.layers.Layer) - BatchNorm layer
    """

    if not isinstance(bn_layer, tf.keras.layers.BatchNormalization):
        raise ValueError("The second layer must be a BatchNormalization layer")

    if not isinstance(layer, SUPPORTED_BN_FOLDING_LAYERS):
        return (
            None  # Indicating that the layer is not supported and thus cannot be folded
        )

    new_layer = cloned_layer

    # Add bias if the layer does not have it
    if not layer.use_bias:
        new_layer = add_bias_to_layer(new_layer)

    assert new_layer.use_bias, "The layer must have bias"
    assert len(new_layer.weights) > 0, "The layer must have some weights"

    # Get BatchNorm parameters
    gamma, beta, moving_mean, moving_variance = bn_layer.get_weights()
    epsilon = bn_layer.epsilon

    # Getting weights
    weights = new_layer.get_weights()
    kernel, bias = weights

    # Compute std and new bias
    std = np.sqrt(moving_variance + epsilon)
    new_bias = gamma * (bias - moving_mean) / std + beta

    # For depthwise convolution, we need to reshape gamma and std separately
    if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
        depth_multiplier = kernel.shape[-1]
        gamma = tf.reshape(gamma, (-1, depth_multiplier))
        std = tf.reshape(std, (-1, depth_multiplier))

    new_kernel = kernel * gamma / std

    # Set new weights, depending on whether the layer uses bias or not
    new_layer.set_weights([new_kernel, new_bias])
    return new_layer


def add_bias_to_layer(layer: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
    """
    Adds bias to the layer if it does not have it.

    Arguments:
        - layer (tf.keras.layers.Layer) - layer to add bias to
    """

    weights = layer.get_weights()

    assert len(weights) > 0, "The layer must have some weights at least"

    if not isinstance(layer, SUPPORTED_BN_FOLDING_LAYERS):
        print(f"Layer {layer} is not supported for bias addition")
        return layer  # Indicating that the layer is not supported and thus cannot be bias-added

    if isinstance(layer, tf.keras.layers.Conv2D):
        kernel = weights[0]

        # Add bias first
        layer_config = layer.get_config()
        layer_config["use_bias"] = True
        new_layer = tf.keras.layers.Conv2D(**layer_config)
        new_layer.build(layer.input_shape)

        # Create dummy zero bias
        bias = np.zeros(
            layer.filters,
        )
        new_layer.set_weights([kernel, bias])
        return new_layer

    if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
        kernel = weights[0]

        # Add bias first
        layer_config = layer.get_config()
        layer_config["use_bias"] = True
        new_layer = tf.keras.layers.DepthwiseConv2D(**layer_config)
        new_layer.build(layer.input_shape)

        # Create dummy zero bias
        channels, depth_multiplier = kernel.shape[-2], kernel.shape[-1]
        bias = np.zeros(
            channels * depth_multiplier,
        )
        new_layer.set_weights([kernel, bias])
        return new_layer

    if isinstance(layer, tf.keras.layers.Dense):
        kernel = weights[0]
        # Add bias first
        layer_config = layer.get_config()
        layer_config["use_bias"] = True
        new_layer = tf.keras.layers.Dense(**layer_config)
        new_layer.build(layer.input_shape)

        # Create dummy zero bias
        bias = np.zeros(
            layer.units,
        )
        new_layer.set_weights([kernel, bias])
        return new_layer

    return layer


def identity_batch_norm(input_shape, name=None):
    """
    Returns a BatchNormalization layer that behaves as an identity function.

    Parameters:
        input_shape (tuple): Shape of the input tensor (excluding batch size).
        name (str): Optional name for the layer.

    Returns:
        BatchNormalization layer configured as identity.
    """
    bn = tf.keras.layers.BatchNormalization(name=name)
    # Build the layer to initialize weights
    bn.build((None,) + input_shape)

    # Set gamma = 1, beta = 0, moving_mean = 0, moving_variance = 1
    bn.gamma.assign(tf.ones_like(bn.gamma))
    bn.beta.assign(tf.zeros_like(bn.beta))
    bn.moving_mean.assign(tf.zeros_like(bn.moving_mean))
    bn.moving_variance.assign(tf.ones_like(bn.moving_variance))

    # Freeze the layer to prevent updates during training
    bn.trainable = False

    return bn
