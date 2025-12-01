"""
Debug Layer that prints the output of a layer during the forward pass. Needed
for debugging purposes when the model's output is mismatched with the
corresponding Rust or Circom implementation.
"""

from pathlib import Path
import numpy as np
import tensorflow as tf


def DebugLayer(
    layer_name: str,
    save_path: Path | None = None,
) -> tf.keras.layers.Layer:
    """
    Returns a Lambda layer that logs the tensor with the provided layer name.

    Args:
        layer_name (str): The name of the layer.
        save_path (Path, optional): The path to save the tensor for the given layer. If None, the tensor
        is not saved. Defaults to None.
    """

    save_path = Path(save_path) if save_path is not None else None

    def print_tensor(x: tf.Tensor) -> tf.Tensor:
        """
        Lambda function that prints the tensor and returns the input
        without any modifications.
        """

        tf.print(f"Layer {layer_name} output:", x)

        if save_path is not None:
            weights_path = save_path / f"{layer_name}.txt"
            np.savetxt(weights_path, x.numpy())

        # The layer does nothing, so simply return the input
        return x

    return tf.keras.layers.Lambda(print_tensor)


def Debugger(
    layer: tf.keras.layers.Layer,
    save_path: Path | None = None,
) -> tf.keras.layers.Layer:
    """
    Wraps the provided layer with a DebugLayer.

    Args:
        layer (tf.keras.layers.Layer): The layer to be wrapped.
        layer_name (str): The name of the layer.
        save_path (Path, optional): The path to save the tensor for the given layer. If None, the tensor
        is not saved. Defaults to None.
    """

    return tf.keras.Sequential([layer, DebugLayer(layer.name, save_path)])
