"""
Module for compressing the neural network to the `.tflite` format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import tensorflow as tf
import numpy as np


class ModelQuantizer:
    """
    Class for compressing the neural network to the `.tflite` format.
    """

    def __init__(
        self,
        model: tf.keras.models.Model,
    ) -> None:
        """
        Initializes the quantizer instance.

        Args:
            - model (`tf.keras.models.Model`): The model to be quantized.
        """

        self.model = model


    def form_quantized_model(self) -> bytes:
        """
        Converts the neural network to the compressed format
        """

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # If u get error Segmentation fault it means that the model contains broken weights(e.x. NaNs) and u need to another model.
        # This issue can be faced on TF 2.13.
        tflite_model = converter.convert()

        return tflite_model


    def save(self, path: Path) -> None:
        """
        Saves the compressed model to the specified path.

        Args:
            - path (`Path`): The path to save the compressed model.
        """

        if not isinstance(path, Path):
            path = Path(path)

        tflite_model = self.form_quantized_model()

        # If path is a folder, append the model name and save the model
        if path.is_dir():
            path = path / f"{self.model.name}.tflite"

        # Assert that the resultant path has the .tflite extension
        assert path.suffix == ".tflite", "The path must have the .tflite extension"
        path.write_bytes(tflite_model)


    def form_hard_quantized_model(
        self,
        test_inputs: np.ndarray | tf.Tensor,
    ) -> bytes:
        """
        Converts the neural network to the compressed format with hard quantization
        """

        assert test_inputs is not None, "Test inputs must be provided"
        assert len(test_inputs) > 0, "At least one test input must be provided"

        def representative_dataset() -> Generator:
            """
            Just a test input generator that takes
            the test inputs and yields them one by one
            """

            for i in range(len(test_inputs)):
                test_input = test_inputs[i]
                test_input = np.float32(test_input)
                test_input = np.expand_dims(test_input, axis=0)
                yield [test_input]

        # Converting the model to .tflite format
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()

        return tflite_model


    def save_heavy_quantization(
        self,
        path: Path,
        test_inputs: np.ndarray | tf.Tensor,
    ) -> None:
        """
        Saves the compressed model to the specified path with hard quantization.

        Args:
            - path (`Path`): The path to save the compressed model.
            - test_inputs (`np.ndarray` | `tf.Tensor`): The test inputs to be used for quantization.
        """

        if not isinstance(path, Path):
            path = Path(path)

        tflite_model = self.form_hard_quantized_model(test_inputs)

        # If path is a folder, append the model name and save the model
        if path.is_dir():
            path = path / f"{self.model.name}.tflite"

        # Assert that the resultant path has the .tflite extension
        assert path.suffix == ".tflite", "The path must have the .tflite extension"
        path.write_bytes(tflite_model)
