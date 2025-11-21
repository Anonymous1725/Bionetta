"""
Custom callback to show predictions after each epoch.
"""

import logging

import tensorflow as tf
import numpy as np

from tf_bionetta.logging.logger import create_logger
from tf_bionetta.logging.verbose import VerboseMode


class ExamplePredictionsCallback(tf.keras.callbacks.Callback):
    """
    Callback to show predictions after each epoch.
    """

    def __init__(
        self,
        test_inputs: tf.Tensor | np.ndarray,
        test_labels: tf.Tensor | np.ndarray | None = None,
        model: tf.keras.models.Model | None = None,
        logger: logging.Logger | None = None,
        **kwargs,
    ) -> None:
        """
        Custom callback to show predictions after each epoch.

        Args:
            - test_inputs (`tf.Tensor` or `np.ndarray`): A batch of test data.
            - test_labels (`tf.Tensor` or `np.ndarray`, optional): A batch of
              test labels, corresponding to test_inputs. Defaults to `None`.
            - model (`tf.keras.models.Model`, optional): The model to predict
              with. Defaults to `None`.
            - logger (`logging.Logger`, optional): A logger object. Defaults to
              `None`.
        """

        super().__init__(**kwargs)

        self.test_inputs = test_inputs
        self.test_labels = test_labels
        self.custom_model = model
        self.logger = logger if logger is not None else create_logger(VerboseMode.INFO)

    def on_epoch_end(self, epoch, _logs=None):
        """
        Called at the end of each epoch.
        """

        DECIMALS_TO_ROUND = (
            5  # Number of decimals to round the predictions to for debugging
        )

        # Get model to predict with
        model = self.custom_model if self.custom_model is not None else self.model

        # Get model predictions
        predictions = model.predict(self.test_inputs, verbose=0)
        formatted_predictions = np.round(predictions, DECIMALS_TO_ROUND)

        # Select a few examples to display
        self.logger.info(
            f"Epoch {epoch+1}: Showing predictions for {len(predictions)} examples"
        )
        for i, prediction in enumerate(formatted_predictions):
            self.logger.info(f"Example {i+1}: {prediction}")
            if self.test_labels is not None:
                self.logger.info(
                    f"Expected Label for example {i+1}: {self.test_labels[i]}"
                )
