"""
Package for saving the specific part of the model after each epoch
"""

from pathlib import Path
import logging

import tensorflow as tf

from tf_bionetta import BionettaModel
from tf_bionetta.save import BionettaModelSaver
from tf_bionetta.logging.logger import create_logger
from tf_bionetta.logging.verbose import VerboseMode


class BionettaSubmodelCheckpoint(tf.keras.callbacks.Callback):
    """
    Callback that saves the selected layers of the Bionetta-based model after
    each epoch.
    """

    def __init__(
        self,
        model: tf.keras.models.Model,
        base_path: Path,
        logger: logging.Logger | None = None,
        start_epoch: int = 0,
        save_weights_only: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """
        Checkpoint for saving the selected layers of the model after each epoch.
        """

        super().__init__(*args, **kwargs)

        # If the model is a Bionetta model, we extract the base model
        if isinstance(model, BionettaModel):
            model = model.base_model

        assert isinstance(model, tf.keras.models.Model), "Model must be a Keras model."

        self.base_path = Path(base_path)
        assert isinstance(self.base_path, Path), "Base path must be a Path object."

        self.saver = BionettaModelSaver(model, logger=logger, ignore_errors=True)
        self.logger = logger if logger is not None else create_logger(VerboseMode.INFO)
        assert start_epoch >= 0, "Start epoch must be >= 0."
        self.start_epoch = start_epoch

        self.save_weights_only = save_weights_only

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        """
        Performs the action of saving the model after each epoch.
        """

        # Save this partial model
        epoch_path = self.base_path / f'epoch_{epoch + self.start_epoch + 1 }'
        epoch_path.mkdir(parents=True, exist_ok=True)

        if self.logger is not None:
            self.logger.info(f"Saving the model after epoch {epoch}")

        # Use model-specific optimizations and post-processing for saving
        if self.save_weights_only:
            self.saver.save_weights(epoch_path / "weights.json")

        self.saver.save(epoch_path)
