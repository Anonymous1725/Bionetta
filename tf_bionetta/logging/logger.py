"""
A package for logging-related utilities.
"""

import logging
import absl.logging
from rich.logging import RichHandler
from rich.console import Console

from tf_bionetta.logging.verbose import VerboseMode

# Create a Console (optional — RichHandler can create one internally)
console = Console()


def create_logger(mode: VerboseMode = VerboseMode.INFO) -> logging.Logger:
    """
    Configures the logging, and returns the logger instance

    ### Args:
    - level (str, optional): Logging level. Defaults to 'INFO'.

    ### Returns:
        logging.Logger: Instance of the logger
    """

    logging.basicConfig(
        level=mode.log_level(), 
        format="%(message)s", 
        datefmt="[%X]", 
        handlers=[RichHandler(console=console, 
                              markup=True, 
                              rich_tracebacks=True,
                              show_time=False)]
    )
    absl.logging.set_verbosity(absl.logging.ERROR)  # Disabling the TensorFlow warnings

    return logging.getLogger("rich")


class MaybeLogger:
    """
    Wraps any nullable object and calls its methods if inner object is not null.
    """

    def __init__(self, logger: logging.Logger | None) -> None:
        self._logger = logger

    def __getattr__(self, name):
        if self._logger is None:
            return lambda *args, **kwargs: None

        return getattr(self._logger, name)

    def __repr__(self):
        return f"Maybe({self._logger})"
