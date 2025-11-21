"""
Enum for verbose modes in logging
"""

from enum import IntEnum


class VerboseMode(IntEnum):
    """
    Enum for logging verbosity
    """

    WARNING = 0
    INFO = 1
    DEBUG = 2

    def log_level(self) -> str:
        """
        Returns the string representation of the enum
        """
        return {0: "WARNING", 1: "INFO", 2: "DEBUG"}.get(self.value, "INFO")
