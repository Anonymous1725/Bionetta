"""
Package containing the specification of the metadata class, which
is needed to specify the name and version on the model in use.
"""

from __future__ import annotations
from typing import Dict, Any


class Metadata:
    """
    Metadata of the model in use
    """

    _FIELDS = [
        "name",
        "version",
        "subversion",
    ]

    def __init__(self, dictionary: Dict[str, Any]) -> None:
        """
        Initializes the metadata.

        Args:
            dictionary (Dict): The dictionary containing the metadata.
        """

        self._dictionary = dictionary

    def raw(self) -> Dict[str, Any]:
        """
        Returns the raw dictionary.
        """

        return self._dictionary

    def __getattr__(self, name: str) -> None:
        """
        Returns the value of the attribute.

        Args:
            name (str): The name of the attribute.
        """
        if name not in Metadata._FIELDS:
            raise AttributeError(f"Attribute {name} not found")

        return self._dictionary.get(name, Metadata.default()._dictionary[name])

    @staticmethod
    def default() -> Metadata:
        """
        Returns the default metadata.
        """

        return Metadata(
            {
                "name": "bionetta_model",
                "version": 0,
                "subversion": 1,
            }
        )
