"""
Package containing the specification of the model class, which
is needed to specify the input/output specification of the model in use.
"""

from __future__ import annotations
from typing import Dict, Any


class ModelIOSpecification:
    """
    Specification of the input/output of the model in use
    """

    _FIELDS = [
        "input_shape",
        "output_shape",
    ]

    def __init__(self, dictionary: Dict[str, Any]) -> None:
        """
        Initializes the model's input/output specification.

        Args:
            dictionary (Dict): The dictionary containing the information.
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
        if name not in ModelIOSpecification._FIELDS:
            raise AttributeError(f"Attribute {name} not found")

        return self._dictionary.get(
            name, ModelIOSpecification.default()._dictionary[name]
        )

    @staticmethod
    def default() -> ModelIOSpecification:
        """
        Returns the default metadata.
        """

        return ModelIOSpecification(
            {
                "input_shape": None,
                "output_shape": None,
            }
        )
