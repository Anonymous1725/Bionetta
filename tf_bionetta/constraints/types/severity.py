"""
Class responsible for calculating the severity of the number
of constraints for a given layer and model overall. Used 
primarily for pretty display in the CLI, so it does not 
contain any major logic.
"""

from __future__ import annotations

from enum import IntEnum


class ConstraintSeverity(IntEnum):
    """
    Enum class for the severity of the number of constraints.
    """

    ZERO = 0
    CHEAP = 1
    MODERATE = 2
    EXPENSIVE = 3

    def as_str(self) -> str:
        """
        Returns the enum as a string
        """

        match self.value:
            case 0:
                return "zero"
            case 1:
                return "cheap"
            case 2:
                return "moderate"
            case 3:
                return "expensive"

    def rich_color(self) -> str:
        """
        Returns the color for the enum
        """

        match self.value:
            case 0:
                return "bold bright_black"
            case 1:
                return "green"
            case 2:
                return "yellow"
            case 3:
                return "bold red"


def severity_from_layer_constraints(constraints: int) -> ConstraintSeverity:
    """
    Returns the severity of the number of constraints for a given layer.

    Parameters:
        - constraints (int) - the number of constraints for the layer
    """

    CHEAP_LEVEL = 50000
    MODERATE_LEVEL = 150000

    if constraints == 0:
        return ConstraintSeverity.ZERO
    if constraints <= CHEAP_LEVEL:
        return ConstraintSeverity.CHEAP
    elif constraints <= MODERATE_LEVEL:
        return ConstraintSeverity.MODERATE
    else:
        return ConstraintSeverity.EXPENSIVE


def severity_from_model_constraints(constraints: int) -> ConstraintSeverity:
    """
    Returns the severity of the number of constraints for a given model.

    Parameters:
        - constraints (int) - the number of constraints for the model
    """

    CHEAP_LEVEL = 700000
    MODERATE_LEVEL = 1100000

    if constraints == 0:
        return ConstraintSeverity.ZERO
    if constraints <= CHEAP_LEVEL:
        return ConstraintSeverity.CHEAP
    elif constraints <= MODERATE_LEVEL:
        return ConstraintSeverity.MODERATE
    else:
        return ConstraintSeverity.EXPENSIVE
