"""
Package responsible for handling the hyperparameters of the training
process, which includes the model metadata and the model input/output
specification.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

from tf_bionetta.hyperparameters.meta import Metadata
from tf_bionetta.hyperparameters.model import ModelIOSpecification


class TrainingHyperparameters:
    """
    Class containing the hyperparameters of the training model in use.
    """

    META_KEYWORD: str = "meta"
    IO_KEYWORD: str = "io_specification"

    def __init__(self, params: Dict[str, Any]) -> None:
        """
        Creates the Training Hyperparameters with the list of custom
        parameters specified in the `params` dictionary.

        Args:
            params (Dict): The dictionary containing the hyperparameters.
        """

        # Initialize the dictionary
        self._dictionary = params

        # Interpret metadata properly if it is not a Metadata object
        metadata_present = TrainingHyperparameters.META_KEYWORD in self._dictionary
        self._dictionary[TrainingHyperparameters.META_KEYWORD] = (
            Metadata.default()
            if not metadata_present
            else Metadata(self._dictionary[TrainingHyperparameters.META_KEYWORD])
        )

        # Interpret the input/output specification properly if it is not a ModelIOSpecification object
        io_spec_present = TrainingHyperparameters.IO_KEYWORD in self._dictionary
        self._dictionary[TrainingHyperparameters.IO_KEYWORD] = (
            ModelIOSpecification.default()
            if not io_spec_present
            else ModelIOSpecification(
                self._dictionary[TrainingHyperparameters.IO_KEYWORD]
            )
        )

        # Initialize a list of parameters in use
        self._params_list = list(self._dictionary.keys())

    @staticmethod
    def from_json(path: Path) -> TrainingHyperparameters:
        """
        Loads the JSON file containing the hyperparameters.

        Args:
            path (Path): Path to the JSON file containing the `SimpleHyperparameters`.
        """

        with open(path, "r") as json_file:
            json_dictionary = json.loads(json_file.read())
            return TrainingHyperparameters(json_dictionary)

    def raw(self) -> Dict[str, Any]:
        """
        Returns the raw dictionary with the hyperparameters.
        """

        raw_dictionary = self._dictionary.copy()
        raw_dictionary[TrainingHyperparameters.META_KEYWORD] = self._dictionary[
            TrainingHyperparameters.META_KEYWORD
        ].raw()
        raw_dictionary[TrainingHyperparameters.IO_KEYWORD] = self._dictionary[
            TrainingHyperparameters.IO_KEYWORD
        ].raw()
        return raw_dictionary

    def save(self, path: Path) -> None:
        """
        Saves the ClassificationHyperparameters to the JSON file.

        Args:
            path (Path): The path to the JSON file.
        """

        if path.is_dir():
            meta = self._dictionary[TrainingHyperparameters.META_KEYWORD]
            path = (
                path
                / f"hyperparameters_{meta.name}_v{meta.version}.{meta.subversion}.json"
            )

        # Create folder if not exists
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as json_file:
            json_file.write(json.dumps(self.raw(), indent=4))

    def __getattr__(self, name: str) -> Any:
        """
        Returns the value of the attribute.

        Args:
            name (str): The name of the attribute.
        """

        if name not in self._params_list:
            raise AttributeError(f"Attribute {name} not found")

        if name in self._dictionary:
            return self._dictionary[name]

        default_hyperparams = TrainingHyperparameters.default()
        return default_hyperparams._dictionary[name]

    @staticmethod
    def default() -> TrainingHyperparameters:
        """
        Returns the default ClassificationHyperparameters.
        """

        return TrainingHyperparameters.from_dictionary(
            {
                TrainingHyperparameters.META_KEYWORD: Metadata.default(),
                TrainingHyperparameters.ModelIOSpecification: ModelIOSpecification.default(),
            }
        )
