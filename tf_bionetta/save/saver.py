"""
File responsible for saving the model after the
training has been completed (or during the training
while saving the concrete epoch).
"""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import tensorflow as tf

from tf_bionetta.logging.logger import MaybeLogger
from tf_bionetta.save.layers import to_saveable_layer, is_uninterpretable_layer
from tf_bionetta.save.layers.custom.finalize.distance import SaveableVerifyDist
from tf_bionetta.save.quantization import ModelQuantizer
from tf_bionetta.utils import unpack_model_layers


class BionettaModelSaver:
    """
    Class responsible for saving the neural network in the format that
    can be used for further Circom code generation and Rust witness generation.
    """

    # Default precision multiplicity: the degree of precision to 
    # be used by default for arithmetization of the tensors


    def __init__(
        self,
        model: tf.keras.models.Model,
        name: str = None,
        logger: logging.Logger | None = None,
        ignore_errors: bool = False,
    ) -> None:
        """
        Initializes the model saver.

        Arguments:
            - model (tf.keras.models.Model) - model to save
            - name (str, optional) - name of the model. If None, the model's name is used.
            - logger (logging.Logger, optional) - logger to use for logging. If None, the default logger is used.
            - ignore_errors (bool, optional) - whether to ignore errors during the saving process. If True, the errors are ignored.
        """

        self.model = model
        self.name = name if name is not None else model.name
        self._logger = MaybeLogger(logger)
        self._ignore_errors = ignore_errors
        self._quantizer = ModelQuantizer(model)
        self._unpack_layers = unpack_model_layers(self.model)


    @staticmethod
    def _form_input_shape(input_shape: Tuple) -> List[int]:
        """
        Forms the input shape to be used for the model.
        """

        POSSIBLE_CHANNELS_NUMBER = [1, 2, 3, 4]
        
        if len(input_shape) == 2:
            return (1, *input_shape)
        
        channel_is_last = input_shape[-1] in POSSIBLE_CHANNELS_NUMBER
        if channel_is_last:
            # Put the last channel to the first position
            return (input_shape[-1], *input_shape[:-1])
        
        # Otherwise, return the input shape as is
        return list(input_shape)


    def form_circuit_specification(self) -> Dict[str, np.ndarray]:
        """
        Forms the circuit parameters needed for further Circom code generation
        and Rust witness generation in the form of dictionary.
        """
        
        # Remove the None batch size if it is present
        input_shape = self.model.input_shape
        if input_shape[0] is None:
            input_shape = input_shape[1:]

        # If the image is grayscale, add the channel dimension
        if len(input_shape) == 2:
            input_shape = (1, *input_shape)
        
        # Assert that the input shape is an image
        assert len(input_shape) == 3, f"Input shape must be an image, got: {input_shape}"

        circuit_params: Dict[str, np.ndarray] = {
            'name': self.name,
            'input_shape': BionettaModelSaver._form_input_shape(input_shape),
            'output_shape': self.model.output_shape[1:],
            "layers": [],
        }
        
        for (previous_layer, layer) in zip([None]+self._unpack_layers, self._unpack_layers):
            # First, check whether we need to process the layer at all
            if is_uninterpretable_layer(layer):
                self._logger.info(f"Skip layer {layer.name}: layer needs no further interpretation")
                continue

            # Next, try converting to the saveable layer
            try:
                saveable_layer = to_saveable_layer(layer, previous_layer=previous_layer)
            except ValueError:
                self._logger.error(f"Layer {layer.name} is not supported")
                if not self._ignore_errors:
                    raise ValueError(f"Layer {layer.name} is not supported")

                continue

            # If the conversion is successful, we can save the layer to the dictionary
            layer_specification = saveable_layer.to_dictionary()
            if layer_specification is not None:
                circuit_params["layers"].append(layer_specification)
                self._logger.debug(f"Layer {layer.name} has been successfully saved")

        # Get the output size
        output_size = self.model.output_shape[1:]
        # Assert that it is a vector
        assert len(output_size) == 1, "Output size must be a vector"

        # Save the verify dist layer (TODO: make it more general)
        distance_layer_dict = SaveableVerifyDist(None).to_dictionary(output_size[0])
        circuit_params["layers"].append(distance_layer_dict)

        return circuit_params


    def save_circuit_specification(self, path: Path) -> Dict[str, np.ndarray]:
        """
        Saves the circuit parameters needed for further Circom code generation
        and Rust witness generation.

        Arguments:
            - path (Path) - path to save the parameters
            
        Returns:
            - circuit_params (Dict[str, np.ndarray]) - dictionary with the saved parameters
        """

        if not isinstance(path, Path):
            path = Path(path)  # Convert to Path object

        # Form circuit params from the inner implementation
        circuit_params: Dict[str, np.ndarray] = self.form_circuit_specification()

        # Save the formed circuit parameters
        self._logger.info(f"Saving circuit parameters to {path}...")
        with open(path, "w") as f:
            json.dump(circuit_params, f, indent=4)
            
        return circuit_params


    def form_weights_dictionary(self) -> Dict[str, np.ndarray]:
        """
        Forms the model's weights in the form of dictionary.
        """

        weights_dict: Dict[str, np.ndarray] = {}

        for layer in self._unpack_layers:
            # First, check whether we need to process the layer at all
            if is_uninterpretable_layer(layer):
                self._logger.info(f"Skip layer {layer.name}: layer needs no further interpretation")
                continue

            # Next, try converting to the saveable layer
            try:
                saveable_layer = to_saveable_layer(layer)
            except ValueError:
                self._logger.error(f"Layer {layer.name} is not supported")
                if not self._ignore_errors:
                    raise ValueError(f"Layer {layer.name} is not supported")

            layer_weights = saveable_layer.to_weights()
            if layer_weights is not None:
                weights_dict[layer.name] = layer_weights
                self._logger.debug(f"Weights for layer {layer.name} have been successfully saved")

        return weights_dict


    def save_weights(self, path: Path, compress: bool = True) -> Dict[str, np.ndarray]:
        """
        Saves the model's weights

        Arguments:
            - path (Path) - path to save the weights
            - compress (bool, optional) - whether to compress the json files or not. Defaults to True
            
        Returns:
            - weights_dict (Dict[str, np.ndarray]) - dictionary with the saved weights
        """

        weights_dict: Dict[str, tf.Tensor] = self.form_weights_dictionary()

        # Saving raw (Keras) weights first
        self._logger.info(f'Saving weights to {path}...')
        self.model.save(path.parent / f'{self.name}.keras')
            
        # Now, saving the dictionary with circuit-formatted weights
        with open(path, "w") as f:
            # Depending on the compress flag, we can either compress the json file or not
            if compress:
                json.dump(weights_dict, f, separators=(",", ":"))
            else:
                json.dump(weights_dict, f, indent=4)
        
        return weights_dict


    def save(self, path: Path, compress: bool = True) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Saves the model to the specified path.

        Arguments:
            - path (Path) - path to save the model
            - compress (bool, optional) - whether to compress the json files or not. Defaults to True
        """

        if not isinstance(path, Path):
            path = Path(path)

        # Creating the folder if it does not already exist
        path.mkdir(parents=True, exist_ok=True)
        assert path.is_dir(), "Path must be a directory"

        # Saving circuit specification
        self._logger.info("Saving circuit specification first...")
        self.save_circuit_specification(path / "architecture_specification.json")

        # Saving weights
        self._logger.info("Saving model weights...")
        self.save_weights(path / "weights.json", compress=compress)

        # Saving the quantized model
        self._logger.info("Applying the quantization on the trained model...")
        self._quantizer.save(path)

        self._logger.info(f'Model has been successfully saved to {path}')


    def arithmetize_tensor(
        self, 
        x: np.ndarray | tf.Tensor,
        precision: int,
        precision_multiplicity: int
    ) -> List[List[str]]:
        """
        Based on the provided tensor, converts it to the Fp elements for further 
        inputting into the circuits.

        Args:
            x - tensor to be converted.
            precision_multiplicity - the degree of precision to be used by default for arithmetization of the tensors.
        """

        # np.floor return float numbers, so we need convert them to int, then to str (need str nums for Rust)
        x = x * (2 ** (precision_multiplicity * precision))
        x = np.floor(x).astype(int).astype(str)
        return x.tolist()
