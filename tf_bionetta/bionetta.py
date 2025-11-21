"""
Package for handling the model wrapping written over Bionetta Framework.
Currently supports only TensorFlow Keras models.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import tensorflow as tf

from tf_bionetta.proving.prover import Prover
from tf_bionetta.proving.verifier import Verifier
from tf_bionetta.proving.prover import Prover
from tf_bionetta.proving.verifier import Verifier
from tf_bionetta.constraints import ModelConstraintsCalculator
from tf_bionetta.hyperparameters import ModelIOSpecification
from tf_bionetta.save import BionettaModelSaver
from tf_bionetta.layers.custom_objects import get_custom_objects
from tf_bionetta.logging.logger import VerboseMode, create_logger
from tf_bionetta.codegen.generator import CircuitGenerator
from tf_bionetta.specs.backend_enums import(
    ProvingBackend,
    WitnessGenerator,
    OptimizationLevel,
    Groth16,
    UltraGroth
)
from tf_bionetta.specs.target import TargetPlatform
from tf_bionetta.specs.engine import Engine



class BionettaModel:
    """
    Class that wraps the keras model and provides additional functionalities
    such as constraint calculation, saving, and compiling to Circom and
    Rust witness generator
    """
    
    # Default precision for two outputs to be the same
    EQUAL_THRESHOLD: float = 1e-2
    
    def __init__(
        self,
        model: tf.keras.layers.Model,
        name: str = None,
        verbose: int | VerboseMode | None = None,
        ignore_errors: bool = False,
    ):
        """
        Initialize the model wrapper.

        Args:
            - model (`tf.keras.models.Model`): A Keras model to be wrapped.
            - name (`str`, optional): The name of the model. Defaults to `None`.
            - verbose (`int`, `VerboseMode`, optional): The verbosity level of the logger. Defaults to `None`.
            `0` is for no logging, `1` is for INFO, `2` is for DEBUG.
            - ignore_errors (`bool`, optional): Whether to ignore the errors (such as unsupported layers)
            and continue with the model wrapping. Defaults to `False`. Note that is errors are ignored,
            the resultant circuit most likely will not compile.
            - precision (`int`, optional): Integer, representing the degree of accuracy of arithmetization
            to be used for further working over circuits
        """

        assert isinstance(model, tf.keras.models.Model), "Input must be a Keras Model."

        self.repos_path = os.path.abspath(os.path.dirname(__file__)) / Path("repos")

        # Setup the logger
        if verbose is None:
            verbose = VerboseMode.WARNING
        elif isinstance(verbose, int):
            verbose = VerboseMode(verbose)
        self._verbose = verbose
        self._logger = create_logger(verbose)
        
        # Save the model and the name
        self.name = name if name is not None else model.name
        self.base_model = model
        self._io_specification = ModelIOSpecification(
            { "input_shape": model.input_shape, "output_shape": model.output_shape }
        )

        # Now, initialize the auxiliary classes
        self._saver = BionettaModelSaver(
            model,
            name=self.name,
            logger=self._logger,
            ignore_errors=ignore_errors,
        )
        self._codegen = CircuitGenerator(
            self.repos_path,
            model,
            name=self.name,
            logger=self._logger,
            verbose=self._verbose
        )

        self._engine = None
        self._compiled = False
        self._compiled_path: Path | None = None


    def constraints_summary(self, proving_backend: ProvingBackend) -> None:
        """
        Prints the number of constraints for each layer in the model.
        """

        ModelConstraintsCalculator(
            self.base_model, 
            backend=proving_backend,
            name=self.name,
            logger=self._logger,
        ).print_constraints_summary()


    def save_circuit_weights(self, path: Path, compress: bool = True) -> None:
        """
        Saves the model to the specified path. The model is saved with all the custom objects
        to avoid any issues with the custom layers.

        Args:
            - path (`Path`): The path to save the model.
            - compress (`bool`, optional): Whether to compress the json files or not. Defaults to `True`.
        """

        self._saver.save(path, compress=compress)


    def compile_circuits(
        self,
        path: Path,
        proving_backend: ProvingBackend,
        target_platform: TargetPlatform = TargetPlatform.DESKTOP,
        witness_backend: WitnessGenerator = WitnessGenerator.CUSTOM,
        optimization_level: OptimizationLevel = OptimizationLevel.O1,
        test_input: np.ndarray | tf.Tensor = None,
        powers_of_tau_path: Path = None,
        save_weights: bool = True,
        rust_witness_generator_options: Dict[str, Any] = None,
    ) -> None:
        """
        Compiles all the source code for the model:
            - Circom code for the circuit
            - Rust code for the witness generator
            - MobileDevKit for the mobile application integration

        Args:
            - path (`Path`): The path to save the compiled model.
            - test_input (`np.ndarray`, `tf.Tensor`, optional): Test input to
              the model to check its correctness. Defaults to `None`: in such 
              case, the model is not tested.
            - powers_of_tau_path (`Path`, optional): Path to the powers of tau file.
              If not specified, it will be downloaded automatically.
              Defaults to `None`. 
            - save_weights (`bool`, optional): Whether to save the weights of the model
                in the circuit. Defaults to `True`. If `True`, the weights will be saved
                in the circuit and the witness generator.
            - proving_backend (`ProvingBackend`): specifies proving algorithm. 
                Defaults to `GROTH16`
            - target_platform (`TargetPlatform`): specifies target platform. 
                Defaults to `DESKTOP`
            - witness_backend (`WitnessGenerator`): specifies which algorithm use to generate witness (i.e. WASM if `ProvingBackend` is `GROTH16`). 
                Defaults to `CUSTOM`.
            - optimization_level (`OptimizationLevel`): specifies which optimization level use to compile Circom's circuits (increase in comp time and decrease in constraint number starting from O0).
                Use `SKIP` if willing to skip compilation of circuits at all (just code generation).
                Defaults to `O1`.
        """

        self._engine = Engine(
            proving_backend=proving_backend,
            target_platform=target_platform,
            witness_backend=witness_backend,
            optimization_level=optimization_level
        )
        self._engine.check_compatibility()
        
        start_dir = os.getcwd()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self._engine.save_comp_specs(path)
        circuit_input = self.generate_circuit_input(test_input)

        # 1. First, form all the necessary dictionaries and save them, if needed
        if save_weights:
            self._logger.info("Forming circuit specification...")
            circuit_specification = self._saver.save_circuit_specification(path / 'circuit.json')
            
            self._logger.info("Forming weights specification...")
            model_weights = self._saver.save_weights(path / 'weights.json', compress=True)
            
            # Also, save the test input
            if test_input is not None:
                self._logger.info("Saving test input...")
                with open(path / 'test_input.json', 'w') as f:
                    json.dump(circuit_input, f, indent=4)
        else:
            self._logger.info("Forming circuit specification...")
            circuit_specification = self._saver.form_circuit_specification()

            self._logger.info("Forming weights specification...")
            model_weights = self._saver.form_weights_dictionary()
            
        # 2. Running the code generator
        try:
            self._codegen.generate(
                output_path=path,
                engine=self._engine,
                architecture=circuit_specification,
                weights=model_weights,
                trusted_setup=powers_of_tau_path,
                test_input=circuit_input,
                rust_witness_generator_options=rust_witness_generator_options,
            )
        except Exception as e:
            self._logger.error(f"Unexpected error during the compilation of the model: {e}", exc_info=True)
        
        os.chdir(start_dir)
        self._compiled_path = path
        self._compiled = True
        

    def set_compiled_path(self, path: Path):
        self._compiled_path = Path(path)


    def generate_circuit_input(
        self,
        input: np.ndarray | tf.Tensor,
    ) -> Dict[str, List[int] | int]:
        """
        Based on the provided input, generates the json input file to the circuit
        """

        if isinstance(self._engine.proving_backend, Groth16):
            precision = self._engine.proving_backend.precision
            precision_multiplicity = 1
        elif isinstance(self._engine.proving_backend, UltraGroth):
            precision = self._engine.proving_backend.limb_size
            precision_multiplicity = self._engine.proving_backend.precision_multiplicity
        else:
            raise ValueError("Unknown proving backend")


        assert len(input.shape) in [3, 4], f'Expected the input shape of 3 or 4, got: {input.shape}'
        
        # If the provided input is a batch with a single image, we 
        # need to exteract the image from the batch
        if len(input.shape) == 3:
            input = tf.expand_dims(input, axis=0)
            self._logger.warning('Input is 3D, adding batch dimension. If this is not intended, please check the input shape.')
        else:
            assert input.shape[0] == 1, f'Expected the batch size of 1, got: {input.shape}'
        
        assert len(input.shape) == 4, f'Expected the full image, got: {input.shape}'

        # Get the output shape
        output_shape = self.base_model.output_shape
        assert len(output_shape) == 2, f'Expecting the flat output, got: {output_shape}'
        _, output_size = output_shape
        
        # Finding the output
        output = self.base_model(input)
        assert len(output.shape) == 2, f'Expecting the flat output, got: {output.shape}'
        assert output.shape[0] == 1, f'Expecting the batch size of 1, got: {output.shape}'
        assert output.shape[1] == output_size, f'Expecting the output size of {output_size}, got: {output.shape}'
        # Remove the batch dimension
        output = tf.squeeze(output, axis=0)
        
        # Now, we need to arithmetize the input and output
        arithmetized_input = self._form_image_circuit_input(input, precision, precision_multiplicity)
        arithmetized_output = self._saver.arithmetize_tensor(output, precision, precision_multiplicity)
        
        MOCKED_ADDRESS = '1' # Mocked address for the circuit (TODO(@ZamDimon): to be replaced with the actual address)
        MOCKED_NONCE = '1' # Mocked nonce for the circuit (TODO(@ZamDimon): to be replaced with the actual nonce)

        circuit_input = {
            'address': MOCKED_ADDRESS,
            'threshold': self._saver.arithmetize_tensor(
                BionettaModel.EQUAL_THRESHOLD**2,
                precision,
                precision_multiplicity*2
            ),
            'nonce': MOCKED_NONCE,
            'features': arithmetized_output,
            'image': arithmetized_input,
        }
        return circuit_input


    def _form_image_circuit_input(
        self,
        image: np.ndarray | tf.Tensor,
        precision: int,
        precision_multiplicity: int,
    ) -> Dict[str, List[int] | int]:
        """
        Forms the circuit input for the image. The input is arithmetized and
        converted to a list of integers.
        """

        assert len(image.shape) == 4, f"Expected the image shape of 4, got: {image.shape}"
        assert image.shape[0] == 1, f"Expected the batch size of 1, got: {image.shape}"
        
        # Now, I need to convert the shape from (1,W,H,C) to (C,W,H)
        # and then arithmetize the input
        image = tf.transpose(image[0], (2, 0, 1))
        return self._saver.arithmetize_tensor(image, precision, precision_multiplicity)


    @classmethod
    def load_from_keras(
        cls,
        path: Path,
        name: str = None,
        verbose: int | VerboseMode | None = None,
        *args, **kwargs
    ) -> BionettaModel:
        """
        Loads the model from the specified path. Includes all the custom objects
        to avoid any issues with the custom layers.
        """

        model = tf.keras.models.load_model(
            Path(path), custom_objects=get_custom_objects(), compile=False
        )
        return BionettaModel(model, name=name, verbose=verbose, *args, **kwargs)
    
    
    @classmethod
    def load_from_compiled_folder(
        cls, 
        path: Path,
        name: str = None,
        verbose: int | VerboseMode | None = None,
    ) -> BionettaModel:
        """
        Loads the Bionetta model class from the compiled folder.
        """
        
        # TODO(@Sdoba16): We do not perform verification 
        # of whether the model was correctly compiled
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".keras"):
                    keras_file = os.path.join(root, file)

        model = BionettaModel.load_from_keras(Path(keras_file), name=name, verbose=verbose)
        model._compiled_path = path
        model._compiled = True
        
        engine = Engine.load_comp_specs(f"{path}/engine_specs.json")
        model._engine = engine
        
        if os.path.isdir(f"{path}/ultragroth"):
            model.proving_backend = ProvingBackend.ULTRAGROTH
            
        return model


    def __getattr__(self, name):
        # We inherit all the attributes from the base model
        return getattr(self.base_model, name)


    def create_input_bin(
        self, 
        input: np.ndarray | tf.Tensor,
    ) -> None:
        """
        Creates the input binary file for the circuit.
        
        Args:
            - input (`np.ndarray`, `tf.Tensor`, optional): The input to the model.
        """
        
        input_file = Path(self._compiled_path) / f'{self.name}_rust/input.bin'
        with open(input_file, 'wb') as file:
            json_data = json.dumps(self.generate_circuit_input(input))
            file.write(json_data.encode('utf-8'))
    
    
    def prove(self,
        input: np.ndarray | tf.Tensor, 
        target_dir: str,
    ) -> Dict:
        """
        Based on the provided input, generates the proof for the model.
        Returns the proof as a dictionary.
        
        Args:
            - input (`np.ndarray`, `tf.Tensor`): The input to the model.
            - target_dir (`str`): The directory to save the proof.
            
        Outputs:
            - `Dict`: The proof in the dictionary format.
        """
        
        if not self._compiled:
            raise RuntimeError("Trying to prove uncompiled model")
        
        self._logger.debug(f'Using the following backend: {self._engine.proving_backend}')

        self.create_input_bin(input=input)
        prover = Prover(
            repos_dir=self.repos_path,
            model_dir=self._compiled_path,
            model_name=self.name,
            engine=self._engine
        )
        return prover.full_prove(target_dir=target_dir)


    def verify(self, proof_dir: str) -> bool:
        if not self._compiled:
            raise RuntimeError("Trying to verify uncompiled model")

        model_dir = self._compiled_path

        verifier = Verifier(
            repos_dir=self.repos_path,
            model_dir=model_dir,
            model_name=self.name,
            engine=self._engine,
            logger=self._logger,
        )
        return verifier.verify(proof_dir=proof_dir)