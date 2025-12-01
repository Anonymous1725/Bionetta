"""
Module for running the compilation of the model to have the circuit, Rust code,
and MobileDevKit code.
"""


from __future__ import annotations

# System imports
import os
import sys
import json
import stat
import shutil
import importlib.util
import logging
from types import ModuleType
from pathlib import Path
from typing import Dict, Callable, Any

# Third-party imports
import tensorflow as tf
import numpy as np
from rich.progress import(
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn
)
from jinja2 import Environment, FileSystemLoader

# Internal imports
from tf_bionetta.codegen.utils import(
    receive_constraints_num_for_r1cs,
    obtain_available_ram_in_the_system,
    log_step,
)
from tf_bionetta.logging.logger import create_logger, console
from tf_bionetta.logging.verbose import VerboseMode
from tf_bionetta.logging.pretty import(
    run_command, 
    print_success_message
)
from tf_bionetta.codegen.powersoftau import PowersOfTauLoader
from tf_bionetta.specs.backend_enums import ProvingBackend, WitnessGenerator, OptimizationLevel, Groth16, UltraGroth
from tf_bionetta.specs.target import TargetPlatform
from tf_bionetta.specs.engine import Engine
from tf_bionetta.specs.rust_generator_options import RustGeneratorOptions


class CircuitGenerator:
    """
    Class for compiling the circuit for the model.
    """
    
    # Template environment
    BASE_DIR = Path(__file__).resolve().parent
    TEMPLATE_ENV = Environment(loader=FileSystemLoader(BASE_DIR / 'templates'))
    
    # Templates retrieved from the templates directory
    HEADER_GROTH16  = BASE_DIR / 'templates' / 'header.h'
    C_WITNESS_CHECK = BASE_DIR / 'templates' / 'witness_check.c'


    RUST_XCFRAMEWORK_GENERATION_SCRIPT = TEMPLATE_ENV.get_template('generate_xcframework.sh')
    RUST_WITNESS_GEN_CONFIG = TEMPLATE_ENV.get_template('witness_generator_config.toml')

    WITNESSGEN_REPO = Path('bionetta-witness-generator')
    CIRCOM_REPO = Path('bionetta-circom')
    ULTRAGROTH_GROTH16_REPO = Path('ultragroth')
    ULTRAGROTH_SNARKJS_REPO = Path('ultragroth-snarkjs')


    @staticmethod
    def init_repos(repos_cache_path: Path):
        """
        Initializes variables for defining repositories and their storage path.
        
        Arguments:
            - repos_cache_path (Path) - the path to preserving repositories.
        """

        CircuitGenerator.REPOS_DIR: Path = repos_cache_path

        CircuitGenerator.WITNESSGEN_REPO = repos_cache_path / CircuitGenerator.WITNESSGEN_REPO
        CircuitGenerator.CIRCOM_REPO = repos_cache_path / CircuitGenerator.CIRCOM_REPO
        CircuitGenerator.ULTRAGROTH_GROTH16_REPO = repos_cache_path / CircuitGenerator.ULTRAGROTH_GROTH16_REPO
        CircuitGenerator.ULTRAGROTH_SNARKJS_REPO = repos_cache_path / CircuitGenerator.ULTRAGROTH_SNARKJS_REPO


    def __init__(
        self,
        repos_path: Path,
        model: tf.keras.models.Model,
        name: str | None = None,
        logger: logging.Logger | None = None,
        verbose: VerboseMode = VerboseMode.INFO,
    ) -> None:
        """
        Initializes the model saver.
        
        Arguments:
            - model (tf.keras.models.Model) - model to save
            - repos_cache_path (Path) - the path to preserving repositories.
            - name (str, optional) - name of the model. If None, the name is generated based on the mo````del's name.
            - logger (logging.Logger, optional) - logger to use for logging. If None, the default logger is used.
            - name (str, optional) - name of the model. If None, the name is generated based on the model's name.
            - verbose (int, VerboseMode, optional) - verbosity level. If None, the default verbosity level is used.
        """

        if not hasattr(CircuitGenerator, "REPOS_DIR"):
            CircuitGenerator.init_repos(repos_path)
        
        self._name = name if name is not None else model.name
        self._verbose = verbose
        self._logger = logger
        if self._logger is None:
            self._logger = create_logger(verbose)

        # Setup paths
        self._circom_dir = Path(f"{self._name}_circom")
        self._rust_dir = Path(f"{self._name}_rust")
        
        # Helper structures
        self._powers_of_tau = PowersOfTauLoader(logger=self._logger)


    def generate(
        self,
        output_path: Path,
        engine: Engine,
        architecture: Dict[str, Any],
        weights: Dict[str, np.ndarray],
        trusted_setup: Path | None = None,
        test_input: np.ndarray | None = None,
        rust_witness_generator_options: Dict[str, Any] = None,
    ) -> None:
        """
        Based on the provided architecture and weights, generates the circuit and Rust code.
        
        Args:
            - name (str): The name of the model.
            - architecture (Dict[str, Any]): The architecture of the model.
            - weights (Dict[str, np.ndarray]): The weights of the model.
            - load_repos (bool, optional): Whether to load the repositories with the code generators. Defaults to False.
            - output_path (Path, optional): The path to save the generated code. Defaults to './mobile'.
        """

        # Must be first
        ptau_dir = Path(__file__).resolve().parents[1] / Path('ptau')
        self.start_dir = Path(os.path.abspath(output_path))

        # Creating the output directory if it does not exist
        # and changing the working directory to it
        self.start_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(self.start_dir)

        # Now, we are conducting all the compilation steps
        steps: Dict[str, Callable] = {}

        steps.update({
            'Generating and compiling Circom circuits': lambda: self.compile_circom_circuit(
                architecture=architecture, 
                weights=weights,
                engine=engine,
            )
        })
        

        if engine.witness_backend == WitnessGenerator.CUSTOM:
            steps.update({
                # It must be before the generation of rust.
                'Create input file': lambda: self.generate_input(test_input=test_input),
                'Generating Rust witness generator code': lambda: self.generate_rust_code(
                    architecture=architecture,
                    engine=engine,
                    rust_witness_generator_options=rust_witness_generator_options
                ),
            })

            if engine.target_platform == TargetPlatform.MOBILE:
                steps.update({
                    'Generate header file': lambda: self.gen_c_and_header_files(engine),
                    # 'Exec C file': lambda: self._exec_c_file()  # NOTE: For testing
                })


            # After circuit compilation, we make sure that we have powers of tau prepared.
            # If it is not specified (which is typically the case), we 
            # download the powers of tau file as the first step.
            # NOTE: This method must be after rust code generation
            trusted_setup = None if trusted_setup is None else os.path.abspath(Path(trusted_setup))
            if trusted_setup is None or not trusted_setup.exists():
                ptau_kwargs = {
                    'target_directory': ptau_dir,
                    'constraints_number': 1  # Plug
                }
                trusted_setup = PowersOfTauLoader.form_ptau_file_path(**ptau_kwargs)
                steps.update({
                    'Downloading powers of tau': lambda progress: self._powers_of_tau.download(**ptau_kwargs, progress=progress)
                })
            trusted_setup = os.path.abspath(trusted_setup)

            if isinstance(engine.proving_backend, UltraGroth):
                steps.update({
                    'Generating zkey and vkey files for ultra groth':
                        lambda: self.generate_ultra_groth_vkey_and_zkey(trusted_setup=trusted_setup)
                })
            elif isinstance(engine.proving_backend, Groth16):
                steps.update({
                    'Generating zkey file': lambda: self.generate_zkey(trusted_setup=trusted_setup)
                })
            else:
                raise ValueError("Unknown proving backend")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Starting...", total=len(steps))
            for i, (description, command) in enumerate(steps.items()):
                task_description = f"Step {i+1}/{len(steps)}: {description}"
                progress.update(task, description=task_description)

                if description == 'Generating Rust witness generator code' and engine.optimization_level == OptimizationLevel.SKIP:
                    command()
                    return

                # If the step is downloading powers of tau, we need to pass the progress bar
                if description == 'Downloading powers of tau':
                    ptau_kwargs['constraints_number'] = self.mConstraints
                    trusted_setup = PowersOfTauLoader.form_ptau_file_path(**ptau_kwargs)
                    trusted_setup = os.path.abspath(trusted_setup)
                    command(progress)
                    progress.advance(task)
                    continue
                
                # Otherwise, we just call the command
                command()
                progress.advance(task)
            else:
                progress.update(task, description="[green]✔ All steps completed!")


    @log_step("Generating the Circom circuit")
    def compile_circom_circuit(
        self,
        architecture: Dict[str, Any],
        weights: Dict[str, np.ndarray],
        engine: Engine,
    ) -> None:
        """
        Generates the Circom circuits code. For that, we first clone 
        the witness generator repository and the Circom template circuits.
        
        Args:
            architecture (Dict[str, Any]): The architecture of the model.
            weights (Dict[str, np.ndarray]): The weights of the model.
        """

        # Prepare the directory for the Circom circuit
        os.makedirs(self._circom_dir, exist_ok=True)

        # We take the CircuitGenerator class from the cloned repository and 
        # generate the code based on the architecture and weights
        self._logger.debug('Loading generator first...')
        module = self._get_py_script("main", str(self.CIRCOM_REPO) + "/generator/main.py")
        self._logger.debug('Generator loaded successfully')
        
        CircuitGenerator = getattr(module, "CircuitGenerator")
        PROVE_BACKEND = getattr(module, "ProvingBackend")

        generator = CircuitGenerator(
            architecture=architecture,
            weights=weights,
            circom_folder=self.CIRCOM_REPO,
            proving_backend=PROVE_BACKEND.create(engine.proving_backend),
            verbose=VerboseMode.DEBUG.value
        )
        generator.build(target_folder=self._circom_dir)


        self._logger.info('Compiling the Circom circuit... That is the longest process during the compilation, so be patient. You might even bring a coffee or two :coffee:')

        args = ["circom", f"{self._circom_dir}/{self._name}.circom", "--r1cs", "--sym", f"--O{engine.optimization_level}", "-o", 
            os.path.abspath(self._circom_dir)]
        
        if isinstance(engine.proving_backend, Groth16) and \
            (engine.witness_backend == WitnessGenerator.DEFAULT or engine.target_platform == TargetPlatform.WEB):
            
            args.append("--wasm")

        run_command(args)
        print_success_message('Circuit compiled successfully!')
    
    
    @log_step("Generating the Rust witness generation code")
    def generate_rust_code(
        self,
        architecture: Dict[str, Any],
        engine: Engine,
        rust_witness_generator_options: Dict[str, Any] = None,
    ) -> None:
        """
        Generates the Rust witness generation code
        """
        
        # First program run
        # Now, we generate the Rust code
        if not os.path.isdir(self._rust_dir):
            # Create the Rust project
            run_command(["cargo", "new", str(self._rust_dir)])
        
        # Cleaning up the lib.rs file to prevent bugs during rust compilation (for example changing proving backend)
        open(f"{self._rust_dir}/src/lib.rs", "w").close()

        # Insert the witness generator configuration
        with open(f"{self._rust_dir}/Cargo.toml", "w") as file:
            witness_gen_config = self.RUST_WITNESS_GEN_CONFIG.render(
                model_name=self._rust_dir,
                witness_gen_path=str(self.WITNESSGEN_REPO),
            )
            file.write(witness_gen_config)

        for dir in ["architecture", "weights"]:
            dest_path = f"{self._rust_dir}/src/{dir}"
            os.makedirs(dest_path, exist_ok=True)

            with open(f"{dest_path}/mod.rs", "w") as file:
                file.write(f"pub(crate) mod {self._name};")
        
        # Copy generated weights.rs file to proper directory
        shutil.copy(f"{self._circom_dir}/{self._name}.rs", 
                    f"{self._rust_dir}/src/weights/{self._name}.rs")

        rust_project_path = os.path.abspath(self._rust_dir)
        rust_gen = self._get_py_script("main", str(self.WITNESSGEN_REPO) + "/generator/main.py")
        PROVE_BACKEND = getattr(rust_gen, "ProvingBackend")

        # Must be first call
        rust_gen.generate(
            nn_architecture=architecture,
            rust_project_path=rust_project_path,
            proving_backend=PROVE_BACKEND.create(engine.proving_backend),
        )

        # The first launch of the program for preliminary calcuation of sym_file values
        main_file = f'{rust_project_path}/src/main.rs'
        rust_indexes = os.path.abspath(f"{self._rust_dir}/src/weights/indexes.bin")
        rust_code = rust_gen.init_main_precompute(
            sym_file=f"{os.path.abspath(self._circom_dir)}/{self._name}.sym",
            rust_indexes=rust_indexes,
        )
        rust_gen.write_file(rust_code, main_file)
        run_command(["cargo", "run", "--manifest-path", f"{rust_project_path}/Cargo.toml"])

        # Second program run
        with open(f"{self._rust_dir}/Cargo.toml", "a") as file:
            file.write("""\n
[[bin]]
name = "gen_wtns"
path = "src/lib.rs"

[lib]
crate-type = ["rlib", "staticlib", "cdylib"]            
""")

        r1cs_file = f"{os.path.abspath(self._circom_dir)}/{self._name}.r1cs"
        rust_code = rust_gen.generate_rust_lib(
            r1cs_file=r1cs_file,
            rust_indexes=rust_indexes,
        )
        rust_gen.write_file(rust_code, f'{rust_project_path}/src/lib.rs')

        if engine.target_platform == TargetPlatform.MOBILE and rust_witness_generator_options is not None:
            if RustGeneratorOptions.RUST_WITNESS_GENERATOR_XCFRAMEWORK_SCRIPTS_KEY in rust_witness_generator_options:
                framework_name = ''.join([word.capitalize() for word in str(self._rust_dir).split('_')])

                generation_script = self.RUST_XCFRAMEWORK_GENERATION_SCRIPT.render(
                    model_name=self._rust_dir,
                    framework_name=framework_name,
                )

                os.makedirs(f"{self._rust_dir}/scripts", exist_ok=True)

                with open(f"{self._rust_dir}/scripts/generate_xcframework.sh", "w") as file:
                    file.write(generation_script)

                script_permissions = os.stat(f"{self._rust_dir}/scripts/generate_xcframework.sh").st_mode
                os.chmod(f"{self._rust_dir}/scripts/generate_xcframework.sh", script_permissions | stat.S_IEXEC)

        # Create wtns or uwtns file
        run_command(["cargo", "run", "--release", "--bin", "gen_wtns", "--manifest-path", f"{rust_project_path}/Cargo.toml"])

        if engine.target_platform == TargetPlatform.MOBILE:
            run_command(["cargo", "build", "--release", "--lib", "--manifest-path", f"{rust_project_path}/Cargo.toml"])
        
        # Delete unnecessary code
        target_dir = Path(f"{rust_project_path}/target/release")
        libs = list(target_dir.glob('lib*'))

        for lib in libs:
            lib.rename(Path('.') / lib.name)
        shutil.rmtree(f"{rust_project_path}/target")

        # For the exact number of constraints
        self.mConstraints = rust_gen.get_number_of_constraints(r1cs_file)
        print_success_message('Witness generated successfully!')


    @log_step("Generate the C header file")
    def gen_c_and_header_files(self, engine: Engine) -> None:
        if isinstance(engine.proving_backend, Groth16):
            struct_name = "WtnsBytes"
            function_name = f'calc_wtns_{self._name}'
            free_function = f'wtns_free'
        elif isinstance(engine.proving_backend, UltraGroth):
            struct_name = "UwtnsBytes"
            function_name = f'calc_uwtns_{self._name}'
            free_function = f'uwtns_free'

        filename = str(CircuitGenerator.HEADER_GROTH16).split('/')[-1]
        file_content = self.TEMPLATE_ENV.get_template('header.h').render(
            struct_name=struct_name,
            function_name=function_name,
            free_function=free_function,
        )
        with open(filename, 'w') as file:
            file.write(file_content)

        # NOTE: Need for testing!
        # if isinstance(engine.proving_backend, Groth16):
        #     generated_filename = f'{self._name}.wtns'
        # elif isinstance(engine.proving_backend, UltraGroth):
        #     generated_filename = f'{self._name}.uwtns'
        # 
        # filename = str(CircuitGenerator.C_WITNESS_CHECK).split('/')[-1]
        # file_content = self.TEMPLATE_ENV.get_template('witness_check.c').render(
        #     struct_name=struct_name,
        #     function_name=function_name,
        #     filename=generated_filename,
        # )
        # with open(filename, 'w') as file:
        #     file.write(file_content)
    

    # NOTE: For testing
    @log_step("Executing the C file")
    def _exec_c_file(self):
        """
        Executes the C file.
        """
        
        filename = str(CircuitGenerator.C_WITNESS_CHECK).split('/')[-1]
        run_command([
            'gcc', filename, '-o', 'executable',
            f'lib{self._name}_rust.a', '-lm',
        ])
        run_command(['./executable', 'input.bin'])


    @log_step("Generating the zkey file")
    def generate_zkey(self, trusted_setup: Path) -> None:
        """
        Generates the zkey file for the circuit.
        
        Args:
            name (str): The name of the circuit.
            circom_dir (Path): The path to the Circom directory.
            trusted_setup (Path): The path to the trusted setup file.
        """
        
        os.chdir(self._circom_dir)
        zkey = Path('zkey')

        if os.path.exists(zkey):
            shutil.rmtree(zkey)
        os.makedirs(zkey)

        # 1. Groth16 setup
        run_command([
            "snarkjs", "groth16", "setup",
            f"{self._name}.r1cs",
            trusted_setup,
            f"{self._name}_0000.zkey",
            "-v"
        ])
        
        # 2. Generate random contribution
        random_entropy = run_command(['xxd', '-l', '128', '-p', '/dev/urandom']).stdout.strip()
        run_command([
            'snarkjs', 'zkey', 'contribute',
            f'{self._name}_0000.zkey',
            f'{self._name}.zkey',
            '--name=Someone',
            '-v'
            ],
            input=random_entropy,
        )

        # 3. Export verification key
        run_command([
            'snarkjs', 'zkey', 'export', 'verificationkey',
            f'{self._name}.zkey',
            'verification_key.json'
        ])

        # 4. Export Solidity verifier
        run_command([
            'snarkjs', 'zkey', 'export', 'solidityverifier',
            f'{self._name}.zkey',
            'verifier.sol'
        ])

        # 5. Clean up redundant files
        shutil.rmtree(zkey)
        os.remove(f'{self._name}_0000.zkey')
        os.chdir('..')


    @log_step("Generating zkey and vkey files for ultra groth")
    def generate_ultra_groth_vkey_and_zkey(self, trusted_setup: Path) -> None:
        abs_circom_dir = os.path.abspath(self._circom_dir)
        target_file = f"{abs_circom_dir}/{self._name}"
        curr_dir = os.getcwd()
        os.chdir(self.ULTRAGROTH_SNARKJS_REPO)


        run_command(["npm", "install", "--save-dev", "snarkjs@latest", "mocha", "cross-env"])
        
        arguments = {
            "NODE_OPTIONS": f"--max-old-space-size={obtain_available_ram_in_the_system()}",
            "R1CS": f"{target_file}_injected.r1cs",
            "INDEXES": f"{target_file}_indexes.json",
            "PTAU": trusted_setup,
            "OUT_DIR": abs_circom_dir,
            "OUT_NAME": self._name,
        }

        env = os.environ.copy()
        env.update(arguments)
        run_command(["npm", "run", "test:ultragroth"], env=env)
        os.chdir(curr_dir)
    
    
    @log_step("Generate a binary input file")
    def generate_input(
        self,
        test_input: Dict[str, Any],
    ) -> None:
        """
        Creates the test input.
        
        Args:
            test_input (Dict[str, Any]): The test input to the model.
        """

        # Write down the input file
        print('inside generate_input:', os.getcwd())
        input_file = Path('input.bin')
        with open(input_file, 'wb') as file:
            json_data = json.dumps(test_input)
            file.write(json_data.encode('utf-8'))


    def _get_py_script(self, module: str, script: Path) -> ModuleType:
        """
        Given a path to a python script, imports the module and returns it.

        Args:
            module (str): The name of the module to import.
            script (Path): The path to the script.
        """
        
        path_without_script_name = script.rsplit("/", 1)[0]
        sys.path.append(os.path.abspath(path_without_script_name))

        spec = importlib.util.spec_from_file_location(module, script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        return module
