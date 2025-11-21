from .backend_enums import ProvingBackend, WitnessGenerator, OptimizationLevel
from .target import TargetPlatform
import json
from pathlib import Path

from tf_bionetta.specs.backend_enums import Groth16, UltraGroth
from tf_bionetta.logging.pretty import Panel
from tf_bionetta.logging.verbose import VerboseMode
from tf_bionetta.logging.logger import create_logger, console
from logging import Logger


class Engine:
    """
    Class to represent all specifications for compiling your model
    """

    def __init__(
        self,
        proving_backend: ProvingBackend,
        target_platform: TargetPlatform,
        witness_backend: WitnessGenerator,
        optimization_level: OptimizationLevel,
        logger: Logger | None = None,
        verbose: VerboseMode = VerboseMode.INFO,
    ) -> None:

        self.proving_backend = proving_backend
        self.target_platform = target_platform
        self.witness_backend = witness_backend
        self.optimization_level = optimization_level


        self._verbose = verbose
        self._logger = logger
        if logger is None:
            self.logger = create_logger(self._verbose)

    def check_compatibility(self) -> None:
        """
        Checks if provided specifications are compatible.
        """
        issue = "Proving specifications do not match:"

        if self.target_platform == TargetPlatform.WEB:
            if isinstance(self.proving_backend, UltraGroth):
                console.print(Panel.fit(f"{issue} Ultra-Groth does not supports Web env yet.", title="Error", border_style="red"))
                raise NotImplementedError()
            
            elif isinstance(self.witness_backend, WitnessGenerator.CUSTOM):
                console.print(Panel.fit(f"{issue} Custom witness generation for Web is not supported yet.", title="Error", border_style="red"))
                raise NotImplementedError()


    def save_comp_specs(self, path: Path) -> None:
        """
        Saves compilation engine specifications to json by given path.
        """

        data = {
            "Proving backend": self.proving_backend.NAME,
            "Target platform": self.target_platform,
            "Witness backend": self.witness_backend,
            "Optimization level": self.optimization_level
        }

        if isinstance(self.proving_backend, UltraGroth):
            data["limb_size"] = self.proving_backend.limb_size
            data["precision_multiplicity"] = self.proving_backend.precision_multiplicity
        elif isinstance(self.proving_backend, Groth16):
            data["precision"] = self.proving_backend.precision
        else:
            raise ValueError("Unknown proving backend")


        with open(f"{path}/engine_specs.json", 'w') as f:
            json.dump(data, f, indent=4)


    @staticmethod
    def load_comp_specs(path: Path):

        with open(path, 'r') as f:
            data = json.load(f)

        prove_back = data["Proving backend"]
        match prove_back:
            case Groth16.NAME:
                proving_backend = ProvingBackend.GROTH16(data["precision"])
            case UltraGroth.NAME:
                proving_backend = ProvingBackend.ULTRAGROTH(data["limb_size"], data["precision_multiplicity"])
            case _:
                raise ValueError(f"Unknown proving backend: {data['Proving backend']}")

        engine = Engine(
            proving_backend=proving_backend,
            target_platform=data["Target platform"],
            witness_backend=data["Witness backend"],
            optimization_level=data["Optimization level"],
        )
        return engine
