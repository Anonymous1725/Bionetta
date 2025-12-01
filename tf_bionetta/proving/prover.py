"""
Package responsible for generating the proof for a given model.
"""

import os
import shutil
import json
from pathlib import Path
from typing import Dict

from tf_bionetta.codegen.utils import get_system
from tf_bionetta.logging.pretty import run_command
from tf_bionetta.specs.backend_enums import ProvingBackend, WitnessGenerator, Groth16, UltraGroth
from tf_bionetta.specs.engine import Engine

SYSTEM_NAME = get_system()


class Prover:
    """
    Class responsible for generating the proof for a given model. It
    supports both Groth16 and UltraGroth proving systems. UltraGroth is
    currently disabled thus the `prove_ultra_groth` method raises a
    `NotImplementedError`.
    """
    
    # We use Groth16 as the default proving backend as of now.
    DEFAULT_PROVING_BACKEND: ProvingBackend = ProvingBackend.GROTH16()
    
    def __init__(
        self,
        repos_dir,
        model_dir,
        model_name,
        engine: Engine  
    ) -> None:
        self.repos_dir = repos_dir
        self.model_dir = model_dir
        self.model_name = model_name
        self.engine = engine
    

    def prove_groth(self, target_dir: str) -> dict :
        match SYSTEM_NAME:
            case "x86_64-apple-darwin" | "x86_64-unknown-linux-gnu" :
                prover_dir = f'{self.repos_dir}/ultragroth/package/bin/prover'
            case "aarch64-apple-darwin" :
                prover_dir = f'{self.repos_dir}/ultragroth/package_macos_arm64/bin/prover'
            case "aarch64-unknown-linux-gnu" :
                prover_dir = f'{self.repos_dir}/ultragroth/package/bin/prover'
        
        run_command([
            prover_dir,
            f'{self.model_dir}/{self.model_name}_circom/{self.model_name}.zkey',
            f'{self.model_dir}/{self.model_name}.wtns',
            f'{target_dir}/{self.model_name}_proof.json',
            f'{target_dir}/{self.model_name}_public.json',
        ])
        
        with open(f'{target_dir}/{self.model_name}_proof.json', "r") as f:
            raw = f.read()
            cleaned = raw[:raw.find('}')+1]
            data = json.loads(cleaned)
            
        with open(f'{target_dir}/{self.model_name}_public.json', "r") as f:
            raw = f.read()
            cleaned = raw[:raw.find(']')+1]
            data["public"] = json.loads(cleaned)
            
        return data


    def prove_ultra_groth(self, target_dir) -> dict:
        """
        Generate the proof for the UltraGroth proving system.
        """
        match SYSTEM_NAME:
            case "x86_64-apple-darwin" | "x86_64-unknown-linux-gnu" :
                prover_dir = 'build_prover'
            case "aarch64-apple-darwin":
                prover_dir = 'build_prover_macos_arm64'
            case "aarch64-unknown-linux-gnu":
                prover_dir = 'build_prover_arm64'

        run_command([
            f"{self.repos_dir}/ultragroth/{prover_dir}/src/prover_ultra_groth",
            f"{self.model_dir}/{self.model_name}_circom/{self.model_name}_final.zkey",
            f"{self.model_dir}/{self.model_name}.uwtns",
            f"{target_dir}/{self.model_name}_proof.json",
            f"{target_dir}/{self.model_name}_public.json"
        ])

        with open(f'{target_dir}/{self.model_name}_proof.json', "r") as f:
            raw = f.read()
            cleaned = raw[:raw.find('}')+1]
            data = json.loads(cleaned)
            
        with open(f'{target_dir}/{self.model_name}_public.json', "r") as f:
            raw = f.read()
            cleaned = raw[:raw.find(']')+1]
            data["public"] = json.loads(cleaned)
            
        return data


    def full_prove(self, target_dir) -> Dict:
        """
        Conducts the full proving process, including witness generation and proof generation.
        """

        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)

        if isinstance(self.engine.proving_backend, Groth16):
            return self.prove_groth(target_dir)
        elif isinstance(self.engine.proving_backend, UltraGroth):
            return self.prove_ultra_groth(target_dir)
        else:
            raise NotImplementedError("This is unknown proving backend")
