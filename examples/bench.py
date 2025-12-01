import os
import time
import re
import subprocess
from pathlib import Path
from typing import Dict
from enum import Enum

import typer
import psutil
import tensorflow as tf
import numpy as np

import tf_bionetta as tfb
from .ptau_files_check_size import get_ptau_size_gb


app = typer.Typer(help=('CLI for benchmark concrete model using bionetta.'))

def obtain_available_ram_in_the_system() -> int:
    return int(psutil.virtual_memory().total // (1024**2) * 0.9)


def get_r1cs_info(r1cs_file: Path) -> int:
    env = os.environ.copy()
    env["NODE_OPTIONS"] = f"--max_old-space-size={obtain_available_ram_in_the_system()}"

    result = subprocess.run(
        ["snarkjs", "r1cs", "info", r1cs_file],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )

    # Example of r1cs info output:
    # [INFO]  snarkJS: Curve: bn-128
    # [INFO]  snarkJS: # of Wires: 101
    # [INFO]  snarkJS: # of Constraints: 102
    # [INFO]  snarkJS: # of Private Inputs: 103
    # [INFO]  snarkJS: # of Public Inputs: 104
    # [INFO]  snarkJS: # of Labels: 105
    # [INFO]  snarkJS: # of Outputs: 2

    # Regex to extract key-value pairs from "[INFO] ..."
    pattern = re.compile(r"# of ([\w\s]+): (\d+)")
    r1cs_info = {match[0]: int(match[1]) for match in pattern.findall(result.stdout)}
    return r1cs_info['Constraints']


def get_special_benches(model_path: Path) -> Dict[str, float | int]:
    # 1. Find directory ending with "_circom"
    circom_dir = next((d for d in model_path.rglob("*_circom") if d.is_dir()), None)
    if not circom_dir:
        raise FileNotFoundError("No '_circom' directory found under given model path")

    # 2. Find R1CS file (prefer *_injected.r1cs)
    r1cs_file = next(circom_dir.rglob("*_injected.r1cs"), None)
    proving_backend = "UltraGroth"
    if not r1cs_file:
        proving_backend = "Groth16"
        r1cs_file = next(circom_dir.rglob("*.r1cs"), None)
    if not r1cs_file:
        raise FileNotFoundError(f"No .r1cs file found in '_circom' directory")

    # 3. Find *_vkey.json file
    vkey_file = next(circom_dir.rglob("*_vkey.json"), None)
    if not vkey_file:
        vkey_file = next(circom_dir.rglob("verification_key.json"))
    if not vkey_file:
        raise FileNotFoundError(f"No '_vkey.json' file found in '_circom' directory")
    
    zkey_file = next(circom_dir.rglob("*_final.zkey"), None)
    if not zkey_file:
        zkey_file = next(circom_dir.rglob("*.zkey"))
    if not zkey_file:
        raise FileNotFoundError("No '*.zkey' file found in '_circom' directory")
    
    proof_file = next(model_path.rglob("proof/*_proof.json"))
    if not proof_file:
        raise FileNotFoundError("No 'proof/*_proof.json' file found in '_circom' directory")

    precision = 5
    constraints = get_r1cs_info(r1cs_file)
    return {
        "ProvingBackend": proving_backend,
        "Constraints": constraints,
        "Proof Size (KB)": round(proof_file.stat().st_size / 1000.0, precision),
        "PK (MB)": round(zkey_file.stat().st_size / 1_000_000.0, precision),
        "VK (KB)": round(vkey_file.stat().st_size / 1000.0, precision),
        "Trusted Setup (GB)": round(get_ptau_size_gb(constraints), precision),
    }


def bench_full(model_path: Path, iters: int = 10, ) -> Dict[str, float | int]:
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    X_train = np.expand_dims(X_train, axis=-1)  # Add channel dimension
    X_test = np.expand_dims(X_test, axis=-1)  # Add channel dimension
    model_name = None

    # MiniVgg11 model or ResNet18
    if "model3" in str(model_path) or "model4" in str(model_path):
        X_train = tf.image.resize(X_train, (32, 32))
        X_test = tf.image.resize(X_test, (32, 32))

        X_train = tf.image.grayscale_to_rgb(tf.convert_to_tensor(X_train)).numpy()
        X_test = tf.image.grayscale_to_rgb(tf.convert_to_tensor(X_test)).numpy()

    # MobileNetV2 imitation
    elif "model5" in str(model_path):
        X_train = tf.image.resize(X_train, (32,32)).numpy()
        X_test = tf.image.resize(X_test, (32,32)).numpy()

        # One-hot encode labels
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)
        model_name = "MobileNetV2"


    loaded_model = tfb.BionettaModel.load_from_compiled_folder(model_path, name=model_name, verbose=2)
    proof_dir = f'{model_path}/proof'

    # Execute proving
    prove_time = 0
    for _ in range(iters):
        test_input = X_test[np.random.randint(len(X_test))]
        start = time.perf_counter()
        loaded_model.prove(
            input=test_input,
            target_dir=proof_dir
        )
        end = time.perf_counter()
        prove_time += end - start

    prf_drtn = prove_time / iters

    verify_time = 0
    for _ in range(iters):
        start = time.perf_counter()
        loaded_model.verify(proof_dir=proof_dir)
        end = time.perf_counter()
        verify_time += end - start

    vrf_drtn = verify_time / iters

    benches = get_special_benches(model_path)
    benches["Prove (s)"] = prf_drtn
    benches["Verify (s)"] = vrf_drtn
    return benches


class ModelName(str, Enum):
    model1 = "model1"
    model2 = "model2"
    model3 = "model3"
    model4 = "model4"
    model5 = "model5"


@app.command()
def bench_model(
    model_name: ModelName = typer.Argument(..., help="Select one of the predefined models"),
    ultragroth: bool = typer.Argument(..., help="True = UltraGroth, False = Groth16"),
):
    """
    Run benchmark for a specific model.
    """
    if not ultragroth and model_name in (ModelName.model4, ModelName.model5):
        raise NotImplementedError("There are no benchmarks for ResNet18 and MobileNetV2 models with Groth16 ProvingBackend.")

    suffix = "ultragroth" if ultragroth else "groth16"
    model_path = f"./examples/saved_models/{model_name.value}_{suffix}"

    res = bench_full(Path(model_path), iters=1)
    
    print(f"\nBenchmarks for {model_path.split('/')[-1]}:")
    for key, value in res.items():
        print(f"{key:<20} : {value}")


if __name__ == "__main__":
    app()