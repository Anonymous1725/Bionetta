"""
Helper utility functions for the codegen module.
"""

import resource
import platform
import psutil
import os
import re
import subprocess
from pathlib import Path
from typing import Tuple, Any, Callable

def log_step(description: str) -> Callable:
    """
    Decorator to log the start and end of a function execution.
    """
    
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            self._logger.info(f"[blue]→ {description}...[/]")
            result, duration = measure_time(lambda: func(self, *args, **kwargs))
            self._logger.info(f"[green]✓ {description} completed in {duration:.2f}s[/]")
            return result
        return wrapper
    return decorator


def measure_time(fn: callable) -> Tuple[Any, float]:
    """
    Executes the function and measures the time it takes to run.
    Args:
        fn (callable): The function to execute.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.
    
    Returns:
        Tuple[Any, float]: A tuple containing the result of the function and the time it took to run.
    """
    
    usage_start = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = fn()
    usage_end = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_time = usage_end.ru_utime - usage_start.ru_utime + usage_end.ru_stime - usage_start.ru_stime
    return result, cpu_time


def get_system() -> str:
    """
    Returns the system name.
    """

    os_name = platform.system()
    arch = platform.machine()

    if os_name == "Linux":
        system = "x86_64-unknown-linux-gnu" if arch == "x86_64" else "aarch64-unknown-linux-gnu"
    elif os_name == "Darwin":
        system = "aarch64-apple-darwin" if arch == "arm64" else "x86_64-apple-darwin"
    else:
        raise Exception("Unsupported OS")

    return system


def obtain_available_ram_in_the_system() -> int:
    return int(psutil.virtual_memory().total // (1024**2) * 0.9)


def receive_constraints_num_for_r1cs(r1cs_file : Path) -> int:
    # Some magic to get constraint number from command
    #r1cs_info_command = f'snarkjs r1cs info {r1cs_file}'.split(' ')
    #r1cs_info = run_command(command=r1cs_info_command).stdout
    #constraint_info = r1cs_info.split('\n')[2]
    #return int(constraint_info[constraint_info.rfind(" ") + 1:])

    header = extract_r1cs_header(r1cs_file)
    return header.mConstraints


class R1CSHeader:
    def __init__(self, field_size: int, prime: int, nWires: int, nPubOut: int, nPubIn: int, nPrvIn: int, nLabels: int, mConstraints: int):
        self.field_size = field_size
        self.prime = prime
        self.nWires = nWires
        self.nPubOut = nPubOut
        self.nPubIn = nPubIn
        self.nPrvIn = nPrvIn
        self.nLabels = nLabels
        self.mConstraints = mConstraints


def gather_le_bytes(file, bytes_num) -> int: 
    """
    Reads given number of bytes from file, creates little endian integer and increments reading index.
    """
    chunk = file.read(bytes_num)
    #print(chunk)
    num = int.from_bytes(chunk, byteorder='little')

    return num

def extract_r1cs_header(r1cs_file: Path) -> R1CSHeader:
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
    return R1CSHeader(
        nWires=r1cs_info['Wires'],
        nPubOut=r1cs_info['Outputs'],
        nPubIn=r1cs_info['Public Inputs'],
        nPrvIn=r1cs_info['Private Inputs'],
        nLabels=r1cs_info['Labels'],
        mConstraints=r1cs_info['Constraints'],
    )
