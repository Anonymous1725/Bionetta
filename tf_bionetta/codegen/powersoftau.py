"""
Code for loading the powers of tau file
"""
from __future__ import annotations

from typing import Dict, List
from pathlib import Path
import logging
import urllib.request
import re
import os

from rich.progress import Progress

# Dictionary, containing power as the key and 
# the corresponding url to download the .ptau 
# file as the value
POWERS_OF_TAU_FILES: Dict[int, str] = {
    8: 'https://storage.googleapis.com/zkevm/ptau/powersOfTau28_hez_final_08.ptau',
    9: 'https://storage.googleapis.com/zkevm/ptau/powersOfTau28_hez_final_09.ptau',
    10: 'https://storage.googleapis.com/zkevm/ptau/powersOfTau28_hez_final_10.ptau',
    11: 'https://storage.googleapis.com/zkevm/ptau/powersOfTau28_hez_final_11.ptau',
    12: 'https://storage.googleapis.com/zkevm/ptau/powersOfTau28_hez_final_12.ptau',
    13: 'https://storage.googleapis.com/zkevm/ptau/powersOfTau28_hez_final_13.ptau',
    14: 'https://storage.googleapis.com/zkevm/ptau/powersOfTau28_hez_final_14.ptau',
    15: 'https://storage.googleapis.com/zkevm/ptau/powersOfTau28_hez_final_15.ptau',
    16: 'https://storage.googleapis.com/zkevm/ptau/powersOfTau28_hez_final_16.ptau',
    17: 'https://storage.googleapis.com/zkevm/ptau/powersOfTau28_hez_final_17.ptau',
    18: 'https://storage.googleapis.com/zkevm/ptau/powersOfTau28_hez_final_18.ptau',
    19: 'https://storage.googleapis.com/zkevm/ptau/powersOfTau28_hez_final_19.ptau',
    20: 'https://storage.googleapis.com/zkevm/ptau/powersOfTau28_hez_final_20.ptau',
    21: 'https://storage.googleapis.com/zkevm/ptau/powersOfTau28_hez_final_21.ptau',
    22: 'https://storage.googleapis.com/zkevm/ptau/powersOfTau28_hez_final_22.ptau',
    23: 'https://storage.googleapis.com/zkevm/ptau/powersOfTau28_hez_final_23.ptau',
    24: 'https://storage.googleapis.com/zkevm/ptau/powersOfTau28_hez_final_24.ptau',
}

# Default power to use when no power nor the number of constraints
# is specified
DEFAULT_POWER: int = 20

# Minimum supported power of tau
MIN_SUPPORTED_POWER: int = min(POWERS_OF_TAU_FILES.keys())

# Maximum supported power of tau
MAX_SUPPORTED_POWER: int = max(POWERS_OF_TAU_FILES.keys())

# In what magnitude to increase the power of tau in safe mode
# (if the number of constraints is an estimate)
SAFE_POWER_OVERHEAD: int = 0


class PowersOfTauLoader:
    """
    Class responsible for downloading the powers of tau files.
    """
    
    def __init__(self, logger: logging.Logger) -> None:
        """
        Initialize the loader with a logger.
        
        Args:
            logger (`logging.Logger`): A logger object.
        """
        
        self._logger = logger
    
    
    @staticmethod
    def _max_constraints_from_power(power: int) -> int:
        """
        Returns the maximum number of allowed constraints in the 
        circuit for the given power of tau.
        """
        
        return 1<<power
    
    
    @staticmethod
    def _compute_optimal_power(constraints_number: int, safe: bool = True) -> int:
        """
        Computes the optimal power of tau for the given number of constraints.
        
        Args:
            constraints_number (`int`): The number of constraints.
        
        Returns:
            `int`: The optimal power of tau.
            `safe`: Whether to use the safe mode. Meaning, if the specified
            number of constraints is the estimate, the function will take the larger power
        """
        
        # Asserting that we can support the given number of constraints
        max_allowed_constraints = PowersOfTauLoader._max_constraints_from_power(MAX_SUPPORTED_POWER)
        assert constraints_number < max_allowed_constraints, f"Constraints number {constraints_number} is too large. The maximum supported power is {MAX_SUPPORTED_POWER}, corresponding to {max_allowed_constraints} constraints."
        
        # NOTE: We can simply take floor(log2(constraints_number)), but 
        # this works only for the current version of PowersOfTauLoader._max_constraints_from_power
        # function. So to make implementation generic, we do the while loop.
        optimal_power = MIN_SUPPORTED_POWER
        while PowersOfTauLoader._max_constraints_from_power(optimal_power) < constraints_number:
            optimal_power += 1
           
        # If the safe mode is enabled, we increase the power of tau
        # to cover the specified estimated number of constraints
        # for sure. This is done by adding the SAFE_POWER_OVERHEAD
        # to the power of tau. 
        if safe:
            optimal_power += SAFE_POWER_OVERHEAD
            # To avoid downloading the power of tau that is not supported,
            # we limit the power to the maximum supported power
            optimal_power = min(optimal_power, MAX_SUPPORTED_POWER)
        
        return optimal_power
    
    
    @staticmethod
    def form_ptau_file_name(power: int) -> str:
        """
        Forms the file name for the powers of tau file.
        
        Args:
            power (`int`): The power of tau.
        
        Returns:
            `str`: The file name.
        """
        
        return f"powersOfTau_{power}.ptau"
    
    
    @staticmethod
    def get_ptau_files(target_directory: Path) -> List[str]:
        files = []
        for file in target_directory.iterdir():
            if file.is_file():
                files.append(file.name)
        return files



    @staticmethod
    def form_ptau_file_path(
        target_directory: Path, 
        power: int = None,
        constraints_number: int = None,
    ) -> Path:
        """
        Returns the path to the powers of tau file.
        
        Args:
            target_directory (`Path`): The directory to save the file.
            power (`int`): The power of tau.
            constraints_number (`int`): The number of constraints.
        
        Returns:
            `Path`: The path to the powers of tau file.
        """
        
        power_to_download = DEFAULT_POWER
        if power is not None:
            power_to_download = power
        
        if constraints_number is not None:
            power_to_download = PowersOfTauLoader._compute_optimal_power(constraints_number)
        
        # Asserting that the power is supported
        assert power_to_download in POWERS_OF_TAU_FILES.keys(), f"Power {power_to_download} is not supported. Supported powers are: {POWERS_OF_TAU_FILES.keys()}"
        
        
        pattern = re.compile(r"powersOfTau_(\d+)\.ptau")
        powers = [power_to_download]
        
        os.makedirs(target_directory, exist_ok=True)
        for filename in PowersOfTauLoader.get_ptau_files(target_directory):
            match = pattern.fullmatch(filename)
            if match:
                powers.append(int(match.group(1)))

        return target_directory / PowersOfTauLoader.form_ptau_file_name(max(powers))
        
    
    def download(
        self, 
        target_directory: Path,
        power: int = None,
        constraints_number: int = None,
        safe: bool = True,
        progress: Progress | None = None,
    ) -> Path:
        """
        Given the power of tau or the number of constraints, downloads the
        corresponding powers of tau file (if only the number of constraints is
        specified, the corresponding minimal power is calculated). If no power
        nor constraints number is specified, the default power is used.
        
        Args:
            target_directory (`Path`): The directory to save the file.
            power (`int`, optional): The power of tau to download. Defaults to `None`.
            constraints_number (`int`, optional): The number of constraints. Defaults to `None`.
            safe (`bool`, optional): Whether to use the safe mode. Meaning, if the specified
            number of constraints is the estimate, the function will take the larger power 
            of tau to cover the specified number of constraints for sure. Defaults to `True`.
            progress (`Progress`, optional): The progress bar to use. Defaults to `None`.
        Returns:
            `Path`: The path to the downloaded file.
        """
        
        # First, we calculate the power to be downloaded
        power_to_download = DEFAULT_POWER
        if power is not None:
            power_to_download = power
        
        if constraints_number is not None:
            power_to_download = PowersOfTauLoader._compute_optimal_power(constraints_number, safe=safe)
            self._logger.info(f"We choose the power of tau {power_to_download} for the specified number of constraints: {constraints_number}")
        
        # Asserting that the power is supported
        assert power_to_download in POWERS_OF_TAU_FILES.keys(), f"Power {power_to_download} is not supported. Supported powers are: {POWERS_OF_TAU_FILES.keys()}"
        
        # Downloading the file
        url = POWERS_OF_TAU_FILES[power_to_download]
        ptau_file = PowersOfTauLoader.form_ptau_file_path(target_directory, power=power_to_download)
        target_directory.mkdir(parents=True, exist_ok=True)  # Create the directory if it does not exist

        if ptau_file.exists():
            self._logger.info(f"Use existing powers of tau {ptau_file}")
            return ptau_file

        self._logger.info(f"Downloading powers of tau file from {url} to {ptau_file}...")

        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                total_size = int(response.getheader("Content-Length").strip())
                if progress is not None:
                    task_id = progress.add_task("Downloading the .ptau file...", total=total_size)

                with open(ptau_file, "wb") as file:
                    CHUNK_SIZE = 8192
                    while (chunk := response.read(CHUNK_SIZE)):
                        file.write(chunk)
                        if progress is not None:
                            progress.update(task_id, advance=len(chunk))
                
                if progress is not None:
                    progress.remove_task(task_id=task_id)
                
            self._logger.info(f"Powers of tau file downloaded to {target_directory}.")
            return ptau_file
        except urllib.error.URLError as e:
            self._logger.error(f"Failed to download {url}: {e}")
            raise
        except IOError as e:
            self._logger.error(f"Failed to save file {target_directory}: {e}")
            raise
        except Exception as e:
            self._logger.exception(f"Unexpected error while downloading {url}: {e}")
            raise
