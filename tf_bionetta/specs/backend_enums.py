"""
Enum for Backends used for compilation and proving
"""

from enum import IntEnum
from typing import Tuple


class Groth16:
    """
    Represents the GROTH16 backend. Acts like a singleton.
    """

    NAME = "GROTH16"
    DEFAULT_PRECISION: int = 15


    def __init__(
        self,
        precision: int = DEFAULT_PRECISION
    ) -> None:
        """
        Initializes the Groth16 backend with a specified precision.
        """

        if not isinstance(precision, int) or precision <= 0:
            raise ValueError(" must be a positive integer.")

        self.precision = precision


    @property
    def name(self) -> str:
        return Groth16.NAME
    

    def initial_constraints(self) -> Tuple[str, int]:
        """
        Returns the proving backend's initial constraints cos (such as lookup table commitment)
        together with the string, indicating what these constraints correspond to
        """
        return ("Groth16 initial constraints", 0)


    def __repr__(self) -> str:
        return "<ProvingBackend.GROTH16>"


class UltraGroth:
    """
    Represents the ULTRAGROTH backend with a configurable limb_size.
    """

    NAME = "ULTRAGROTH"
    DEFAULT_LIMB_SIZE: int = 15
    DEFAULT_PRECISION_MULTIPLICITY: int = 1


    def __init__(
        self,
        limb_size: int = DEFAULT_LIMB_SIZE,
        precision_multiplicity: int = DEFAULT_PRECISION_MULTIPLICITY
    ) -> None:
        """
        Initializes the UltraGroth backend with a specified limb size.
        
        Args:
            limb_size (int): The size of the limbs used in the ULTRAGROTH backend.
            precision_multiplicity (int): The degree of precision to be used by default for arithmetization of the tensors.
        """
        
        if not isinstance(limb_size, int) or limb_size <= 0:
            raise ValueError("limb_size must be a positive integer.")
        
        if not isinstance(precision_multiplicity, int) or precision_multiplicity <= 0:
            raise ValueError("precision_multiplicity must be a positive integer.")
        
        self.limb_size = limb_size
        self.precision_multiplicity = precision_multiplicity


    @property
    def name(self) -> str:
        return UltraGroth.NAME
    

    def initial_constraints(self) -> Tuple[str, int]:
        """
        Returns the proving backend's initial constraints cos (such as lookup table commitment)
        together with the string, indicating what these constraints correspond to
        """

        return ("UltraGroth Lookup Commitment", 1 << (self.limb_size + 1))


    def __repr__(self) -> str:
        return f"<ProvingBackend.ULTRAGROTH(limb_size={self.limb_size})>"


class ProvingBackend:
    """
    Factory class that provides access to proving backend configurations.
    """

    # NOTE: Required for child repositories
    NAME = ""
    
    GROTH16 = Groth16
    ULTRAGROTH = UltraGroth

    def __init__(self):
        raise TypeError("ProvingBackend cannot be instantiated. Use its members directly.")
    

    def initial_constraints(self) -> Tuple[str, int]:
        """
        Returns the proving backend's initial constraints cos (such as lookup table commitment)
        together with the string, indicating what these constraints correspond to
        """
        
        return None, 0


class WitnessGenerator(IntEnum):
    """
    Enum for custom/default witness generation
    """

    DEFAULT = 0
    CUSTOM = 1


    def log_wtns_gen(self) -> str:
        """
        Returns the string representation of the enum
        """
        return {0: "DEFAULT", 1: "CUSTOM"}.get(self.value, "INFO")


class OptimizationLevel(IntEnum):
    """
    Enum for optimization level of circuits. R1CS backend case: 
    - O0 - no simplifications,
    - O1 - only substitutions,
    - O2 - linear simplifications ( be patient !)
    """


    O0 = 0
    O1 = 1
    O2 = 2
    SKIP = 3


    def log_wtns_gen(self) -> str:
        """
        Returns the string representation of the enum
        """
        return {0: "O0", 1: "O1", 2: "O2", 3: "SKIP"}.get(self.value, "INFO")