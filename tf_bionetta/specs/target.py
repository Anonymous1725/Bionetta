
from enum import IntEnum

class TargetPlatform(IntEnum):
    """
    Enum for target platforms
    """

    DESKTOP = 0 # don't sure about this option, although it's possible to simplify some procedures if we targeting desktops, 
                # i.e. we could compile Rust witness-gen to dynamic libs to prevent recompiling C++ code for each model;
                # (it's not possible on IOS since we cannot read dynamic libraries there)

    MOBILE = 1

    WEB = 2 # compile relevant parts of code to Web-friendly environments, i.e. default Circom WASM witness-gen or SNARKJS for proving

    def log_backend(self) -> str:
        """
        Returns the string representation of the enum
        """
        return {0: "DESKTOP", 1: "MOBILE", 2: "WEB"}.get(self.value, "INFO")