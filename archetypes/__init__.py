from importlib.metadata import version

from archetypes.numpy import (
    AA,
    ADA,
    BiAA,
    FairAA,
    FairKernelAA,
    KernelAA,
    SymmetricBiAA,
)

__all__ = [
    "AA",
    "BiAA",
    "NAA",
    "ADA",
    "SymmetricBiAA",
    "FairAA",
    "KernelAA",
    "FairKernelAA",
]

__version__ = version("archetypes")


def __getattr__(name):
    if name == "NAA":
        from archetypes.numpy import NAA

        return NAA
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
