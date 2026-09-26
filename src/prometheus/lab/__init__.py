"""Hidden-Law Lab: procedurally generated finite worlds with secret algebraic laws."""

from .laws import LAWS, evaluate_laws
from .lean_layer import verify_world, verify_lean_source, world_lean_source
from .world import HiddenLawWorld

__all__ = [
    "HiddenLawWorld",
    "LAWS",
    "evaluate_laws",
    "verify_world",
    "verify_lean_source",
    "world_lean_source",
]
