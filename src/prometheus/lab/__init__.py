"""Hidden-Law Lab: procedurally generated finite worlds with secret algebraic laws."""

from .laws import LAWS, evaluate_laws
from .lean_layer import (
    BatchVerifyResult,
    batch_lean_source,
    verify_lean_source,
    verify_world,
    verify_world_batch,
    world_lean_source,
)
from .world import HiddenLawWorld

__all__ = [
    "BatchVerifyResult",
    "HiddenLawWorld",
    "LAWS",
    "batch_lean_source",
    "evaluate_laws",
    "verify_world",
    "verify_world_batch",
    "verify_lean_source",
    "world_lean_source",
]
