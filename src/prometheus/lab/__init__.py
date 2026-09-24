"""Hidden-Law Lab: procedurally generated finite worlds with secret algebraic laws."""

from .laws import LAWS, evaluate_laws
from .world import HiddenLawWorld

__all__ = ["HiddenLawWorld", "LAWS", "evaluate_laws"]
