"""Foundation-model bake-off for Phase 0.

Runs 200 Hidden-Law Lab episodes against each candidate model and records
accuracy, latency, and per-family breakdown to an MLflow file store.
"""

from .models import MODEL_REGISTRY, ModelSpec
from .runner import BakeoffConfig, BakeoffRunner, BakeoffSummary

__all__ = [
    "MODEL_REGISTRY",
    "ModelSpec",
    "BakeoffConfig",
    "BakeoffRunner",
    "BakeoffSummary",
]
