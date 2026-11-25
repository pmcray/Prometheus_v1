"""Foundation-model bake-off for Phase 0.

Runs 200 Hidden-Law Lab episodes against each candidate model and records
accuracy, latency, and per-family breakdown to an MLflow file store.
"""

from .models import BAKEOFF_MODELS, MODEL_REGISTRY, ModelSpec, vllm_flags_for_gpu
from .prover_runner import (
    PROVER_BAKEOFF_MODELS,
    ProverBakeoffConfig,
    ProverBakeoffRunner,
    ProverBakeoffSummary,
    summarise_prover_results,
)
from .runner import BakeoffConfig, BakeoffRunner, BakeoffSummary

__all__ = [
    "BAKEOFF_MODELS",
    "MODEL_REGISTRY",
    "ModelSpec",
    "PROVER_BAKEOFF_MODELS",
    "ProverBakeoffConfig",
    "ProverBakeoffRunner",
    "ProverBakeoffSummary",
    "summarise_prover_results",
    "vllm_flags_for_gpu",
    "BakeoffConfig",
    "BakeoffRunner",
    "BakeoffSummary",
]
