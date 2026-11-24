"""Prometheus v1 Phase 1 Cognitive Core (protected/immutable code).

Implements the Markov blanket, Friston active-inference / EFE backend,
Good's MDL assembly-style memory, strict immutable->mutable harness loader,
hybrid multi-turn episode runner, and Phase 2/3 extension protocols.
"""

from .agent import CognitiveAgent, CognitiveEpisodeResult
from .blanket import (
    DEFAULT_ESSENTIAL_BOUNDS,
    ActionScore,
    EssentialBounds,
    EssentialMetrics,
    ExperimentAction,
    LawBelief,
    Observation,
    RestrictedWorld,
    WorldView,
)
from .harness_loader import (
    ALLOWED_SUBDIRS,
    HarnessValidationError,
    LoadedHarness,
    MeshConfig,
    load_harness,
    run_harness_evaluator,
    run_harness_tool,
)
from .inference import (
    NUM_VALID_PROFILES,
    UNIFORM_JOINT_PRIOR,
    VALID_LAW_PROFILES,
    EFEParams,
    ExactBayesianBackend,
)
from .interfaces import (
    AnalogyTransferAdapter,
    InferenceBackend,
    TransferHint,
    UltrastabilityMonitor,
)
from .memory import (
    Assembly,
    AssemblyMemory,
    ConsolidationReport,
    MemoryParams,
    RecallResult,
)

__all__ = [
    "ALLOWED_SUBDIRS",
    "ActionScore",
    "AnalogyTransferAdapter",
    "Assembly",
    "AssemblyMemory",
    "CognitiveAgent",
    "CognitiveEpisodeResult",
    "ConsolidationReport",
    "DEFAULT_ESSENTIAL_BOUNDS",
    "EFEParams",
    "EssentialBounds",
    "EssentialMetrics",
    "ExactBayesianBackend",
    "ExperimentAction",
    "HarnessValidationError",
    "InferenceBackend",
    "LawBelief",
    "LoadedHarness",
    "MemoryParams",
    "MeshConfig",
    "NUM_VALID_PROFILES",
    "Observation",
    "RecallResult",
    "RestrictedWorld",
    "TransferHint",
    "UNIFORM_JOINT_PRIOR",
    "UltrastabilityMonitor",
    "VALID_LAW_PROFILES",
    "WorldView",
    "load_harness",
    "run_harness_evaluator",
    "run_harness_tool",
]
