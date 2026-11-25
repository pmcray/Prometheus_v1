"""Prometheus v1 Phase 1 & 2 Cognitive Core (protected/immutable code).

Implements the Markov blanket, Friston active-inference / EFE backend,
Good's MDL assembly-style memory, Hofstadter structural analogy transfer,
Ashby ultrastability monitoring & bounded self-improvement proposer,
strict immutable->mutable harness loader, and hybrid multi-turn episode runner.
"""

from .agent import CognitiveAgent, CognitiveEpisodeResult
from .ashby import (
    AshbyConfig,
    AshbyProposer,
    CandidateEvaluation,
    HomeostaticMonitor,
    ProposalOutcome,
    StressDiagnosis,
)
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
from .law_shift import (
    LawShiftBenchmarkSummary,
    LawShiftEpisodeResult,
    run_law_shift_benchmark,
    run_law_shift_episode,
    select_ashby_forgetting_harness,
)
from .memory import (
    Assembly,
    AssemblyMemory,
    ConsolidationReport,
    MemoryParams,
    RecallResult,
)
from .multigen import (
    DEFAULT_RUN002_MODEL_PINS,
    GenerationRecord,
    MultiGenConfig,
    MultiGenRunner,
    MultiGenSummary,
    build_pinned_baseline_bundle,
    validate_model_pins,
)
from .transfer import (
    StructuralAnalogyTransfer,
    StructuralDescriptor,
    TransferParams,
    extract_descriptor,
)

__all__ = [
    "ALLOWED_SUBDIRS",
    "ActionScore",
    "AnalogyTransferAdapter",
    "AshbyConfig",
    "AshbyProposer",
    "Assembly",
    "AssemblyMemory",
    "CandidateEvaluation",
    "CognitiveAgent",
    "CognitiveEpisodeResult",
    "ConsolidationReport",
    "DEFAULT_ESSENTIAL_BOUNDS",
    "DEFAULT_RUN002_MODEL_PINS",
    "EFEParams",
    "EssentialBounds",
    "EssentialMetrics",
    "ExactBayesianBackend",
    "ExperimentAction",
    "GenerationRecord",
    "HarnessValidationError",
    "HomeostaticMonitor",
    "InferenceBackend",
    "LawBelief",
    "LawShiftBenchmarkSummary",
    "LawShiftEpisodeResult",
    "LoadedHarness",
    "MemoryParams",
    "MeshConfig",
    "MultiGenConfig",
    "MultiGenRunner",
    "MultiGenSummary",
    "NUM_VALID_PROFILES",
    "Observation",
    "ProposalOutcome",
    "RecallResult",
    "RestrictedWorld",
    "StressDiagnosis",
    "StructuralAnalogyTransfer",
    "StructuralDescriptor",
    "TransferHint",
    "TransferParams",
    "UNIFORM_JOINT_PRIOR",
    "UltrastabilityMonitor",
    "VALID_LAW_PROFILES",
    "WorldView",
    "build_pinned_baseline_bundle",
    "extract_descriptor",
    "load_harness",
    "run_harness_evaluator",
    "run_harness_tool",
    "run_law_shift_benchmark",
    "run_law_shift_episode",
    "select_ashby_forgetting_harness",
    "validate_model_pins",
]
