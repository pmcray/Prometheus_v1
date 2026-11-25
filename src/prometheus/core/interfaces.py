"""Pluggable backend and Phase 2/3 extension protocols.

- ``InferenceBackend`` abstracts the Friston active-inference / EFE engine so
  ``ExactBayesianBackend`` (pure Python, Phase 1 default) can later be swapped
  with ``pymdp`` or ``RxInfer.jl`` without changing callers.
- ``AnalogyTransferAdapter`` defines the Phase 2 Hofstadter structural-analogy
  and cross-world transfer seam.
- ``UltrastabilityMonitor`` defines the Phase 2/3 Ashby essential-variable
  monitor seam that triggers bounded harness-mutation proposals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

from .blanket import ActionScore, EssentialMetrics, ExperimentAction, LawBelief, Observation


@runtime_checkable
class InferenceBackend(Protocol):
    """Protocol implemented by Friston active-inference backends."""

    def reset(
        self,
        symbols: Sequence[str],
        *,
        noise: float,
        combine_cost: float,
        transform_cost: float,
        prior_joint: tuple[float, ...] | None = None,
    ) -> None: ...

    def set_prior(self, prior_joint: tuple[float, ...]) -> None: ...

    def observe(self, action: ExperimentAction, result: str) -> float:
        """Assimilate one observation and return its Bayesian surprise (nats)."""
        ...

    def belief(self) -> LawBelief: ...

    def score_actions(self, remaining_budget: float) -> list[ActionScore]: ...

    def should_stop(self, scores: Sequence[ActionScore]) -> tuple[bool, str]: ...

    def feature_summary(self) -> dict[str, float]:
        """Return symbol-agnostic structural invariants inferred from beliefs."""
        ...

    def surprise_stats(self) -> tuple[float, float]:
        """Return ``(mean_surprise, max_surprise)`` across the episode so far."""
        ...


@dataclass(frozen=True)
class TransferHint:
    """Phase 2 Hofstadter analogy/transfer output consumed by the cognitive loop."""

    prior_joint: tuple[float, ...] | None = None
    suggested_actions: tuple[ExperimentAction, ...] = ()
    source_assembly_id: str | None = None
    confidence: float = 0.0


@runtime_checkable
class AnalogyTransferAdapter(Protocol):
    """Phase 2 seam for Hofstadter structural analogy and law-shift transfer."""

    def propose_transfer(
        self,
        belief: LawBelief,
        observations: Sequence[Observation],
        assemblies: Sequence[Any],
    ) -> TransferHint | None: ...


@dataclass(frozen=True)
class AshbyConfig:
    """Mutable cadence settings for Ashby monitoring (never weakens EssentialBounds)."""

    window_size: int = 8
    min_episodes_before_trigger: int = 4
    min_breach_episodes: int = 2
    cooldown_episodes: int = 6
    surprise_spike_threshold: float = 3.5


@runtime_checkable
class UltrastabilityMonitor(Protocol):
    """Phase 2/3 seam for Ashby essential-variable monitoring."""

    def record(self, metrics: EssentialMetrics) -> bool:
        """Record episode essential metrics; return True if adaptation should trigger."""
        ...

    def violations(self) -> list[str]: ...
