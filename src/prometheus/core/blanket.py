"""Immutable Markov blanket schema and essential-variable bounds.

Protected core code: harness variants may NEVER modify this schema or the
essential-variable bounds.  All cognitive-core components interact with the
Hidden-Law Lab exclusively through ``RestrictedWorld`` (which enforces the
Markov blanket by hiding ground truth, family metadata, and internal tables).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence, runtime_checkable

from ..lab.laws import LAW_NAMES
from ..lab.world import HiddenLawWorld

ActionKind = Literal["combine", "transform"]


@runtime_checkable
class WorldView(Protocol):
    """Sensory/active boundary exposed to the cognitive core."""

    @property
    def symbols(self) -> list[str]: ...

    @property
    def noise(self) -> float: ...

    @property
    def budget(self) -> float: ...

    @property
    def remaining(self) -> float: ...

    @property
    def spent(self) -> float: ...

    @property
    def combine_cost(self) -> float: ...

    @property
    def transform_cost(self) -> float: ...

    def combine(self, a: str, b: str) -> str: ...

    def transform(self, a: str) -> str: ...


class RestrictedWorld:
    """Strict Markov-blanket proxy around :class:`HiddenLawWorld`.

    Exposes only the lawful sensory/active interface.  Ground-truth attributes
    (``ground_truth``, ``family``, ``map_kind``, ``_op``, ``_f``, ``true_tables``,
    ``shift_laws``, ``seed``) are inaccessible.
    """

    __slots__ = ("_world",)

    def __init__(self, world: HiddenLawWorld) -> None:
        self._world = world

    @property
    def symbols(self) -> list[str]:
        return list(self._world.symbols)

    @property
    def noise(self) -> float:
        return self._world.noise

    @property
    def budget(self) -> float:
        return self._world.budget

    @property
    def remaining(self) -> float:
        return self._world.remaining

    @property
    def spent(self) -> float:
        return self._world.spent

    @property
    def combine_cost(self) -> float:
        return self._world.combine_cost

    @property
    def transform_cost(self) -> float:
        return self._world.transform_cost

    def combine(self, a: str, b: str) -> str:
        return self._world.combine(a, b)

    def transform(self, a: str) -> str:
        return self._world.transform(a)


@dataclass(frozen=True)
class ExperimentAction:
    """One active-state query emitted across the Markov blanket."""

    kind: ActionKind
    args: tuple[str, ...]
    cost: float

    def __post_init__(self) -> None:
        if self.kind == "combine":
            if len(self.args) != 2:
                raise ValueError(f"combine action requires 2 args, got {self.args!r}")
        elif self.kind == "transform":
            if len(self.args) != 1:
                raise ValueError(f"transform action requires 1 arg, got {self.args!r}")
        else:
            raise ValueError(f"unknown action kind: {self.kind!r}")
        if self.cost < 0:
            raise ValueError(f"action cost must be non-negative, got {self.cost}")

    def execute(self, view: WorldView) -> str:
        if self.kind == "combine":
            return view.combine(self.args[0], self.args[1])
        return view.transform(self.args[0])


@dataclass(frozen=True)
class Observation:
    """One sensory-state return across the Markov blanket."""

    step: int
    action: ExperimentAction
    result: str
    surprise: float


@dataclass(frozen=True)
class ActionScore:
    """Expected Free Energy breakdown for one candidate experiment."""

    action: ExperimentAction
    efe: float
    law_eig_bits: float
    cell_eig_bits: float
    expected_cost: float


@dataclass(frozen=True)
class LawBelief:
    """Internal belief state over the eight Hidden-Law Lab laws."""

    law_probs: dict[str, float]
    joint_probs: tuple[float, ...]
    entropy_bits: float
    map_claims: dict[str, bool]
    identity_probs: dict[str, float]

    @property
    def mean_confidence(self) -> float:
        if not self.law_probs:
            return 0.5
        return sum(max(p, 1.0 - p) for p in self.law_probs.values()) / len(self.law_probs)

    def ordered_probs(self) -> tuple[float, ...]:
        return tuple(float(self.law_probs[name]) for name in LAW_NAMES)


@dataclass(frozen=True)
class EssentialBounds:
    """Immutable Ashby essential-variable bounds (protected from variants)."""

    min_confidence: float = 0.80
    max_budget_fraction: float = 0.90
    max_rolling_surprise: float = 2.50
    max_sandbox_error_rate: float = 0.10


DEFAULT_ESSENTIAL_BOUNDS = EssentialBounds()


@dataclass(frozen=True)
class EssentialMetrics:
    """Observable essential-variable telemetry for one episode."""

    mean_confidence: float
    final_entropy_bits: float
    budget_spent: float
    budget_fraction: float
    observations_used: int
    mean_surprise: float
    max_surprise: float
    mdl_compression_bits: float
    llm_calls: int = 0
    sandbox_errors: int = 0

    def check_bounds(
        self, bounds: EssentialBounds = DEFAULT_ESSENTIAL_BOUNDS
    ) -> tuple[bool, list[str]]:
        violations: list[str] = []
        if self.mean_confidence < bounds.min_confidence:
            violations.append(
                f"mean_confidence {self.mean_confidence:.3f} < {bounds.min_confidence:.3f}"
            )
        if self.budget_fraction > bounds.max_budget_fraction:
            violations.append(
                f"budget_fraction {self.budget_fraction:.3f} > {bounds.max_budget_fraction:.3f}"
            )
        if self.mean_surprise > bounds.max_rolling_surprise:
            violations.append(
                f"mean_surprise {self.mean_surprise:.3f} > {bounds.max_rolling_surprise:.3f}"
            )
        err_rate = (
            self.sandbox_errors / max(1, self.observations_used)
            if self.sandbox_errors
            else 0.0
        )
        if err_rate > bounds.max_sandbox_error_rate:
            violations.append(
                f"sandbox_error_rate {err_rate:.3f} > {bounds.max_sandbox_error_rate:.3f}"
            )
        return (len(violations) == 0, violations)
