"""Single-episode runner for the Hidden-Law Lab bake-off.

Strategy
--------
The cortex model is given the *entire* experiment budget in one shot: the
harness pre-samples a fixed set of observations (all combine pairs + all
transform values, up to the budget) with no noise correction, then asks
the model to classify the eight laws from that transcript.

This "full-information" baseline is appropriate for Phase 0: the point is
to pick a model and calibrate its raw accuracy, not to optimise the
interaction strategy (that comes in later Ashby-loop generations).

The episode is run OUTSIDE the sandbox — the harness code is human-authored
and lives in the repo.  Agent-generated code that may replace
``AGENT_CODE`` in Cell 4 of the Colab notebook is still sandboxed; this
module is the scaffold that surrounds it.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from typing import Any

from ..lab.laws import LAW_NAMES
from ..lab.world import BudgetExhausted, HiddenLawWorld
from .client import ModelClient, ModelClientError
from .prompt import build_messages, parse_claims


@dataclass
class EpisodeResult:
    seed: int
    family: str
    n: int                             # |symbols|
    claims: dict[str, bool]
    truth: dict[str, bool]
    correct: int                       # number of laws correctly classified
    n_laws: int = len(LAW_NAMES)
    observations_used: int = 0
    budget_spent: float = 0.0
    latency_s: float = 0.0
    error: str | None = None           # set if the model call failed

    @property
    def accuracy(self) -> float:
        return self.correct / self.n_laws if self.n_laws else 0.0

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Observation sampler
# ---------------------------------------------------------------------------

def _sample_observations(
    world: HiddenLawWorld,
    max_obs: int = 200,
) -> list[dict[str, Any]]:
    """Fill the world's budget with a deterministic set of experiments.

    Priority order:
      1. All (a, a) combine pairs (idempotency test, cheap).
      2. All (a, b) combine pairs where a ≠ b (commutativity / associativity).
      3. All transform(a) values (f-law tests, half-price).

    Stops early when the budget is exhausted or ``max_obs`` is reached.
    Returns observations as plain dicts (matching world.history schema).
    """
    s = world.symbols
    n = len(s)
    observations: list[dict[str, Any]] = []

    def _record(kind: str, args: tuple, result: str, cost: float) -> None:
        observations.append({
            "kind": kind,
            "args": args,
            "result": result,
            "cost": cost,
        })

    # Self-combine: a ⋆ a
    for a in s:
        if len(observations) >= max_obs or world.remaining < world.combine_cost:
            break
        try:
            result = world.combine(a, a)
            _record("combine", (a, a), result, world.combine_cost)
        except BudgetExhausted:
            break

    # Cross-combine: a ⋆ b (a ≠ b)
    for i, a in enumerate(s):
        for j, b in enumerate(s):
            if i == j:
                continue
            if len(observations) >= max_obs or world.remaining < world.combine_cost:
                break
            try:
                result = world.combine(a, b)
                _record("combine", (a, b), result, world.combine_cost)
            except BudgetExhausted:
                break

    # Transform: f(a)
    for a in s:
        if len(observations) >= max_obs or world.remaining < world.transform_cost:
            break
        try:
            result = world.transform(a)
            _record("transform", (a,), result, world.transform_cost)
        except BudgetExhausted:
            break

    return observations


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    world: HiddenLawWorld,
    client: ModelClient,
    *,
    max_obs: int = 200,
) -> EpisodeResult:
    """Run one Hidden-Law Lab episode using the given model client.

    1. Sample observations from *world* (deterministic, no strategy yet).
    2. Build a prompt from the transcript.
    3. Call the model.
    4. Parse and validate the claims.
    5. Score against ground truth.

    Never raises; errors are captured in ``EpisodeResult.error``.
    """
    truth = world.ground_truth()
    base_result = EpisodeResult(
        seed=world.seed,
        family=world.family,
        n=len(world.symbols),
        claims={name: False for name in LAW_NAMES},
        truth=truth,
        correct=0,
    )

    # Gather observations
    observations = _sample_observations(world, max_obs=max_obs)
    base_result.observations_used = len(observations)
    base_result.budget_spent = world.spent

    # Build prompt
    messages = build_messages(
        symbols=world.symbols,
        observations=observations,
        budget_remaining=world.remaining,
    )

    # Call model
    t0 = time.monotonic()
    try:
        raw = client.chat_json(messages)
    except ModelClientError as exc:
        base_result.error = str(exc)
        base_result.latency_s = time.monotonic() - t0
        return base_result
    base_result.latency_s = time.monotonic() - t0

    # Parse claims
    try:
        claims = parse_claims(raw)
    except ValueError as exc:
        base_result.error = f"ParseError: {exc}"
        return base_result

    # Score
    correct = sum(claims.get(k) == v for k, v in truth.items())
    base_result.claims = claims
    base_result.correct = correct
    return base_result
