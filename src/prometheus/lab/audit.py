"""Frozen audit suite scoring.

The runtime submits claims {seed: {law_name: bool}} for every audit world. The
governor rebuilds each world from its seed and scores the claims itself, so a
variant cannot improve its audit score by misreporting it; only by actually
getting more laws right.
"""

from __future__ import annotations

from typing import Any, Iterable

from .laws import LAW_NAMES
from .world import HiddenLawWorld


class AuditError(ValueError):
    pass


def score_claims(claims: Any, seeds: Iterable[int]) -> float:
    """Fraction of (world, law) pairs claimed correctly. Missing claims count as wrong."""
    if not isinstance(claims, dict):
        raise AuditError("claims must be an object keyed by world seed")
    seeds = list(seeds)
    if not seeds:
        raise AuditError("audit suite is empty")
    correct = 0
    for seed in seeds:
        truth = HiddenLawWorld(seed=seed).ground_truth()
        world_claims = claims.get(str(seed), {})
        if not isinstance(world_claims, dict):
            raise AuditError(f"claims for world {seed} must be an object")
        for law in LAW_NAMES:
            claim = world_claims.get(law)
            if isinstance(claim, bool) and claim == truth[law]:
                correct += 1
    return correct / (len(seeds) * len(LAW_NAMES))
