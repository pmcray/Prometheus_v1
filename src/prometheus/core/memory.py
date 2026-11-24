"""Good's MDL & assembly-style structural memory with probabilistic regeneration.

Meaning as compression: recurring algebraic patterns observed across episodes
are consolidated into symbol-agnostic *assemblies* only when doing so lowers
total Minimum Description Length (MDL) in bits.

Recall as probabilistic regeneration: during an episode, assemblies activate in
proportion to their Hebbian salience and how well they compress the current
belief/features, regenerating a blended prior over the 120 logically consistent
law profiles.  Because recall acts strictly as a soft Bayesian prior, concrete
observations always override stale or mismatched memories.

Integrity guarantee: ``AssemblyMemory`` never receives ground truth or family
names — only the agent's own inferred beliefs and symbol-agnostic invariants.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..lab.laws import LAW_NAMES
from .blanket import LawBelief
from .inference import (
    NUM_VALID_PROFILES,
    PROFILE_KEYS,
    UNIFORM_JOINT_PRIOR,
    VALID_LAW_PROFILES,
)

_LN2 = math.log(2.0)
_FEATURE_KEYS: tuple[str, ...] = (
    "diag_fixpoint_rate",
    "f_fixpoint_rate",
    "row_uniqueness",
    "identity_prob",
)


@dataclass(frozen=True)
class MemoryParams:
    """Mutable memory hyperparameters (loaded from harness/memory_params/)."""

    prior_blend: float = 0.35
    recall_temperature: float = 2.0
    spawn_mdl_threshold_bits: float = 1.25
    merge_kl_threshold: float = 0.25
    hebbian_decay: float = 0.985
    max_assemblies: int = 24
    min_confidence_to_learn: float = 0.72
    model_cost_bits: float = 2.5


@dataclass
class Assembly:
    """One compressed structural motif learned purely from experience."""

    assembly_id: str
    created_episode: int
    uses: int = 1
    salience: float = 1.0
    law_alpha: list[float] = field(default_factory=lambda: [1.0] * len(LAW_NAMES))
    law_beta: list[float] = field(default_factory=lambda: [1.0] * len(LAW_NAMES))
    profile_counts: dict[str, float] = field(default_factory=dict)
    size_counts: dict[str, float] = field(default_factory=dict)
    feature_means: dict[str, float] = field(
        default_factory=lambda: {k: 0.5 for k in _FEATURE_KEYS}
    )

    def law_probs(self) -> list[float]:
        return [
            self.law_alpha[k] / (self.law_alpha[k] + self.law_beta[k])
            for k in range(len(LAW_NAMES))
        ]

    def regenerated_joint_prior(self) -> tuple[float, ...]:
        """Probabilistically regenerate a distribution over the 120 valid law profiles."""
        lp = self.law_probs()
        total_prof = sum(self.profile_counts.values())
        probs: list[float] = []
        for idx, prof in enumerate(VALID_LAW_PROFILES):
            # Factored Beta-Bernoulli likelihood across the 8 laws
            p_fact = 1.0
            for k in range(len(LAW_NAMES)):
                pk = min(max(lp[k], 1e-4), 1.0 - 1e-4)
                p_fact *= pk if prof[k] else (1.0 - pk)
            # Blend with empirical joint profile density (captures higher-order correlations)
            if total_prof > 0.0:
                key = PROFILE_KEYS[idx]
                p_emp = (self.profile_counts.get(key, 0.0) + 0.05) / (
                    total_prof + 0.05 * NUM_VALID_PROFILES
                )
                p = 0.5 * p_fact + 0.5 * p_emp
            else:
                p = p_fact
            probs.append(max(p, 1e-12))
        s = sum(probs)
        return tuple(p / s for p in probs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Assembly":
        return cls(
            assembly_id=str(data["assembly_id"]),
            created_episode=int(data["created_episode"]),
            uses=int(data.get("uses", 1)),
            salience=float(data.get("salience", 1.0)),
            law_alpha=[float(x) for x in data["law_alpha"]],
            law_beta=[float(x) for x in data["law_beta"]],
            profile_counts={str(k): float(v) for k, v in data.get("profile_counts", {}).items()},
            size_counts={str(k): float(v) for k, v in data.get("size_counts", {}).items()},
            feature_means={str(k): float(v) for k, v in data.get("feature_means", {}).items()},
        )


@dataclass(frozen=True)
class RecallResult:
    """Output of probabilistic memory recall for the current episode."""

    activations: dict[str, float]
    prior_joint: tuple[float, ...]
    compression_gain_bits: float


@dataclass(frozen=True)
class ConsolidationReport:
    """Summary of post-episode memory consolidation."""

    action: str  # "skipped_low_confidence" | "updated" | "spawned"
    assembly_id: str | None
    compression_gain_bits: float
    merged_count: int = 0
    pruned_count: int = 0


def _bits_cross_entropy(q: tuple[float, ...], p: tuple[float, ...]) -> float:
    total = 0.0
    for q_i, p_i in zip(q, p):
        if q_i > 1e-15:
            total -= q_i * math.log(max(p_i, 1e-15))
    return total / _LN2


def _kl_bits(p: tuple[float, ...], q: tuple[float, ...]) -> float:
    total = 0.0
    for pi, qi in zip(p, q):
        if pi > 1e-15:
            total += pi * math.log(pi / max(qi, 1e-15))
    return max(0.0, total / _LN2)


class AssemblyMemory:
    """MDL-driven assembly memory supporting recall, consolidation, and checkpointing."""

    def __init__(self, params: MemoryParams | None = None) -> None:
        self.params = params or MemoryParams()
        self.assemblies: list[Assembly] = []
        self.episodes_consolidated: int = 0
        self._next_id: int = 1

    # ----- MDL code lengths -------------------------------------------------

    def null_code_length_bits(self) -> float:
        """Description length (bits) of an episode under the uniform prior over valid profiles."""
        return math.log(NUM_VALID_PROFILES) / _LN2

    def data_code_length_bits(
        self,
        assembly: Assembly,
        belief: LawBelief,
        features: dict[str, float],
        n_symbols: int,
    ) -> float:
        """Negative log-likelihood L(D | A_m) in bits for an episode under ``assembly``."""
        prior = assembly.regenerated_joint_prior()
        law_bits = _bits_cross_entropy(belief.joint_probs, prior)

        # Structural feature coding penalty (Gaussian with fixed sigma = 0.25)
        feat_bits = 0.0
        sigma2 = 0.25 * 0.25
        for key in _FEATURE_KEYS:
            if key in features and key in assembly.feature_means:
                diff = features[key] - assembly.feature_means[key]
                feat_bits += (0.5 * (diff * diff) / sigma2) / _LN2

        # Carrier-size mild coding term (Laplace-smoothed over sizes 3..7)
        s_tot = sum(assembly.size_counts.values())
        p_size = (assembly.size_counts.get(str(n_symbols), 0.0) + 0.5) / (s_tot + 2.5)
        size_bits = -math.log(max(p_size, 1e-6)) / _LN2
        # Baseline size bits is log2(5); only charge relative excess/savings
        rel_size_bits = size_bits - (math.log(5.0) / _LN2)

        return max(0.0, law_bits + 0.25 * feat_bits + 0.25 * rel_size_bits)

    # ----- Recall as probabilistic regeneration -----------------------------

    def recall(
        self,
        belief: LawBelief,
        features: dict[str, float],
        n_symbols: int,
    ) -> RecallResult:
        """Activate matching assemblies and probabilistically regenerate a joint prior."""
        if not self.assemblies:
            return RecallResult(
                activations={},
                prior_joint=UNIFORM_JOINT_PRIOR,
                compression_gain_bits=0.0,
            )

        temp = max(self.params.recall_temperature, 1e-3)
        null_bits = self.null_code_length_bits()
        log_weights: list[float] = []
        code_lengths: list[float] = []
        for a in self.assemblies:
            dl = self.data_code_length_bits(a, belief, features, n_symbols)
            code_lengths.append(dl)
            sal = max(a.salience, 1e-6)
            log_weights.append(math.log(sal) - (dl * _LN2) / temp)

        m = max(log_weights)
        exps = [math.exp(v - m) for v in log_weights]
        s = sum(exps)
        probs = [e / s for e in exps]
        activations = {a.assembly_id: p for a, p in zip(self.assemblies, probs)}

        # Mixture regeneration across activated assemblies
        mix = [0.0] * NUM_VALID_PROFILES
        for a, w in zip(self.assemblies, probs):
            if w < 1e-6:
                continue
            ap = a.regenerated_joint_prior()
            for idx in range(NUM_VALID_PROFILES):
                mix[idx] += w * ap[idx]

        alpha = min(max(self.params.prior_blend, 0.0), 1.0)
        blended = tuple(
            (1.0 - alpha) * UNIFORM_JOINT_PRIOR[i] + alpha * mix[i]
            for i in range(NUM_VALID_PROFILES)
        )
        # Normalize defensively
        bs = sum(blended)
        blended = tuple(p / bs for p in blended)

        expected_dl = sum(w * dl for w, dl in zip(probs, code_lengths))
        gain_bits = max(0.0, null_bits - expected_dl)
        return RecallResult(
            activations=activations,
            prior_joint=blended,
            compression_gain_bits=gain_bits,
        )

    # ----- Consolidation (MDL spawn / update / merge / prune) ---------------

    def _absorb(
        self,
        assembly: Assembly,
        belief: LawBelief,
        features: dict[str, float],
        n_symbols: int,
    ) -> None:
        weight = max(0.2, belief.mean_confidence)
        for k, name in enumerate(LAW_NAMES):
            p = belief.law_probs[name]
            assembly.law_alpha[k] += weight * p
            assembly.law_beta[k] += weight * (1.0 - p)
        best_idx = max(range(NUM_VALID_PROFILES), key=lambda i: belief.joint_probs[i])
        key = PROFILE_KEYS[best_idx]
        assembly.profile_counts[key] = assembly.profile_counts.get(key, 0.0) + weight
        sz = str(n_symbols)
        assembly.size_counts[sz] = assembly.size_counts.get(sz, 0.0) + 1.0
        assembly.uses += 1
        assembly.salience += weight
        lr = 1.0 / assembly.uses
        for fk in _FEATURE_KEYS:
            if fk in features:
                old = assembly.feature_means.get(fk, 0.5)
                assembly.feature_means[fk] = (1.0 - lr) * old + lr * features[fk]

    def _spawn(
        self,
        belief: LawBelief,
        features: dict[str, float],
        n_symbols: int,
        episode_idx: int,
    ) -> Assembly:
        aid = f"asm-{self._next_id:03d}"
        self._next_id += 1
        asm = Assembly(
            assembly_id=aid,
            created_episode=episode_idx,
            uses=0,
            salience=0.5,
            feature_means={fk: features.get(fk, 0.5) for fk in _FEATURE_KEYS},
        )
        self._absorb(asm, belief, features, n_symbols)
        self.assemblies.append(asm)
        return asm

    def _merge_similar(self) -> int:
        thresh = self.params.merge_kl_threshold
        if thresh <= 0.0 or len(self.assemblies) < 2:
            return 0
        merged = 0
        i = 0
        while i < len(self.assemblies):
            j = i + 1
            while j < len(self.assemblies):
                ai, aj = self.assemblies[i], self.assemblies[j]
                pi = ai.regenerated_joint_prior()
                pj = aj.regenerated_joint_prior()
                sym_kl = 0.5 * (_kl_bits(pi, pj) + _kl_bits(pj, pi))
                if sym_kl <= thresh:
                    # Merge aj into ai (preserve higher-use ID)
                    if aj.uses > ai.uses:
                        ai, aj = aj, ai
                        self.assemblies[i], self.assemblies[j] = ai, aj
                    tot_uses = ai.uses + aj.uses
                    for k in range(len(LAW_NAMES)):
                        ai.law_alpha[k] += aj.law_alpha[k] - 1.0
                        ai.law_beta[k] += aj.law_beta[k] - 1.0
                    for pk, pv in aj.profile_counts.items():
                        ai.profile_counts[pk] = ai.profile_counts.get(pk, 0.0) + pv
                    for sk, sv in aj.size_counts.items():
                        ai.size_counts[sk] = ai.size_counts.get(sk, 0.0) + sv
                    for fk in _FEATURE_KEYS:
                        ai.feature_means[fk] = (
                            ai.feature_means.get(fk, 0.5) * ai.uses
                            + aj.feature_means.get(fk, 0.5) * aj.uses
                        ) / max(1, tot_uses)
                    ai.uses = tot_uses
                    ai.salience += aj.salience
                    self.assemblies.pop(j)
                    merged += 1
                    continue
                j += 1
            i += 1
        return merged

    def _prune(self) -> int:
        cap = max(1, self.params.max_assemblies)
        if len(self.assemblies) <= cap:
            return 0
        self.assemblies.sort(key=lambda a: (a.salience, a.uses), reverse=True)
        removed = len(self.assemblies) - cap
        self.assemblies = self.assemblies[:cap]
        return removed

    def consolidate(
        self,
        belief: LawBelief,
        features: dict[str, float],
        n_symbols: int,
        *,
        episode_idx: int = 0,
    ) -> ConsolidationReport:
        """Update or spawn an assembly based on MDL compression gain."""
        if belief.mean_confidence < self.params.min_confidence_to_learn:
            return ConsolidationReport(
                action="skipped_low_confidence",
                assembly_id=None,
                compression_gain_bits=0.0,
            )

        # Gentle Hebbian decay across existing assemblies
        decay = self.params.hebbian_decay
        for a in self.assemblies:
            a.salience = max(0.05, a.salience * decay)

        null_bits = self.null_code_length_bits()
        best_asm: Assembly | None = None
        best_dl = float("inf")
        for a in self.assemblies:
            dl = self.data_code_length_bits(a, belief, features, n_symbols)
            if dl < best_dl:
                best_dl = dl
                best_asm = a

        # Evaluate MDL cost of spawning a fresh assembly vs reusing the best existing one
        # New assembly compresses this episode near its own entropy + model_cost_bits
        spawn_dl = belief.entropy_bits + self.params.model_cost_bits
        should_spawn = (best_asm is None) or (
            (best_dl - spawn_dl) >= self.params.spawn_mdl_threshold_bits
        )

        if should_spawn:
            created = self._spawn(belief, features, n_symbols, episode_idx)
            gain = max(0.0, null_bits - self.data_code_length_bits(created, belief, features, n_symbols))
            action = "spawned"
            aid = created.assembly_id
        else:
            assert best_asm is not None
            self._absorb(best_asm, belief, features, n_symbols)
            gain = max(0.0, null_bits - best_dl)
            action = "updated"
            aid = best_asm.assembly_id

        self.episodes_consolidated += 1
        merged = self._merge_similar()
        pruned = self._prune()
        return ConsolidationReport(
            action=action,
            assembly_id=aid,
            compression_gain_bits=gain,
            merged_count=merged,
            pruned_count=pruned,
        )

    # ----- Serialization & atomic Drive checkpointing -----------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "episodes_consolidated": self.episodes_consolidated,
            "next_id": self._next_id,
            "params": asdict(self.params),
            "assemblies": [a.to_dict() for a in self.assemblies],
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        params_override: MemoryParams | None = None,
    ) -> "AssemblyMemory":
        raw_params = data.get("params") or {}
        if params_override is not None:
            params = params_override
        else:
            params = MemoryParams(**{k: raw_params[k] for k in raw_params if hasattr(MemoryParams, k)})
        mem = cls(params=params)
        mem.episodes_consolidated = int(data.get("episodes_consolidated", 0))
        mem._next_id = int(data.get("next_id", len(data.get("assemblies", [])) + 1))
        mem.assemblies = [Assembly.from_dict(a) for a in data.get("assemblies", [])]
        return mem

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f".{p.name}.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, sort_keys=True, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        params_override: MemoryParams | None = None,
    ) -> "AssemblyMemory":
        p = Path(path)
        if not p.is_file():
            return cls(params=params_override)
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_dict(data, params_override=params_override)
