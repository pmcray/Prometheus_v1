"""Hofstadter structural analogy and cross-world transfer (Phase 2).

Implements :class:`AnalogyTransferAdapter` using purely symbol-agnostic
structural invariants inferred across the Markov blanket:

- Every Hidden-Law Lab world uses a freshly permuted opaque symbol alphabet
  (``ka, lu, mo, ...``), so raw symbol-level memorization cannot transfer.
- ``StructuralAnalogyTransfer`` extracts a symbol-invariant
  :class:`StructuralDescriptor` from the current observations and belief
  (diagonal fixpoint rate, commutativity agreement, Latin-square row
  uniqueness, unary fixpoint/involution rates, candidate identity mass),
  retrieves the closest consolidated :class:`Assembly` motif from Good's MDL
  memory, regenerates a soft Bayesian prior over the 120 valid law profiles,
  and proposes high-leverage discriminative probes on the current world's
  symbols to confirm or falsify the analogy rapidly.
- **Anti-dogmatism / law-shift guard:** if recent observations exhibit a
  Bayesian surprise spike (e.g. a Phase 3 law shift), the adapter immediately
  retracts transfer bias toward uniform so stale analogies never trap the
  agent in obsolete habits.

Integrity guarantee: never accesses ground truth, family names, or raw tables.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

from ..lab.laws import LAW_NAMES
from .blanket import ExperimentAction, LawBelief, Observation
from .inference import NUM_VALID_PROFILES, UNIFORM_JOINT_PRIOR
from .interfaces import TransferHint
from .memory import Assembly

_LN2 = math.log(2.0)
_LAW_INDEX = {name: i for i, name in enumerate(LAW_NAMES)}


@dataclass(frozen=True)
class TransferParams:
    """Mutable Hofstadter transfer hyperparameters (``harness/memory_params/transfer_params.json``)."""

    enabled: bool = True
    min_observations: int = 3
    min_confidence: float = 0.45
    max_prior_blend: float = 0.35
    temperature: float = 1.5
    surprise_veto_threshold: float = 5.0
    enable_probe_suggestions: bool = True
    max_probe_steps: int = 6


@dataclass(frozen=True)
class StructuralDescriptor:
    """Symbol-agnostic structural invariants extracted solely from observations + belief."""

    n_symbols: int
    diag_fixpoint_rate: float
    diag_count: int
    f_fixpoint_rate: float
    f_count: int
    comm_agreement_rate: float
    comm_pairs: int
    row_uniqueness: float
    row_count: int
    f_involution_rate: float
    f_inv_pairs: int
    identity_prob: float
    recent_max_surprise: float


def _majority_tables(
    observations: Sequence[Observation],
) -> tuple[dict[tuple[str, str], str], dict[str, str]]:
    """Build noise-resistant majority-vote partial tables from observations."""
    op_votes: dict[tuple[str, str], Counter[str]] = {}
    f_votes: dict[str, Counter[str]] = {}
    for obs in observations:
        if obs.action.kind == "combine":
            a, b = obs.action.args
            op_votes.setdefault((a, b), Counter())[obs.result] += 1
        else:
            (a,) = obs.action.args
            f_votes.setdefault(a, Counter())[obs.result] += 1
    op_map = {k: cnt.most_common(1)[0][0] for k, cnt in op_votes.items()}
    f_map = {k: cnt.most_common(1)[0][0] for k, cnt in f_votes.items()}
    return op_map, f_map


def extract_descriptor(
    belief: LawBelief,
    observations: Sequence[Observation],
) -> StructuralDescriptor:
    """Extract purely symbol-agnostic structural invariants from ``(belief, observations)``."""
    op_map, f_map = _majority_tables(observations)

    if belief.identity_probs:
        n_symbols = len(belief.identity_probs)
    else:
        seen: set[str] = set()
        for obs in observations:
            seen.update(obs.action.args)
            seen.add(obs.result)
        n_symbols = max(2, len(seen))

    # Diagonal fixpoints: op(a, a) == a
    diags = [(a, res) for (a, b), res in op_map.items() if a == b]
    diag_count = len(diags)
    diag_fix = (
        sum(1 for a, res in diags if res == a) / diag_count
        if diag_count > 0
        else (1.0 / n_symbols)
    )

    # Unary fixpoints: f(a) == a
    f_count = len(f_map)
    f_fix = (
        sum(1 for a, res in f_map.items() if res == a) / f_count
        if f_count > 0
        else (1.0 / n_symbols)
    )

    # Commutative pair agreement: op(a, b) == op(b, a) for unordered pairs a != b
    seen_pairs: set[tuple[str, str]] = set()
    comm_hits = 0
    comm_pairs = 0
    for (a, b), res_ab in op_map.items():
        if a == b:
            continue
        pair = (a, b) if a < b else (b, a)
        if pair in seen_pairs:
            continue
        if (b, a) in op_map:
            seen_pairs.add(pair)
            comm_pairs += 1
            if op_map[(b, a)] == res_ab:
                comm_hits += 1
    comm_rate = comm_hits / comm_pairs if comm_pairs > 0 else 0.5

    # Row uniqueness (Latin-square / cancellation signal)
    rows: dict[str, list[str]] = {}
    for (a, _b), res in op_map.items():
        rows.setdefault(a, []).append(res)
    row_scores = [len(set(vals)) / len(vals) for vals in rows.values() if len(vals) >= 2]
    row_count = len(row_scores)
    row_uniq = sum(row_scores) / row_count if row_count > 0 else 0.5

    # Unary involution agreement: f(f(a)) == a
    f_inv_hits = 0
    f_inv_pairs = 0
    for a, fa in f_map.items():
        if fa in f_map:
            f_inv_pairs += 1
            if f_map[fa] == a:
                f_inv_hits += 1
    f_inv_rate = f_inv_hits / f_inv_pairs if f_inv_pairs > 0 else 0.5

    recent = observations[-4:]
    recent_max_surp = max((o.surprise for o in recent), default=0.0)

    return StructuralDescriptor(
        n_symbols=n_symbols,
        diag_fixpoint_rate=diag_fix,
        diag_count=diag_count,
        f_fixpoint_rate=f_fix,
        f_count=f_count,
        comm_agreement_rate=comm_rate,
        comm_pairs=comm_pairs,
        row_uniqueness=row_uniq,
        row_count=row_count,
        f_involution_rate=f_inv_rate,
        f_inv_pairs=f_inv_pairs,
        identity_prob=float(belief.law_probs.get("has_identity", 0.5)),
        recent_max_surprise=recent_max_surp,
    )


def _cross_entropy_bits(q: tuple[float, ...], p: tuple[float, ...]) -> float:
    total = 0.0
    for qi, pi in zip(q, p):
        if qi > 1e-15:
            total -= qi * math.log(max(pi, 1e-15))
    return total / _LN2


class StructuralAnalogyTransfer:
    """Hofstadter Slipnet-style structural analogy adapter across opaque symbol sets."""

    def __init__(
        self,
        params: TransferParams | None = None,
        *,
        combine_cost: float = 1.0,
        transform_cost: float = 0.5,
    ) -> None:
        self.params = params or TransferParams()
        self._combine_cost = float(combine_cost)
        self._transform_cost = float(transform_cost)
        self.hints_emitted: int = 0
        self.surprise_vetoes: int = 0
        self.source_counts: dict[str, int] = {}

    def _assembly_fit(
        self,
        asm: Assembly,
        belief: LawBelief,
        desc: StructuralDescriptor,
    ) -> tuple[float, float]:
        """Return ``(log_score, abs_fit_01)`` measuring structural analogy to ``asm``."""
        prior = asm.regenerated_joint_prior()
        null_bits = math.log(NUM_VALID_PROFILES) / _LN2
        ce_bits = _cross_entropy_bits(belief.joint_probs, prior)
        # Positive bits saved relative to uniform (clamped for stability)
        belief_gain = null_bits - ce_bits

        # Evidence-weighted invariant distance against assembly structural features
        weighted_sq = 0.0
        total_w = 0.0
        fm = asm.feature_means
        if desc.diag_count > 0 and "diag_fixpoint_rate" in fm:
            w = min(1.0, desc.diag_count / 3.0)
            diff = desc.diag_fixpoint_rate - fm["diag_fixpoint_rate"]
            weighted_sq += w * diff * diff
            total_w += w
        if desc.f_count > 0 and "f_fixpoint_rate" in fm:
            w = min(1.0, desc.f_count / 3.0)
            diff = desc.f_fixpoint_rate - fm["f_fixpoint_rate"]
            weighted_sq += w * diff * diff
            total_w += w
        if desc.row_count > 0 and "row_uniqueness" in fm:
            w = min(1.0, desc.row_count / 2.0)
            diff = desc.row_uniqueness - fm["row_uniqueness"]
            weighted_sq += w * diff * diff
            total_w += w
        if "identity_prob" in fm:
            w = 0.75
            diff = desc.identity_prob - fm["identity_prob"]
            weighted_sq += w * diff * diff
            total_w += w

        # Extra check against assembly's learned law probabilities for observed comm/inv signals
        lp = asm.law_probs()
        if desc.comm_pairs > 0:
            w = min(1.0, desc.comm_pairs / 2.0)
            diff = desc.comm_agreement_rate - lp[_LAW_INDEX["commutative"]]
            weighted_sq += w * diff * diff
            total_w += w
        if desc.f_inv_pairs > 0:
            w = min(1.0, desc.f_inv_pairs / 2.0)
            diff = desc.f_involution_rate - lp[_LAW_INDEX["f_involution"]]
            weighted_sq += w * diff * diff
            total_w += w

        mean_sq = weighted_sq / max(total_w, 1e-6)
        feat_sim = math.exp(-mean_sq / (2.0 * 0.25 * 0.25))  # in (0, 1]

        # Carrier-size compatibility (mild preference, never hard rejection)
        s_tot = sum(asm.size_counts.values())
        p_size = (asm.size_counts.get(str(desc.n_symbols), 0.0) + 0.5) / (s_tot + 2.5)
        size_bonus = 0.2 * math.log(max(p_size * 5.0, 0.2))

        sal_bonus = 0.15 * math.log(max(asm.salience, 0.05) + 1.0)
        temp = max(self.params.temperature, 0.1)
        log_score = (belief_gain * _LN2 - 2.0 * mean_sq / (0.25 * 0.25)) / temp + size_bonus + sal_bonus

        # Absolute fit combines structural feature similarity and belief agreement
        law_l1 = sum(abs(belief.law_probs[name] - lp[k]) for k, name in enumerate(LAW_NAMES)) / len(LAW_NAMES)
        law_sim = max(0.0, 1.0 - 2.0 * law_l1)
        abs_fit = 0.55 * feat_sim + 0.45 * law_sim
        return (log_score, max(0.0, min(1.0, abs_fit)))

    def _suggest_discriminative_probes(
        self,
        asm: Assembly,
        belief: LawBelief,
        observations: Sequence[Observation],
    ) -> tuple[ExperimentAction, ...]:
        """Propose symbol-grounded probes that rapidly confirm or falsify the analogy."""
        if not self.params.enable_probe_suggestions:
            return ()
        if len(observations) > self.params.min_observations + self.params.max_probe_steps:
            return ()

        op_map, f_map = _majority_tables(observations)
        symbols = list(belief.identity_probs.keys()) if belief.identity_probs else sorted(
            {s for o in observations for s in (*o.action.args, o.result)}
        )
        if not symbols:
            return ()

        # Determine action costs from recent observations if available
        c_cost = self._combine_cost
        t_cost = self._transform_cost
        for o in observations:
            if o.action.kind == "combine":
                c_cost = o.action.cost
            elif o.action.kind == "transform":
                t_cost = o.action.cost

        asm_lp = {name: asm.law_probs()[k] for k, name in enumerate(LAW_NAMES)}
        # Rank laws where the analogy makes a strong claim yet current belief is still uncertain
        uncertain_laws: list[tuple[float, str]] = []
        for name in LAW_NAMES:
            p_cur = belief.law_probs[name]
            p_asm = asm_lp[name]
            if 0.15 < p_cur < 0.85 and abs(p_asm - 0.5) >= 0.28:
                # Prefer laws closest to 0.5 uncertainty weighted by analogy conviction
                priority = (1.0 - 2.0 * abs(p_cur - 0.5)) * abs(p_asm - 0.5)
                uncertain_laws.append((priority, name))
        uncertain_laws.sort(reverse=True)

        suggestions: list[ExperimentAction] = []
        seen_keys: set[tuple[str, tuple[str, ...]]] = set()

        def _add(kind: str, args: tuple[str, ...], cost: float) -> None:
            key = (kind, args)
            if key not in seen_keys:
                seen_keys.add(key)
                suggestions.append(ExperimentAction(kind, args, cost))  # type: ignore[arg-type]

        for _prio, law in uncertain_laws[:3]:
            if law == "f_involution":
                # Test f(f(a)) where f(a) is already observed
                for a, fa in sorted(f_map.items()):
                    if fa not in f_map:
                        _add("transform", (fa,), t_cost)
                        break
                else:
                    for s in symbols:
                        if s not in f_map:
                            _add("transform", (s,), t_cost)
                            break
            elif law == "f_bijective":
                for s in symbols:
                    if s not in f_map:
                        _add("transform", (s,), t_cost)
                        break
            elif law == "commutative":
                # Mirror an already-observed off-diagonal pair (a, b) -> test (b, a)
                for (a, b) in sorted(op_map):
                    if a != b and (b, a) not in op_map:
                        _add("combine", (b, a), c_cost)
                        break
            elif law == "idempotent":
                for s in symbols:
                    if (s, s) not in op_map:
                        _add("combine", (s, s), c_cost)
                        break
            elif law in ("has_identity", "has_inverses"):
                if belief.identity_probs:
                    cand_e = max(belief.identity_probs, key=lambda s: belief.identity_probs[s])
                    for x in symbols:
                        if (cand_e, x) not in op_map:
                            _add("combine", (cand_e, x), c_cost)
                            break
                        if (x, cand_e) not in op_map:
                            _add("combine", (x, cand_e), c_cost)
                            break
            elif law == "f_homomorphism":
                # Pick a pair (a, b) where op(a, b)=c, f(a), f(b) exist -> test f(c) or op(f(a), f(b))
                for (a, b), c in sorted(op_map.items()):
                    if a in f_map and b in f_map:
                        if c not in f_map:
                            _add("transform", (c,), t_cost)
                            break
                        fa, fb = f_map[a], f_map[b]
                        if (fa, fb) not in op_map:
                            _add("combine", (fa, fb), c_cost)
                            break
            if len(suggestions) >= 3:
                break

        return tuple(suggestions)

    def propose_transfer(
        self,
        belief: LawBelief,
        observations: Sequence[Observation],
        assemblies: Sequence[Any],
    ) -> TransferHint | None:
        """Retrieve an analogous structural motif and return a soft prior + probe hint."""
        if not self.params.enabled:
            return None
        if not assemblies or len(observations) < self.params.min_observations:
            return None

        desc = extract_descriptor(belief, observations)

        # Anti-dogmatism / law-shift guard: retract transfer bias on surprise spikes
        if desc.recent_max_surprise > self.params.surprise_veto_threshold:
            self.surprise_vetoes += 1
            return TransferHint(
                prior_joint=UNIFORM_JOINT_PRIOR,
                suggested_actions=(),
                source_assembly_id=None,
                confidence=0.0,
            )

        # Transfer is for sparse-data bootstrapping: once substantial direct
        # observations exist on the current world, gracefully release the prior
        # back to uniform so mature observations dominate without circular bias.
        horizon = self.params.min_observations + max(2 * self.params.max_probe_steps, 8)
        if len(observations) > horizon:
            return TransferHint(
                prior_joint=UNIFORM_JOINT_PRIOR,
                suggested_actions=(),
                source_assembly_id=None,
                confidence=0.0,
            )

        valid_asms: list[Assembly] = [a for a in assemblies if isinstance(a, Assembly)]
        if not valid_asms:
            return None

        log_scores: list[float] = []
        abs_fits: list[float] = []
        for a in valid_asms:
            ls, af = self._assembly_fit(a, belief, desc)
            log_scores.append(ls)
            abs_fits.append(af)

        max_ls = max(log_scores)
        exps = [math.exp(max(-50.0, ls - max_ls)) for ls in log_scores]
        s_exp = sum(exps)
        weights = [e / s_exp for e in exps]

        best_idx = max(range(len(valid_asms)), key=lambda i: (weights[i], abs_fits[i]))
        best_asm = valid_asms[best_idx]
        # Confidence requires BOTH strong absolute structural fit and clear selection weight
        # (single-assembly memory gets full selection weight 1.0, gated purely by abs_fit).
        sharpness = weights[best_idx] if len(valid_asms) > 1 else 1.0
        confidence = abs_fits[best_idx] * (0.5 + 0.5 * sharpness)

        if confidence < self.params.min_confidence:
            return None

        # Mixture prior across matched assemblies, square-root tempered and
        # annealed as direct observations accumulate so a single sharp assembly
        # cannot inject an extreme prior-odds spike over noisy early evidence.
        mix = [0.0] * NUM_VALID_PROFILES
        for a, w in zip(valid_asms, weights):
            if w < 1e-4:
                continue
            ap = a.regenerated_joint_prior()
            for idx in range(NUM_VALID_PROFILES):
                mix[idx] += w * ap[idx]

        tempered = [math.sqrt(max(p, 1e-6)) for p in mix]
        ts = sum(tempered)
        tempered = [p / ts for p in tempered]

        excess_obs = max(0, len(observations) - self.params.min_observations)
        anneal = max(0.0, 1.0 - excess_obs / max(float(horizon - self.params.min_observations + 1), 1.0))
        alpha = min(
            max(self.params.max_prior_blend * confidence * anneal, 0.0),
            self.params.max_prior_blend,
        )
        blended = tuple(
            (1.0 - alpha) * UNIFORM_JOINT_PRIOR[i] + alpha * tempered[i]
            for i in range(NUM_VALID_PROFILES)
        )
        bs = sum(blended)
        blended = tuple(p / bs for p in blended)

        probes = self._suggest_discriminative_probes(best_asm, belief, observations)

        self.hints_emitted += 1
        self.source_counts[best_asm.assembly_id] = self.source_counts.get(best_asm.assembly_id, 0) + 1
        return TransferHint(
            prior_joint=blended,
            suggested_actions=probes,
            source_assembly_id=best_asm.assembly_id,
            confidence=round(confidence, 4),
        )

    def telemetry(self) -> dict[str, Any]:
        top_sources = sorted(self.source_counts.items(), key=lambda kv: -kv[1])[:5]
        return {
            "enabled": self.params.enabled,
            "hints_emitted": self.hints_emitted,
            "surprise_vetoes": self.surprise_vetoes,
            "top_sources": dict(top_sources),
        }
