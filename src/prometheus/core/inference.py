"""Pure-Python exact Bayesian inference and Friston Expected Free Energy (EFE).

Implements :class:`InferenceBackend` with zero third-party dependencies:

1. **Exact observation channel**: maintains sufficient observation counts per
   cell of the hidden binary table ``op[i][j]`` and unary map ``f[i]`` under
   the exact noise channel ``P(y | z) = (1 - eps) + eps/n`` if ``y == z`` else
   ``eps/n`` (with optional exponential forgetting ``forgetting_factor`` for
   Phase 3 law-shift readiness).
2. **Exact Bayesian surprise**: every observation produces its negative log
   predictive probability ``S_t = -log P(y_t | D_{1:t-1})``.
3. **Zero-truncated hierarchical law Bayes factors**: compares universal
   satisfaction ``H1`` (all ``K`` witnesses hold, ``rho = 0``) against
   falsification ``H0`` (at least one of the ``K`` witnesses violates,
   conditioned on ``|S| >= 1`` using exact match ``bf+`` and exact violation
   ``bf-`` likelihoods), with composite-likelihood overlap tempering.
4. **Exact joint belief over logically valid law profiles**: maintains ``Q(V)``
   over all 120 logically consistent 8-law truth assignments (factored as
   20 op-law profiles x 6 f-law profiles, enforcing pure algebraic tautologies:
   ``has_inverses => has_identity``, ``f_involution => f_bijective``, and
   ``not (idempotent and has_inverses)`` for ``n >= 2``).
5. **Fast incremental Friston Expected Free Energy**: caches per-witness
   accumulators, exact collapsed 20-state/6-state marginal entropies, and
   symmetric unobserved-cell deduplication for millisecond step latency.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Sequence

from ..lab.laws import LAW_NAMES
from .blanket import ActionScore, ExperimentAction, LawBelief

_LN2 = math.log(2.0)

# LAW_NAMES order:
#   0: commutative     1: associative    2: has_identity   3: has_inverses
#   4: idempotent      5: f_involution   6: f_homomorphism 7: f_bijective


def _build_factored_profiles():
    op_profiles: list[tuple[bool, bool, bool, bool, bool]] = []
    for comm, assoc, has_id, has_inv, idemp in itertools.product((False, True), repeat=5):
        if has_inv and not has_id:
            continue
        if idemp and has_inv:
            continue
        op_profiles.append((comm, assoc, has_id, has_inv, idemp))

    f_profiles: list[tuple[bool, bool, bool]] = []
    for finv, fhomo, fbij in itertools.product((False, True), repeat=3):
        if finv and not fbij:
            continue
        f_profiles.append((finv, fhomo, fbij))

    valid: list[tuple[bool, ...]] = []
    for op_p in op_profiles:
        for f_p in f_profiles:
            valid.append(op_p + f_p)
    return tuple(op_profiles), tuple(f_profiles), tuple(valid)


_OP_PROFILES, _F_PROFILES, VALID_LAW_PROFILES = _build_factored_profiles()
NUM_OP_PROFILES: int = len(_OP_PROFILES)           # 20
NUM_F_PROFILES: int = len(_F_PROFILES)             # 6
NUM_VALID_PROFILES: int = len(VALID_LAW_PROFILES)  # 120
PROFILE_KEYS: tuple[str, ...] = tuple(
    "".join("1" if b else "0" for b in prof) for prof in VALID_LAW_PROFILES
)
PROFILE_INDEX: dict[str, int] = {k: i for i, k in enumerate(PROFILE_KEYS)}
UNIFORM_JOINT_PRIOR: tuple[float, ...] = tuple(
    1.0 / NUM_VALID_PROFILES for _ in range(NUM_VALID_PROFILES)
)


@dataclass(frozen=True)
class EFEParams:
    """Mutable cognitive hyperparameters (loaded from harness/memory_params/)."""

    beta_law: float = 4.0
    beta_cell: float = 0.25
    lambda_cost: float = 0.08
    action_precision: float = 0.0  # 0.0 => deterministic argmin G(a); >0 => softmax precision
    stop_entropy_bits: float = 0.12
    stop_eig_per_cost: float = 0.008
    min_observations_before_stop: int = 4
    violation_grid: tuple[float, ...] = (0.10, 0.30, 0.60, 1.00)
    noise_floor: float = 1e-4
    forgetting_factor: float = 1.0
    max_assoc_triples: int = 343
    max_homo_pairs: int = 49


def _logsumexp(values: Sequence[float]) -> float:
    m = values[0]
    for v in values[1:]:
        if v > m:
            m = v
    if math.isinf(m):
        return m
    _exp = math.exp
    s = 0.0
    for v in values:
        s += _exp(v - m)
    return m + math.log(s)


def _normalize_probs(probs: Sequence[float]) -> tuple[float, ...]:
    s = sum(probs)
    if s <= 0.0 or not math.isfinite(s):
        k = len(probs)
        return tuple(1.0 / k for _ in range(k))
    inv_s = 1.0 / s
    return tuple(p * inv_s for p in probs)


def _entropy_bits(probs: Sequence[float]) -> float:
    h = 0.0
    _log = math.log
    for p in probs:
        if p > 1e-15:
            h -= p * _log(p)
    return max(0.0, h / _LN2)


def _bf_minus(bf_plus: float, p0: float, inv_one_minus_p0: float) -> float:
    """Exact violation Bayes factor ``(1 - p_match) / (1 - p0)`` given ``bf+ = p_match / p0``."""
    p_match = p0 * bf_plus
    if p_match >= 1.0 - 1e-12:
        return 1e-12 * inv_one_minus_p0
    if p_match <= 1e-12:
        return inv_one_minus_p0
    return (1.0 - p_match) * inv_one_minus_p0


def _acc_init(
    witness_bfs: Sequence[float],
    grid: tuple[float, ...],
    p0: float,
) -> tuple[float, list[float]]:
    """Return ``(sum_h1, sum_h0[len(grid)])`` using exact ``(bf+, bf-)`` pairs."""
    s1 = 0.0
    m = len(grid)
    s0 = [0.0] * m
    inv_1mp0 = 1.0 / max(1.0 - p0, 1e-12)
    _log = math.log
    for bf in witness_bfs:
        if bf == 1.0:
            continue
        bf_pos = bf if bf > 1e-15 else 1e-15
        bf_neg = _bf_minus(bf_pos, p0, inv_1mp0)
        s1 += _log(bf_pos)
        for g in range(m):
            rho = grid[g]
            v = (1.0 - rho) * bf_pos + rho * bf_neg
            s0[g] += _log(v if v > 1e-15 else 1e-15)
    return s1, s0


def _acc_init_pairs(
    witness_pairs: Sequence[tuple[float, float]],
    grid: tuple[float, ...],
) -> tuple[float, list[float]]:
    """Return ``(sum_h1, sum_h0[len(grid)])`` given explicit ``(L1, L0)`` pairs."""
    s1 = 0.0
    m = len(grid)
    s0 = [0.0] * m
    _log = math.log
    for l1, l0 in witness_pairs:
        if l1 == 1.0 and l0 == 1.0:
            continue
        p1 = l1 if l1 > 1e-15 else 1e-15
        p0 = l0 if l0 > 1e-15 else 1e-15
        s1 += _log(p1)
        for g in range(m):
            rho = grid[g]
            v = (1.0 - rho) * p1 + rho * p0
            s0[g] += _log(v if v > 1e-15 else 1e-15)
    return s1, s0


def _acc_to_lbf(
    s1: float,
    s0: Sequence[float],
    grid: tuple[float, ...],
    temper: float,
    k_eff: float,
    log_m: float,
) -> float:
    """Zero-truncated log Bayes factor ``log P(D | H1) - log P(D | H0, |S| >= 1)``."""
    if s1 == 0.0:
        all_zero = True
        for v in s0:
            if v != 0.0:
                all_zero = False
                break
        if all_zero:
            return 0.0
    log_h1 = s1 * temper
    m = len(grid)
    _log = math.log
    _expm1 = math.expm1
    h0_terms = [0.0] * m
    for g in range(m):
        rho = grid[g]
        log_a = s0[g] * temper
        if rho >= 1.0 - 1e-12:
            h0_terms[g] = log_a
        else:
            log_empty = k_eff * _log(1.0 - rho)
            log_b = log_h1 + log_empty
            diff = log_b - log_a
            if diff >= -1e-12:
                diff = -1e-12
            log_num = log_a + _log(-_expm1(diff))
            log_den = _log(-_expm1(log_empty))
            h0_terms[g] = log_num - log_den
    return log_h1 - (_logsumexp(h0_terms) - log_m)


def _acc_delta_lbf(
    base_s1: float,
    base_s0: Sequence[float],
    deltas: Sequence[tuple[float, float]],
    grid: tuple[float, ...],
    p0: float,
    temper: float,
    k_eff: float,
    log_m: float,
) -> float:
    """Compute exact updated zero-truncated log-BF after ``old_bf -> new_bf`` deltas."""
    s1 = base_s1
    m = len(grid)
    s0 = list(base_s0)
    inv_1mp0 = 1.0 / max(1.0 - p0, 1e-12)
    _log = math.log
    for old_bf, new_bf in deltas:
        if old_bf == new_bf:
            continue
        if old_bf != 1.0:
            ob_pos = old_bf if old_bf > 1e-15 else 1e-15
            ob_neg = _bf_minus(ob_pos, p0, inv_1mp0)
            s1 -= _log(ob_pos)
            for g in range(m):
                rho = grid[g]
                v = (1.0 - rho) * ob_pos + rho * ob_neg
                s0[g] -= _log(v if v > 1e-15 else 1e-15)
        if new_bf != 1.0:
            nb_pos = new_bf if new_bf > 1e-15 else 1e-15
            nb_neg = _bf_minus(nb_pos, p0, inv_1mp0)
            s1 += _log(nb_pos)
            for g in range(m):
                rho = grid[g]
                v = (1.0 - rho) * nb_pos + rho * nb_neg
                s0[g] += _log(v if v > 1e-15 else 1e-15)
    return _acc_to_lbf(s1, s0, grid, temper, k_eff, log_m)


def _acc_delta_pairs_lbf(
    base_s1: float,
    base_s0: Sequence[float],
    deltas: Sequence[tuple[tuple[float, float], tuple[float, float]]],
    grid: tuple[float, ...],
    temper: float,
    k_eff: float,
    log_m: float,
) -> float:
    """Exact updated zero-truncated log-BF for explicit ``((old_l1, old_l0), (new_l1, new_l0))``."""
    s1 = base_s1
    m = len(grid)
    s0 = list(base_s0)
    _log = math.log
    for (ol1, ol0), (nl1, nl0) in deltas:
        if ol1 == nl1 and ol0 == nl0:
            continue
        if not (ol1 == 1.0 and ol0 == 1.0):
            op1 = ol1 if ol1 > 1e-15 else 1e-15
            op0 = ol0 if ol0 > 1e-15 else 1e-15
            s1 -= _log(op1)
            for g in range(m):
                rho = grid[g]
                v = (1.0 - rho) * op1 + rho * op0
                s0[g] -= _log(v if v > 1e-15 else 1e-15)
        if not (nl1 == 1.0 and nl0 == 1.0):
            np1 = nl1 if nl1 > 1e-15 else 1e-15
            np0 = nl0 if nl0 > 1e-15 else 1e-15
            s1 += _log(np1)
            for g in range(m):
                rho = grid[g]
                v = (1.0 - rho) * np1 + rho * np0
                s0[g] += _log(v if v > 1e-15 else 1e-15)
    return _acc_to_lbf(s1, s0, grid, temper, k_eff, log_m)


def _mixture_log_bf(
    witness_bfs: Sequence[float],
    violation_grid: tuple[float, ...],
    *,
    p0: float,
    k_total: int,
    temper: float = 1.0,
) -> float:
    if not witness_bfs:
        return 0.0
    s1, s0 = _acc_init(witness_bfs, violation_grid, p0)
    k_eff = max(1.0, k_total * temper)
    return _acc_to_lbf(s1, s0, violation_grid, temper, k_eff, math.log(len(violation_grid)))


class ExactBayesianBackend:
    """Exact pure-Python Bayesian belief and Friston EFE inference engine."""

    def __init__(self, params: EFEParams | None = None) -> None:
        self.params = params or EFEParams()
        self._grid: tuple[float, ...] = self.params.violation_grid
        self._log_m: float = math.log(len(self._grid))
        self._symbols: list[str] = []
        self._index: dict[str, int] = {}
        self._n: int = 0
        self._noise: float = 0.05
        self._p_hit: float = 0.96
        self._p_miss: float = 0.01
        self._delta: float = 0.0
        self._h_cond_bits: float = 0.0
        self._combine_cost: float = 1.0
        self._transform_cost: float = 0.5
        self._prior_log: list[float] = [math.log(p) for p in UNIFORM_JOINT_PRIOR]
        self._op_counts: list[list[list[float]]] = []
        self._op_totals: list[list[float]] = []
        self._f_counts: list[list[float]] = []
        self._f_totals: list[float] = []
        self._op_q: list[list[list[float]]] = []
        self._f_q: list[list[float]] = []
        self._surprises: list[float] = []
        self._obs_count: int = 0

    def reset(
        self,
        symbols: Sequence[str],
        *,
        noise: float,
        combine_cost: float = 1.0,
        transform_cost: float = 0.5,
        prior_joint: tuple[float, ...] | None = None,
    ) -> None:
        self._grid = self.params.violation_grid
        self._log_m = math.log(len(self._grid))
        self._symbols = list(symbols)
        self._index = {s: i for i, s in enumerate(self._symbols)}
        self._n = len(self._symbols)
        if self._n < 2:
            raise ValueError(f"carrier size must be >= 2, got {self._n}")
        self._noise = float(noise)
        eff_noise = min(max(self._noise, self.params.noise_floor), 1.0 - 1e-6)
        self._p_hit = (1.0 - eff_noise) + eff_noise / self._n
        self._p_miss = eff_noise / self._n
        self._delta = math.log(self._p_hit / self._p_miss)
        self._h_cond_bits = _entropy_bits([self._p_hit] + [self._p_miss] * (self._n - 1))
        self._combine_cost = float(combine_cost)
        self._transform_cost = float(transform_cost)

        u = 1.0 / self._n
        self._op_counts = [[[0.0] * self._n for _ in range(self._n)] for _ in range(self._n)]
        self._op_totals = [[0.0] * self._n for _ in range(self._n)]
        self._f_counts = [[0.0] * self._n for _ in range(self._n)]
        self._f_totals = [0.0] * self._n
        self._op_q = [[[u] * self._n for _ in range(self._n)] for _ in range(self._n)]
        self._f_q = [[u] * self._n for _ in range(self._n)]
        self._surprises = []
        self._obs_count = 0
        self.set_prior(prior_joint if prior_joint is not None else UNIFORM_JOINT_PRIOR)

    def set_prior(self, prior_joint: tuple[float, ...]) -> None:
        if len(prior_joint) != NUM_VALID_PROFILES:
            raise ValueError(
                f"prior_joint length must be {NUM_VALID_PROFILES}, got {len(prior_joint)}"
            )
        norm = _normalize_probs([max(float(p), 1e-12) for p in prior_joint])
        self._prior_log = [math.log(p) for p in norm]

    # ----- cell posteriors & observations -----------------------------------

    def _cell_posterior_from_counts(self, counts: Sequence[float], total: float) -> list[float]:
        n = self._n
        if total <= 0.0:
            u = 1.0 / n
            return [u] * n
        delta = self._delta
        max_c = counts[0]
        for c in counts[1:]:
            if c > max_c:
                max_c = c
        _exp = math.exp
        exps = [_exp((c - max_c) * delta) for c in counts]
        inv_s = 1.0 / sum(exps)
        return [e * inv_s for e in exps]

    def _predictive_dist(self, q: Sequence[float]) -> list[float]:
        diff = self._p_hit - self._p_miss
        pm = self._p_miss
        return [pm + diff * qz for qz in q]

    def observe(self, action: ExperimentAction, result: str) -> float:
        y = self._index[result]
        gamma = self.params.forgetting_factor
        if action.kind == "combine":
            i, j = self._index[action.args[0]], self._index[action.args[1]]
            pred = self._p_miss + (self._p_hit - self._p_miss) * self._op_q[i][j][y]
            surprise = -math.log(max(pred, 1e-15))
            counts = self._op_counts[i][j]
            if gamma < 1.0 and self._op_totals[i][j] > 0.0:
                for k in range(self._n):
                    counts[k] *= gamma
                self._op_totals[i][j] *= gamma
            counts[y] += 1.0
            self._op_totals[i][j] += 1.0
            self._op_q[i][j] = self._cell_posterior_from_counts(counts, self._op_totals[i][j])
        else:
            i = self._index[action.args[0]]
            pred = self._p_miss + (self._p_hit - self._p_miss) * self._f_q[i][y]
            surprise = -math.log(max(pred, 1e-15))
            counts = self._f_counts[i]
            if gamma < 1.0 and self._f_totals[i] > 0.0:
                for k in range(self._n):
                    counts[k] *= gamma
                self._f_totals[i] *= gamma
            counts[y] += 1.0
            self._f_totals[i] += 1.0
            self._f_q[i] = self._cell_posterior_from_counts(counts, self._f_totals[i])

        self._surprises.append(surprise)
        self._obs_count += 1
        return surprise

    def surprise_stats(self) -> tuple[float, float]:
        if not self._surprises:
            return (0.0, 0.0)
        return (sum(self._surprises) / len(self._surprises), max(self._surprises))

    # ----- individual witness likelihood helpers ----------------------------

    def _inv_witness_l1_l0(
        self,
        e: int,
        x: int,
        op_q: list[list[list[float]]],
        op_tot: list[list[float]],
    ) -> tuple[float, float]:
        """Exact ``(L1, L0)`` likelihood ratio for 'element x has a two-sided inverse for e'."""
        n = self._n
        row_x_tot = op_tot[x]
        any_obs = False
        for y in range(n):
            if row_x_tot[y] > 0.0 or op_tot[y][x] > 0.0:
                any_obs = True
                break
        if not any_obs:
            return (1.0, 1.0)

        inv_1m_invn = 1.0 / (1.0 - 1.0 / n)
        inv_1m_invn2 = 1.0 / (1.0 - 1.0 / (n * n))
        n2 = float(n * n)
        row_x_q = op_q[x]

        # y == x is a single cell (self-inverse, e.g. Klein-4)
        q_xx_e = row_x_q[x][e]
        l1_sum = n * q_xx_e
        l0_prod = max(1.0 - q_xx_e, 1e-12) * inv_1m_invn

        for y in range(n):
            if y == x:
                continue
            m = row_x_q[y][e] * op_q[y][x][e]
            l1_sum += n2 * m
            l0_prod *= max(1.0 - m, 1e-12) * inv_1m_invn2

        return (l1_sum / n, l0_prod)

    def _assoc_triple_bf(
        self,
        a: int,
        b: int,
        c: int,
        op_q: list[list[list[float]]],
        op_totals: list[list[float]],
        supp_ab: list[tuple[int, float]] | None,
        supp_bc: list[tuple[int, float]] | None,
    ) -> float:
        if supp_ab is None or supp_bc is None:
            return 1.0
        n = self._n
        row_a_tot = op_totals[a]
        l_ok = False
        for u, _ in supp_ab:
            if op_totals[u][c] > 0.0:
                l_ok = True
                break
        if not l_ok:
            return 1.0
        r_ok = False
        for v, _ in supp_bc:
            if row_a_tot[v] > 0.0:
                r_ok = True
                break
        if not r_ok:
            return 1.0
        row_a_q = op_q[a]
        if len(supp_ab) == 1 and len(supp_bc) == 1:
            quc = op_q[supp_ab[0][0]][c]
            qav = row_a_q[supp_bc[0][0]]
            dot = 0.0
            for z in range(n):
                dot += quc[z] * qav[z]
            return n * dot
        p_l = [0.0] * n
        for u, pu in supp_ab:
            quc = op_q[u][c]
            for z in range(n):
                p_l[z] += pu * quc[z]
        p_r = [0.0] * n
        for v, pv in supp_bc:
            qav = row_a_q[v]
            for z in range(n):
                p_r[z] += pv * qav[z]
        dot = 0.0
        for z in range(n):
            dot += p_l[z] * p_r[z]
        return n * dot

    def _homo_pair_bf(
        self,
        a: int,
        b: int,
        op_q: list[list[list[float]]],
        op_totals: list[list[float]],
        f_q: list[list[float]],
        f_totals: list[float],
        sa: list[tuple[int, float]] | None,
        sb: list[tuple[int, float]] | None,
        sab: list[tuple[int, float]] | None,
    ) -> float:
        if sa is None or sb is None or sab is None:
            return 1.0
        l_ok = False
        for u, _ in sab:
            if f_totals[u] > 0.0:
                l_ok = True
                break
        if not l_ok:
            return 1.0
        r_ok = False
        for u, _ in sa:
            row_u = op_totals[u]
            for v, _ in sb:
                if row_u[v] > 0.0:
                    r_ok = True
                    break
            if r_ok:
                break
        if not r_ok:
            return 1.0
        n = self._n
        if len(sab) == 1 and len(sa) == 1 and len(sb) == 1:
            fu = f_q[sab[0][0]]
            q_fafb = op_q[sa[0][0]][sb[0][0]]
            dot = 0.0
            for z in range(n):
                dot += fu[z] * q_fafb[z]
            return n * dot
        p_l = [0.0] * n
        for u, pu in sab:
            fu = f_q[u]
            for z in range(n):
                p_l[z] += pu * fu[z]
        p_r = [0.0] * n
        for u, pu in sa:
            for v, pv in sb:
                w = pu * pv
                quv = op_q[u][v]
                for z in range(n):
                    p_r[z] += w * quv[z]
        dot = 0.0
        for z in range(n):
            dot += p_l[z] * p_r[z]
        return n * dot

    def _build_cache(self) -> dict:
        """Build full per-witness BF accumulators and inverse indices for fast deltas."""
        n = self._n
        p0_inv_n = 1.0 / n
        p0_bij = (n - 1) / n
        grid = self._grid
        log_m = self._log_m
        op_q = self._op_q
        op_tot = self._op_totals
        f_q = self._f_q
        f_tot = self._f_totals

        op_supp: list[list[list[tuple[int, float]] | None]] = [[None] * n for _ in range(n)]
        inv_op: list[list[tuple[int, int]]] = [[] for _ in range(n)]
        for a in range(n):
            for b in range(n):
                if op_tot[a][b] > 0.0:
                    s = [(z, p) for z, p in enumerate(op_q[a][b]) if p > 1e-3]
                    op_supp[a][b] = s
                    for u, _ in s:
                        inv_op[u].append((a, b))

        f_supp: list[list[tuple[int, float]] | None] = [None] * n
        inv_f: list[list[int]] = [[] for _ in range(n)]
        for a in range(n):
            if f_tot[a] > 0.0:
                s = [(z, p) for z, p in enumerate(f_q[a]) if p > 1e-3]
                f_supp[a] = s
                for u, _ in s:
                    inv_f[u].append(a)

        # 0. commutative
        k_pairs = n * (n - 1) // 2
        comm_bf = [[1.0] * n for _ in range(n)]
        comm_list: list[float] = []
        for a in range(n):
            for b in range(a + 1, n):
                if op_tot[a][b] > 0.0 and op_tot[b][a] > 0.0:
                    qa, qb = op_q[a][b], op_q[b][a]
                    dot = 0.0
                    for z in range(n):
                        dot += qa[z] * qb[z]
                    v = n * dot
                    comm_bf[a][b] = v
                    comm_list.append(v)
        comm_acc = _acc_init(comm_list, grid, p0_inv_n)
        lbf_comm = _acc_to_lbf(comm_acc[0], comm_acc[1], grid, 1.0, float(k_pairs), log_m)

        # 4. idempotent
        idemp_bf = [n * op_q[a][a][a] if op_tot[a][a] > 0.0 else 1.0 for a in range(n)]
        idemp_acc = _acc_init(idemp_bf, grid, p0_inv_n)
        lbf_idemp = _acc_to_lbf(idemp_acc[0], idemp_acc[1], grid, 1.0, float(n), log_m)

        # 2. has_identity
        k_id = float(2 * n - 1)
        id_accs: list[tuple[float, list[float]]] = []
        id_cand_lbfs: list[float] = []
        for e in range(n):
            w_bfs: list[float] = []
            if op_tot[e][e] > 0.0:
                w_bfs.append(n * op_q[e][e][e])
            for x in range(n):
                if x == e:
                    continue
                if op_tot[e][x] > 0.0:
                    w_bfs.append(n * op_q[e][x][x])
                if op_tot[x][e] > 0.0:
                    w_bfs.append(n * op_q[x][e][x])
            acc_e = _acc_init(w_bfs, grid, p0_inv_n)
            id_accs.append(acc_e)
            id_cand_lbfs.append(_acc_to_lbf(acc_e[0], acc_e[1], grid, 1.0, k_id, log_m))
        lbf_id = _logsumexp(id_cand_lbfs) - math.log(n)

        # 3. has_inverses (exact L1/L0 pairs per candidate e and element x != e)
        k_inv = float(n - 1)
        inv_x_pairs: list[list[tuple[float, float]]] = [
            [(1.0, 1.0)] * n for _ in range(n)
        ]
        inv_accs: list[tuple[float, list[float]]] = []
        inv_cand_raw: list[float] = []
        for e in range(n):
            p_list: list[tuple[float, float]] = []
            for x in range(n):
                if x == e:
                    continue
                pair = self._inv_witness_l1_l0(e, x, op_q, op_tot)
                inv_x_pairs[e][x] = pair
                if not (pair[0] == 1.0 and pair[1] == 1.0):
                    p_list.append(pair)
            acc_e = _acc_init_pairs(p_list, grid)
            inv_accs.append(acc_e)
            inv_cand_raw.append(_acc_to_lbf(acc_e[0], acc_e[1], grid, 1.0, k_inv, log_m))

        id_norm_lse = _logsumexp(id_cand_lbfs)
        id_weights = [math.exp(v - id_norm_lse) for v in id_cand_lbfs]
        inv_terms = [
            math.log(id_weights[e]) + inv_cand_raw[e]
            for e in range(n)
            if id_weights[e] >= 1e-4
        ]
        lbf_inv = _logsumexp(inv_terms) if inv_terms else 0.0

        # 1. associative
        assoc_bf: dict[tuple[int, int, int], float] = {}
        for a in range(n):
            row_a_supp = op_supp[a]
            for b in range(n):
                sab = row_a_supp[b]
                if sab is None:
                    continue
                row_b_supp = op_supp[b]
                for c in range(n):
                    sbc = row_b_supp[c]
                    if sbc is None:
                        continue
                    v = self._assoc_triple_bf(a, b, c, op_q, op_tot, sab, sbc)
                    if v != 1.0:
                        assoc_bf[(a, b, c)] = v
        assoc_temper = 1.0 / (2.0 * n)
        assoc_k_eff = max(1.0, (n * n * n) * assoc_temper)
        assoc_acc = _acc_init(list(assoc_bf.values()), grid, p0_inv_n)
        lbf_assoc = _acc_to_lbf(
            assoc_acc[0], assoc_acc[1], grid, assoc_temper, assoc_k_eff, log_m
        )

        # 5. f_involution
        finv_bfs: list[float] = []
        for a in range(n):
            if f_tot[a] <= 0.0:
                continue
            ra = f_q[a]
            val = ra[a] * n
            for b in range(n):
                if b != a:
                    val += ra[b] * (n * f_q[b][a])
            finv_bfs.append(val)
        lbf_finv = _mixture_log_bf(finv_bfs, grid, p0=p0_inv_n, k_total=n, temper=0.75)

        # 7. f_bijective
        fbij_bf = [[1.0] * n for _ in range(n)]
        fbij_list: list[float] = []
        scale_bij = n / (n - 1)
        for a in range(n):
            if f_tot[a] <= 0.0:
                continue
            ra = f_q[a]
            for b in range(a + 1, n):
                if f_tot[b] <= 0.0:
                    continue
                rb = f_q[b]
                dot = 0.0
                for z in range(n):
                    dot += ra[z] * rb[z]
                v = scale_bij * max(1.0 - dot, 1e-12)
                fbij_bf[a][b] = v
                fbij_list.append(v)
        fbij_acc = _acc_init(fbij_list, grid, p0_bij)
        lbf_fbij = _acc_to_lbf(fbij_acc[0], fbij_acc[1], grid, 1.0, float(k_pairs), log_m)

        # 6. f_homomorphism
        homo_bf: dict[tuple[int, int], float] = {}
        for a in range(n):
            sa = f_supp[a]
            if sa is None:
                continue
            for b in range(n):
                sb = f_supp[b]
                sab = op_supp[a][b]
                if sb is None or sab is None:
                    continue
                v = self._homo_pair_bf(a, b, op_q, op_tot, f_q, f_tot, sa, sb, sab)
                if v != 1.0:
                    homo_bf[(a, b)] = v
        homo_temper = 1.0 / n
        homo_k_eff = max(1.0, (n * n) * homo_temper)
        homo_acc = _acc_init(list(homo_bf.values()), grid, p0_inv_n)
        lbf_fhomo = _acc_to_lbf(
            homo_acc[0], homo_acc[1], grid, homo_temper, homo_k_eff, log_m
        )

        lbfs = [lbf_comm, lbf_assoc, lbf_id, lbf_inv, lbf_idemp, lbf_finv, lbf_fhomo, lbf_fbij]
        return {
            "lbfs": lbfs,
            "p0_inv_n": p0_inv_n,
            "p0_bij": p0_bij,
            "k_pairs": float(k_pairs),
            "k_id": k_id,
            "k_inv": k_inv,
            "id_cand_lbfs": id_cand_lbfs,
            "op_supp": op_supp,
            "inv_op": inv_op,
            "f_supp": f_supp,
            "inv_f": inv_f,
            "comm_bf": comm_bf,
            "comm_acc": comm_acc,
            "idemp_bf": idemp_bf,
            "idemp_acc": idemp_acc,
            "id_accs": id_accs,
            "inv_x_pairs": inv_x_pairs,
            "inv_accs": inv_accs,
            "inv_cand_raw": inv_cand_raw,
            "assoc_bf": assoc_bf,
            "assoc_acc": assoc_acc,
            "assoc_temper": assoc_temper,
            "assoc_k_eff": assoc_k_eff,
            "fbij_bf": fbij_bf,
            "fbij_acc": fbij_acc,
            "homo_bf": homo_bf,
            "homo_acc": homo_acc,
            "homo_temper": homo_temper,
            "homo_k_eff": homo_k_eff,
        }

    # ----- exact factored 20 x 6 joint belief & collapsed entropy -----------

    def _op_profile_lbfs(self, lbfs: Sequence[float]) -> list[float]:
        c, a, d, v, m = lbfs[0], lbfs[1], lbfs[2], lbfs[3], lbfs[4]
        out = [0.0] * NUM_OP_PROFILES
        for idx, (pc, pa, pd, pv, pm) in enumerate(_OP_PROFILES):
            s = 0.0
            if pc:
                s += c
            if pa:
                s += a
            if pd:
                s += d
            if pv:
                s += v
            if pm:
                s += m
            out[idx] = s
        return out

    def _f_profile_lbfs(self, lbfs: Sequence[float]) -> list[float]:
        fi, fh, fb = lbfs[5], lbfs[6], lbfs[7]
        out = [0.0] * NUM_F_PROFILES
        for idx, (pfi, pfh, pfb) in enumerate(_F_PROFILES):
            s = 0.0
            if pfi:
                s += fi
            if pfh:
                s += fh
            if pfb:
                s += fb
            out[idx] = s
        return out

    def _joint_from_factored(
        self, op_vals: Sequence[float], f_vals: Sequence[float]
    ) -> tuple[float, ...]:
        prior_log = self._prior_log
        log_post = [0.0] * NUM_VALID_PROFILES
        idx = 0
        max_lp = -math.inf
        for ov in op_vals:
            for fv in f_vals:
                lp = prior_log[idx] + ov + fv
                log_post[idx] = lp
                if lp > max_lp:
                    max_lp = lp
                idx += 1
        _exp = math.exp
        exps = [_exp(lp - max_lp) for lp in log_post]
        inv_s = 1.0 / sum(exps)
        return tuple(e * inv_s for e in exps)

    def _collapse_for_op(self, f_vals: Sequence[float]) -> tuple[list[float], list[float], list[float]]:
        """Pre-collapse the 6 f-columns so exact 120-state entropy runs in 20 steps!

        For each ``oi in 0..19``:
          ``m_oi = max_fi (prior_log[oi, fi] + f_vals[fi])``
          ``u_oi = sum_fi exp(prior_log[oi, fi] + f_vals[fi] - m_oi)``
          ``e_oi = sum_fi exp(...) * (prior_log[oi, fi] + f_vals[fi] - m_oi)``
        """
        prior_log = self._prior_log
        _exp = math.exp
        m_arr = [0.0] * NUM_OP_PROFILES
        u_arr = [0.0] * NUM_OP_PROFILES
        e_arr = [0.0] * NUM_OP_PROFILES
        base = 0
        for oi in range(NUM_OP_PROFILES):
            m = prior_log[base] + f_vals[0]
            for fi in range(1, NUM_F_PROFILES):
                v = prior_log[base + fi] + f_vals[fi]
                if v > m:
                    m = v
            u = 0.0
            e = 0.0
            for fi in range(NUM_F_PROFILES):
                d = (prior_log[base + fi] + f_vals[fi]) - m
                w = _exp(d)
                u += w
                e += w * d
            m_arr[oi] = m
            u_arr[oi] = u
            e_arr[oi] = e
            base += NUM_F_PROFILES
        return m_arr, u_arr, e_arr

    def _collapse_for_f(self, op_vals: Sequence[float]) -> tuple[list[float], list[float], list[float]]:
        """Pre-collapse the 20 op-rows so exact 120-state entropy runs in 6 steps!"""
        prior_log = self._prior_log
        _exp = math.exp
        m_arr = [0.0] * NUM_F_PROFILES
        u_arr = [0.0] * NUM_F_PROFILES
        e_arr = [0.0] * NUM_F_PROFILES
        for fi in range(NUM_F_PROFILES):
            m = prior_log[fi] + op_vals[0]
            for oi in range(1, NUM_OP_PROFILES):
                v = prior_log[oi * NUM_F_PROFILES + fi] + op_vals[oi]
                if v > m:
                    m = v
            u = 0.0
            e = 0.0
            for oi in range(NUM_OP_PROFILES):
                d = (prior_log[oi * NUM_F_PROFILES + fi] + op_vals[oi]) - m
                w = _exp(d)
                u += w
                e += w * d
            m_arr[fi] = m
            u_arr[fi] = u
            e_arr[fi] = e
        return m_arr, u_arr, e_arr

    def _entropy_collapsed(
        self,
        vals: Sequence[float],
        m_arr: Sequence[float],
        u_arr: Sequence[float],
        e_arr: Sequence[float],
    ) -> float:
        """Exact 120-state Shannon entropy in bits using pre-collapsed arrays."""
        k = len(vals)
        max_shift = vals[0] + m_arr[0]
        for i in range(1, k):
            sh = vals[i] + m_arr[i]
            if sh > max_shift:
                max_shift = sh
        _exp = math.exp
        z = 0.0
        weighted = 0.0
        for i in range(k):
            sh = (vals[i] + m_arr[i]) - max_shift
            scale = _exp(sh)
            z += scale * u_arr[i]
            weighted += scale * (e_arr[i] + sh * u_arr[i])
        return max(0.0, (math.log(z) - weighted / z) / _LN2)

    def _entropy_from_factored(
        self, op_vals: Sequence[float], f_vals: Sequence[float]
    ) -> float:
        prior_log = self._prior_log
        log_post = [0.0] * NUM_VALID_PROFILES
        idx = 0
        max_lp = -math.inf
        for ov in op_vals:
            for fv in f_vals:
                lp = prior_log[idx] + ov + fv
                log_post[idx] = lp
                if lp > max_lp:
                    max_lp = lp
                idx += 1
        _exp = math.exp
        s = 0.0
        weighted = 0.0
        for lp in log_post:
            dl = lp - max_lp
            w = _exp(dl)
            s += w
            weighted += w * dl
        return max(0.0, (math.log(s) - weighted / s) / _LN2)

    def belief(self) -> LawBelief:
        cache = self._build_cache()
        lbfs = cache["lbfs"]
        id_cand_lbfs = cache["id_cand_lbfs"]
        joint = self._joint_from_factored(
            self._op_profile_lbfs(lbfs), self._f_profile_lbfs(lbfs)
        )
        marginals = [0.0] * len(LAW_NAMES)
        for idx, p in enumerate(joint):
            prof = VALID_LAW_PROFILES[idx]
            for k in range(8):
                if prof[k]:
                    marginals[k] += p
        law_probs = {name: min(max(marginals[k], 0.0), 1.0) for k, name in enumerate(LAW_NAMES)}
        map_claims = {name: law_probs[name] > 0.5 for name in LAW_NAMES}
        id_lse = _logsumexp(id_cand_lbfs)
        p_has_id = law_probs["has_identity"]
        id_probs = {
            self._symbols[e]: p_has_id * math.exp(id_cand_lbfs[e] - id_lse)
            for e in range(self._n)
        }
        return LawBelief(
            law_probs=law_probs,
            joint_probs=joint,
            entropy_bits=_entropy_bits(joint),
            map_claims=map_claims,
            identity_probs=id_probs,
        )

    # ----- fast incremental counterfactual log-BFs --------------------------

    def _delta_lbfs_combine(
        self,
        cache: dict,
        i: int,
        j: int,
        affected_triples: list[tuple[int, int, int]],
        affected_homo: list[tuple[int, int]],
        old_q: list[float],
        old_tot: float,
        new_q: list[float],
        new_tot: float,
        *,
        skip_comm: bool = False,
        skip_idemp: bool = False,
        skip_id: bool = False,
        skip_assoc: bool = False,
        skip_homo: bool = False,
        skip_inv: bool = False,
        assoc_scale: float = 1.0,
    ) -> list[float]:
        n = self._n
        p0_inv_n = cache["p0_inv_n"]
        grid = self._grid
        log_m = self._log_m
        op_q = self._op_q
        op_tot = self._op_totals
        f_q = self._f_q
        f_tot = self._f_totals
        op_supp = cache["op_supp"]
        f_supp = cache["f_supp"]

        base_lbfs = cache["lbfs"]
        lbf_comm = base_lbfs[0]
        lbf_idemp = base_lbfs[4]

        # 0. commutative
        if not skip_comm and i != j and op_tot[j][i] > 0.0:
            a, b = (i, j) if i < j else (j, i)
            old_bf = cache["comm_bf"][a][b]
            q_other = op_q[j][i]
            dot = 0.0
            for z in range(n):
                dot += new_q[z] * q_other[z]
            lbf_comm = _acc_delta_lbf(
                cache["comm_acc"][0],
                cache["comm_acc"][1],
                [(old_bf, n * dot)],
                grid,
                p0_inv_n,
                1.0,
                cache["k_pairs"],
                log_m,
            )

        # 4. idempotent
        if not skip_idemp and i == j:
            old_bf = cache["idemp_bf"][i]
            lbf_idemp = _acc_delta_lbf(
                cache["idemp_acc"][0],
                cache["idemp_acc"][1],
                [(old_bf, n * new_q[i])],
                grid,
                p0_inv_n,
                1.0,
                float(n),
                log_m,
            )

        # 2. has_identity
        if skip_id and skip_inv:
            id_cand_lbfs = cache["id_cand_lbfs"]
            lbf_id = base_lbfs[2]
        else:
            id_cand_lbfs = list(cache["id_cand_lbfs"])
            k_id = cache["k_id"]
            if i == j:
                old_w = n * old_q[i] if old_tot > 0.0 else 1.0
                acc_i = cache["id_accs"][i]
                id_cand_lbfs[i] = _acc_delta_lbf(
                    acc_i[0], acc_i[1], [(old_w, n * new_q[i])], grid, p0_inv_n, 1.0, k_id, log_m
                )
            else:
                old_wi = n * old_q[j] if old_tot > 0.0 else 1.0
                acc_i = cache["id_accs"][i]
                id_cand_lbfs[i] = _acc_delta_lbf(
                    acc_i[0], acc_i[1], [(old_wi, n * new_q[j])], grid, p0_inv_n, 1.0, k_id, log_m
                )
                old_wj = n * old_q[i] if old_tot > 0.0 else 1.0
                acc_j = cache["id_accs"][j]
                id_cand_lbfs[j] = _acc_delta_lbf(
                    acc_j[0], acc_j[1], [(old_wj, n * new_q[i])], grid, p0_inv_n, 1.0, k_id, log_m
                )
            lbf_id = _logsumexp(id_cand_lbfs) - math.log(n)

        need_mutate = (not skip_inv) or (not skip_assoc and bool(affected_triples)) or (
            not skip_homo and bool(affected_homo)
        )
        if need_mutate:
            old_supp_ij = op_supp[i][j]
            op_q[i][j] = new_q
            op_tot[i][j] = new_tot
            op_supp[i][j] = [(z, p) for z, p in enumerate(new_q) if p > 1e-3]

        # 3. has_inverses
        if skip_inv:
            lbf_inv = base_lbfs[3]
        else:
            id_norm_lse = _logsumexp(id_cand_lbfs)
            inv_x_pairs = cache["inv_x_pairs"]
            inv_accs = cache["inv_accs"]
            k_inv = cache["k_inv"]
            inv_terms: list[float] = []
            touched_xs = (i,) if i == j else (i, j)
            for e in range(n):
                we = math.exp(id_cand_lbfs[e] - id_norm_lse)
                if we < 1e-4:
                    continue
                deltas_e: list[tuple[tuple[float, float], tuple[float, float]]] = []
                for x in touched_xs:
                    if x == e:
                        continue
                    new_pair = self._inv_witness_l1_l0(e, x, op_q, op_tot)
                    deltas_e.append((inv_x_pairs[e][x], new_pair))
                if deltas_e:
                    acc_e = inv_accs[e]
                    cand_lbf = _acc_delta_pairs_lbf(
                        acc_e[0], acc_e[1], deltas_e, grid, 1.0, k_inv, log_m
                    )
                else:
                    cand_lbf = cache["inv_cand_raw"][e]
                inv_terms.append(math.log(we) + cand_lbf)
            lbf_inv = _logsumexp(inv_terms) if inv_terms else 0.0

        # 1. associative
        if skip_assoc or not affected_triples:
            lbf_assoc = base_lbfs[1]
        else:
            assoc_bf = cache["assoc_bf"]
            assoc_deltas: list[tuple[float, float]] = []
            for a, b, c in affected_triples:
                old_v = assoc_bf.get((a, b, c), 1.0)
                new_v = self._assoc_triple_bf(a, b, c, op_q, op_tot, op_supp[a][b], op_supp[b][c])
                if old_v != new_v:
                    assoc_deltas.append((old_v, new_v))
            if assoc_deltas:
                if assoc_scale == 1.0:
                    lbf_assoc = _acc_delta_lbf(
                        cache["assoc_acc"][0],
                        cache["assoc_acc"][1],
                        assoc_deltas,
                        grid,
                        p0_inv_n,
                        cache["assoc_temper"],
                        cache["assoc_k_eff"],
                        log_m,
                    )
                else:
                    # Unbiased stride-scaled delta for lookahead when >16 triples touch (i, j)
                    s1_d, s0_d = _acc_init([nv for _, nv in assoc_deltas], grid, p0_inv_n)
                    s1_o, s0_o = _acc_init([ov for ov, _ in assoc_deltas], grid, p0_inv_n)
                    base_s1, base_s0 = cache["assoc_acc"]
                    new_s1 = base_s1 + assoc_scale * (s1_d - s1_o)
                    new_s0 = [
                        base_s0[g] + assoc_scale * (s0_d[g] - s0_o[g])
                        for g in range(len(grid))
                    ]
                    lbf_assoc = _acc_to_lbf(
                        new_s1, new_s0, grid, cache["assoc_temper"], cache["assoc_k_eff"], log_m
                    )
            else:
                lbf_assoc = base_lbfs[1]

        # 6. f_homomorphism
        if skip_homo or not affected_homo:
            lbf_fhomo = base_lbfs[6]
        else:
            homo_bf = cache["homo_bf"]
            h_deltas: list[tuple[float, float]] = []
            for a, b in affected_homo:
                old_v = homo_bf.get((a, b), 1.0)
                new_v = self._homo_pair_bf(
                    a, b, op_q, op_tot, f_q, f_tot, f_supp[a], f_supp[b], op_supp[a][b]
                )
                if old_v != new_v:
                    h_deltas.append((old_v, new_v))
            if h_deltas:
                lbf_fhomo = _acc_delta_lbf(
                    cache["homo_acc"][0],
                    cache["homo_acc"][1],
                    h_deltas,
                    grid,
                    p0_inv_n,
                    cache["homo_temper"],
                    cache["homo_k_eff"],
                    log_m,
                )
            else:
                lbf_fhomo = base_lbfs[6]

        if need_mutate:
            op_q[i][j] = old_q
            op_tot[i][j] = old_tot
            op_supp[i][j] = old_supp_ij

        return [
            lbf_comm,
            lbf_assoc,
            lbf_id,
            lbf_inv,
            lbf_idemp,
            base_lbfs[5],
            lbf_fhomo,
            base_lbfs[7],
        ]

    def _delta_lbfs_transform(
        self,
        cache: dict,
        i: int,
        affected_homo: list[tuple[int, int]],
        old_q: list[float],
        old_tot: float,
        new_q: list[float],
        new_tot: float,
        *,
        skip_homo: bool = False,
    ) -> list[float]:
        n = self._n
        p0_inv_n = cache["p0_inv_n"]
        p0_bij = cache["p0_bij"]
        grid = self._grid
        log_m = self._log_m
        op_q = self._op_q
        op_tot = self._op_totals
        f_q = self._f_q
        f_tot = self._f_totals
        op_supp = cache["op_supp"]
        f_supp = cache["f_supp"]
        base_lbfs = cache["lbfs"]

        old_supp_i = f_supp[i]
        f_q[i] = new_q
        f_tot[i] = new_tot
        f_supp[i] = [(z, p) for z, p in enumerate(new_q) if p > 1e-3]

        # 5. f_involution
        finv_bfs: list[float] = []
        for a in range(n):
            if f_tot[a] <= 0.0:
                continue
            ra = f_q[a]
            val = ra[a] * n
            for b in range(n):
                if b != a:
                    val += ra[b] * (n * f_q[b][a])
            finv_bfs.append(val)
        lbf_finv = _mixture_log_bf(finv_bfs, grid, p0=p0_inv_n, k_total=n, temper=0.75)

        # 7. f_bijective
        scale_bij = n / (n - 1)
        fbij_bf = cache["fbij_bf"]
        bij_deltas: list[tuple[float, float]] = []
        for b in range(n):
            if b == i or f_tot[b] <= 0.0:
                continue
            a, c = (i, b) if i < b else (b, i)
            old_v = fbij_bf[a][c]
            rb = f_q[b]
            dot = 0.0
            for z in range(n):
                dot += new_q[z] * rb[z]
            new_v = scale_bij * max(1.0 - dot, 1e-12)
            bij_deltas.append((old_v, new_v))
        lbf_fbij = _acc_delta_lbf(
            cache["fbij_acc"][0],
            cache["fbij_acc"][1],
            bij_deltas,
            grid,
            p0_bij,
            1.0,
            cache["k_pairs"],
            log_m,
        )

        # 6. f_homomorphism
        if skip_homo or not affected_homo:
            lbf_fhomo = base_lbfs[6]
        else:
            homo_bf = cache["homo_bf"]
            h_deltas: list[tuple[float, float]] = []
            for a, b in affected_homo:
                old_v = homo_bf.get((a, b), 1.0)
                new_v = self._homo_pair_bf(
                    a, b, op_q, op_tot, f_q, f_tot, f_supp[a], f_supp[b], op_supp[a][b]
                )
                if old_v != new_v:
                    h_deltas.append((old_v, new_v))
            if h_deltas:
                lbf_fhomo = _acc_delta_lbf(
                    cache["homo_acc"][0],
                    cache["homo_acc"][1],
                    h_deltas,
                    grid,
                    p0_inv_n,
                    cache["homo_temper"],
                    cache["homo_k_eff"],
                    log_m,
                )
            else:
                lbf_fhomo = base_lbfs[6]

        f_q[i] = old_q
        f_tot[i] = old_tot
        f_supp[i] = old_supp_i

        return [
            base_lbfs[0],
            base_lbfs[1],
            base_lbfs[2],
            base_lbfs[3],
            base_lbfs[4],
            lbf_finv,
            lbf_fhomo,
            lbf_fbij,
        ]

    # ----- Friston Expected Free Energy (EFE) -------------------------------

    def _cell_eig_bits(self, pred: Sequence[float]) -> float:
        return max(0.0, _entropy_bits(pred) - self._h_cond_bits)

    def _candidate_actions(self, remaining_budget: float) -> list[ExperimentAction]:
        actions: list[ExperimentAction] = []
        if remaining_budget + 1e-9 >= self._transform_cost:
            for a in self._symbols:
                actions.append(ExperimentAction("transform", (a,), self._transform_cost))
        if remaining_budget + 1e-9 >= self._combine_cost:
            for a in self._symbols:
                for b in self._symbols:
                    actions.append(ExperimentAction("combine", (a, b), self._combine_cost))
        return actions

    def score_actions(self, remaining_budget: float) -> list[ActionScore]:
        actions = self._candidate_actions(remaining_budget)
        if not actions:
            return []

        cache = self._build_cache()
        base_lbfs = cache["lbfs"]
        base_op_vals = self._op_profile_lbfs(base_lbfs)
        base_f_vals = self._f_profile_lbfs(base_lbfs)
        # Pre-collapse 20x6 matrix into O(20) and O(6) exact entropy evaluators
        op_m, op_u, op_e = self._collapse_for_op(base_f_vals)
        f_m, f_u, f_e = self._collapse_for_f(base_op_vals)
        h_cur = self._entropy_collapsed(base_op_vals, op_m, op_u, op_e)

        # Lookahead skip flags for already decisively settled laws (>1,000:1 odds)
        skip_comm = abs(base_lbfs[0]) > 7.5
        skip_assoc = abs(base_lbfs[1]) > 7.5
        skip_id = abs(base_lbfs[2]) > 7.5
        skip_inv = abs(base_lbfs[3]) > 7.5 or base_lbfs[2] < -7.5
        skip_idemp = abs(base_lbfs[4]) > 7.5
        skip_homo = abs(base_lbfs[6]) > 7.5

        beta_law = self.params.beta_law
        beta_cell = self.params.beta_cell
        lam_cost = self.params.lambda_cost
        gamma = self.params.forgetting_factor
        n = self._n
        op_tot = self._op_totals
        f_tot = self._f_totals
        inv_op = cache["inv_op"]
        inv_f = cache["inv_f"]
        id_cand_lbfs = cache["id_cand_lbfs"]

        # Check whether any unconfirmed cells remain (lets unanimously-confirmed cells fast-path)
        has_unconfirmed = False
        for a in range(n):
            if f_tot[a] < 2.0 or max(self._f_q[a]) < 0.995:
                has_unconfirmed = True
                break
            for b in range(n):
                if op_tot[a][b] < 2.0 or max(self._op_q[a][b]) < 0.995:
                    has_unconfirmed = True
                    break
            if has_unconfirmed:
                break

        # Exact symmetry deduplication for isolated unobserved actions
        sym_cache_combine: dict[tuple, float] = {}
        sym_cache_transform: dict[tuple, float] = {}

        scored: list[ActionScore] = []
        for action in actions:
            if action.kind == "combine":
                i, j = self._index[action.args[0]], self._index[action.args[1]]
                q_old = self._op_q[i][j]
                pred = self._predictive_dist(q_old)
                cell_eig = self._cell_eig_bits(pred)
                old_counts = self._op_counts[i][j]
                old_tot = op_tot[i][j]

                # Already unanimously confirmed (>=2 obs, q_max > 0.995): negligible law EIG
                if has_unconfirmed and old_tot >= 2.0 and max(q_old) > 0.995:
                    law_eig = 0.0
                    efe = -(beta_cell * cell_eig) + (lam_cost * action.cost)
                    scored.append(ActionScore(action, efe, law_eig, cell_eig, action.cost))
                    continue

                if skip_assoc:
                    affected_triples: list[tuple[int, int, int]] = []
                    assoc_scale = 1.0
                else:
                    t_set: set[tuple[int, int, int]] = set()
                    for c in range(n):
                        if op_tot[j][c] > 0.0 or (j == i and c == j):
                            t_set.add((i, j, c))
                    for a in range(n):
                        if op_tot[a][i] > 0.0 or (a == i and i == j):
                            t_set.add((a, i, j))
                    for a, b in inv_op[i]:
                        if op_tot[b][j] > 0.0 or (b == i and j == j):
                            t_set.add((a, b, j))
                    for b, c in inv_op[j]:
                        if op_tot[i][b] > 0.0 or (i == i and b == j):
                            t_set.add((i, b, c))
                    affected_triples = list(t_set)
                    n_aff = len(affected_triples)
                    if n_aff > 16:
                        step_sz = n_aff / 16.0
                        affected_triples = [affected_triples[int(k * step_sz)] for k in range(16)]
                        assoc_scale = n_aff / 16.0
                    else:
                        assoc_scale = 1.0

                if skip_homo:
                    affected_homo: list[tuple[int, int]] = []
                else:
                    h_set: set[tuple[int, int]] = set()
                    if f_tot[i] > 0.0 and f_tot[j] > 0.0:
                        h_set.add((i, j))
                    for a in inv_f[i]:
                        for b in inv_f[j]:
                            if op_tot[a][b] > 0.0 or (a == i and b == j):
                                h_set.add((a, b))
                    affected_homo = list(h_set)

                # Deduplicate structurally identical isolated unobserved cells
                sym_key: tuple | None = None
                if (
                    old_tot == 0.0
                    and op_tot[j][i] == 0.0
                    and not affected_triples
                    and not affected_homo
                    and (
                        skip_inv
                        or all(
                            op_tot[i][y] == 0.0 and op_tot[y][i] == 0.0
                            and op_tot[j][y] == 0.0 and op_tot[y][j] == 0.0
                            for y in range(n)
                        )
                    )
                ):
                    sym_key = (
                        i == j,
                        round(id_cand_lbfs[i], 4),
                        round(id_cand_lbfs[j], 4),
                    )
                    if sym_key in sym_cache_combine:
                        law_eig = sym_cache_combine[sym_key]
                        efe = -(beta_law * law_eig) - (beta_cell * cell_eig) + (lam_cost * action.cost)
                        scored.append(ActionScore(action, efe, law_eig, cell_eig, action.cost))
                        continue

                # For sharp observed cells (q_max >= 0.90), fold non-MAP noise outcomes into
                # the most informative counter-outcome (symmetric partner / diag / fallback)
                outcomes_w: list[tuple[int, float]]
                q_max = max(q_old)
                if old_tot >= 1.0 and q_max >= 0.90:
                    y_map = q_old.index(q_max)
                    y_alt = -1
                    if i == j and y_map != i:
                        y_alt = i
                    elif i != j and op_tot[j][i] > 0.0:
                        q_sym = self._op_q[j][i]
                        z_sym = q_sym.index(max(q_sym))
                        if z_sym != y_map:
                            y_alt = z_sym
                    if y_alt == -1:
                        y_alt = (y_map + 1) % n
                    p_map = pred[y_map]
                    outcomes_w = [(y_map, p_map), (y_alt, max(0.0, 1.0 - p_map))]
                else:
                    outcomes_w = [(y, pred[y]) for y in range(n) if pred[y] >= 1e-6]

                exp_h = 0.0
                for y, py in outcomes_w:
                    if py < 1e-6:
                        continue
                    if gamma < 1.0 and old_tot > 0.0:
                        new_counts = [c * gamma for c in old_counts]
                        new_tot = old_tot * gamma + 1.0
                    else:
                        new_counts = list(old_counts)
                        new_tot = old_tot + 1.0
                    new_counts[y] += 1.0
                    new_q = self._cell_posterior_from_counts(new_counts, new_tot)
                    lbfs_y = self._delta_lbfs_combine(
                        cache,
                        i,
                        j,
                        affected_triples,
                        affected_homo,
                        q_old,
                        old_tot,
                        new_q,
                        new_tot,
                        skip_comm=skip_comm,
                        skip_idemp=skip_idemp,
                        skip_id=skip_id,
                        skip_assoc=skip_assoc,
                        skip_homo=skip_homo,
                        skip_inv=skip_inv,
                        assoc_scale=assoc_scale,
                    )
                    op_vals_y = self._op_profile_lbfs(lbfs_y)
                    if lbfs_y[6] == base_lbfs[6]:
                        exp_h += py * self._entropy_collapsed(op_vals_y, op_m, op_u, op_e)
                    else:
                        f_vals_y = self._f_profile_lbfs(lbfs_y)
                        exp_h += py * self._entropy_from_factored(op_vals_y, f_vals_y)

                law_eig = max(0.0, h_cur - exp_h)
                if sym_key is not None:
                    sym_cache_combine[sym_key] = law_eig
            else:
                i = self._index[action.args[0]]
                q_old = self._f_q[i]
                pred = self._predictive_dist(q_old)
                cell_eig = self._cell_eig_bits(pred)
                old_counts = self._f_counts[i]
                old_tot = f_tot[i]

                if has_unconfirmed and old_tot >= 2.0 and max(q_old) > 0.995:
                    law_eig = 0.0
                    efe = -(beta_cell * cell_eig) + (lam_cost * action.cost)
                    scored.append(ActionScore(action, efe, law_eig, cell_eig, action.cost))
                    continue

                if skip_homo:
                    affected_homo = []
                else:
                    h_set = set()
                    for b in range(n):
                        if op_tot[i][b] > 0.0 and (f_tot[b] > 0.0 or b == i):
                            h_set.add((i, b))
                    for a in range(n):
                        if op_tot[a][i] > 0.0 and (f_tot[a] > 0.0 or a == i):
                            h_set.add((a, i))
                    for a, b in inv_op[i]:
                        if (f_tot[a] > 0.0 or a == i) and (f_tot[b] > 0.0 or b == i):
                            h_set.add((a, b))
                    affected_homo = list(h_set)

                t_sym_key: tuple | None = None
                if old_tot == 0.0 and not affected_homo and not inv_f[i]:
                    t_sym_key = (len(inv_op[i]) == 0,)
                    if t_sym_key in sym_cache_transform:
                        law_eig = sym_cache_transform[t_sym_key]
                        efe = -(beta_law * law_eig) - (beta_cell * cell_eig) + (lam_cost * action.cost)
                        scored.append(ActionScore(action, efe, law_eig, cell_eig, action.cost))
                        continue

                q_max = max(q_old)
                if old_tot >= 1.0 and q_max >= 0.90:
                    y_map = q_old.index(q_max)
                    y_alt = i if y_map != i else (y_map + 1) % n
                    p_map = pred[y_map]
                    outcomes_w = [(y_map, p_map), (y_alt, max(0.0, 1.0 - p_map))]
                else:
                    outcomes_w = [(y, pred[y]) for y in range(n) if pred[y] >= 1e-6]

                exp_h = 0.0
                for y, py in outcomes_w:
                    if py < 1e-6:
                        continue
                    if gamma < 1.0 and old_tot > 0.0:
                        new_counts = [c * gamma for c in old_counts]
                        new_tot = old_tot * gamma + 1.0
                    else:
                        new_counts = list(old_counts)
                        new_tot = old_tot + 1.0
                    new_counts[y] += 1.0
                    new_q = self._cell_posterior_from_counts(new_counts, new_tot)
                    lbfs_y = self._delta_lbfs_transform(
                        cache,
                        i,
                        affected_homo,
                        q_old,
                        old_tot,
                        new_q,
                        new_tot,
                        skip_homo=skip_homo,
                    )
                    f_vals_y = self._f_profile_lbfs(lbfs_y)
                    exp_h += py * self._entropy_collapsed(f_vals_y, f_m, f_u, f_e)

                law_eig = max(0.0, h_cur - exp_h)
                if t_sym_key is not None:
                    sym_cache_transform[t_sym_key] = law_eig

            efe = -(beta_law * law_eig) - (beta_cell * cell_eig) + (lam_cost * action.cost)
            scored.append(
                ActionScore(
                    action=action,
                    efe=efe,
                    law_eig_bits=law_eig,
                    cell_eig_bits=cell_eig,
                    expected_cost=action.cost,
                )
            )

        scored.sort(key=lambda s: (s.efe, s.action.cost, s.action.kind, s.action.args))
        return scored

    def select_action(
        self, scores: Sequence[ActionScore], rng: random.Random | None = None
    ) -> ExperimentAction:
        if not scores:
            raise ValueError("no affordable actions to select from")
        prec = self.params.action_precision
        if prec <= 0.0 or rng is None:
            return scores[0].action
        min_g = min(s.efe for s in scores)
        weights = [math.exp(-prec * (s.efe - min_g)) for s in scores]
        total = sum(weights)
        r = rng.random() * total
        acc = 0.0
        for s, w in zip(scores, weights):
            acc += w
            if r <= acc:
                return s.action
        return scores[-1].action

    def should_stop(self, scores: Sequence[ActionScore]) -> tuple[bool, str]:
        if not scores:
            return (True, "budget_exhausted")
        if self._obs_count < self.params.min_observations_before_stop:
            return (False, "")
        b = self.belief()
        if b.entropy_bits <= self.params.stop_entropy_bits:
            return (True, "entropy_converged")
        best_eig_per_cost = max(
            (s.law_eig_bits + 0.1 * s.cell_eig_bits) / max(s.expected_cost, 1e-6)
            for s in scores
        )
        if best_eig_per_cost < self.params.stop_eig_per_cost:
            return (True, "low_epistemic_gain")
        return (False, "")

    def feature_summary(self) -> dict[str, float]:
        """Symbol-agnostic structural invariants inferred solely from agent beliefs."""
        n = self._n
        diag_obs = [self._op_q[i][i][i] for i in range(n) if self._op_totals[i][i] > 0.0]
        diag_fix = sum(diag_obs) / len(diag_obs) if diag_obs else (1.0 / n)

        f_obs = [self._f_q[i][i] for i in range(n) if self._f_totals[i] > 0.0]
        f_fix = sum(f_obs) / len(f_obs) if f_obs else (1.0 / n)

        row_uniq: list[float] = []
        for i in range(n):
            obs_cols = [j for j in range(n) if self._op_totals[i][j] > 0.0]
            if len(obs_cols) >= 2:
                vals = {max(range(n), key=lambda z: self._op_q[i][j][z]) for j in obs_cols}
                row_uniq.append(len(vals) / len(obs_cols))
        latin_score = sum(row_uniq) / len(row_uniq) if row_uniq else 0.5

        b = self.belief()
        op_cov = sum(1 for i in range(n) for j in range(n) if self._op_totals[i][j] > 0.0) / (n * n)
        f_cov = sum(1 for i in range(n) if self._f_totals[i] > 0.0) / n
        return {
            "diag_fixpoint_rate": round(diag_fix, 4),
            "f_fixpoint_rate": round(f_fix, 4),
            "row_uniqueness": round(latin_score, 4),
            "identity_prob": round(b.law_probs["has_identity"], 4),
            "op_coverage": round(op_cov, 4),
            "f_coverage": round(f_cov, 4),
        }
