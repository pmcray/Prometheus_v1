"""Tests for Phase 1 Friston active inference and exact Bayesian EFE backend."""

from __future__ import annotations

import math
import random

import pytest

from prometheus.core.blanket import ExperimentAction
from prometheus.core.inference import (
    NUM_VALID_PROFILES,
    UNIFORM_JOINT_PRIOR,
    VALID_LAW_PROFILES,
    EFEParams,
    ExactBayesianBackend,
)
from prometheus.lab.laws import LAW_NAMES
from prometheus.lab.world import HiddenLawWorld


def test_valid_law_profiles_logically_consistent():
    assert NUM_VALID_PROFILES == 120
    assert len(VALID_LAW_PROFILES) == 120
    assert math.isclose(sum(UNIFORM_JOINT_PRIOR), 1.0, rel_tol=1e-12)
    idx = {name: i for i, name in enumerate(LAW_NAMES)}
    for prof in VALID_LAW_PROFILES:
        if prof[idx["has_inverses"]]:
            assert prof[idx["has_identity"]]
            assert not prof[idx["idempotent"]]
        if prof[idx["f_involution"]]:
            assert prof[idx["f_bijective"]]


def test_exact_cell_posterior_calibration():
    be = ExactBayesianBackend(EFEParams(noise_floor=1e-6))
    syms = ["ka", "lu", "mo", "ne", "pi"]
    be.reset(syms, noise=0.05, combine_cost=1.0, transform_cost=0.5)

    # Exact channel: p_hit = 0.95 + 0.05/5 = 0.96, p_miss = 0.01
    act = ExperimentAction("combine", ("ka", "lu"), 1.0)
    surp = be.observe(act, "mo")
    assert math.isclose(surp, math.log(5.0), rel_tol=1e-6)
    q = be._op_q[0][1]
    assert math.isclose(q[2], 0.96, rel_tol=1e-6)
    assert math.isclose(q[0], 0.01, rel_tol=1e-6)

    # Second identical observation has very low surprise (-log(0.96*0.96 + 4*0.01*0.01))
    surp2 = be.observe(act, "mo")
    expected_pred = 0.96 * 0.96 + 4 * 0.01 * 0.01
    assert math.isclose(surp2, -math.log(expected_pred), rel_tol=1e-5)
    assert surp2 < 0.15


def test_unobserved_witnesses_give_neutral_bayes_factor():
    be = ExactBayesianBackend()
    syms = ["ka", "lu", "mo", "ne"]
    be.reset(syms, noise=0.05)

    # Before any observations, all log-BFs are identically 0.0
    cache = be._build_cache()
    assert all(math.isclose(v, 0.0, abs_tol=1e-12) for v in cache["lbfs"])

    # Observing only one side of a pair (ka, lu) leaves commutative BF neutral (0.0)
    be.observe(ExperimentAction("combine", ("ka", "lu"), 1.0), "mo")
    cache2 = be._build_cache()
    assert math.isclose(cache2["lbfs"][0], 0.0, abs_tol=1e-12)


def test_noise_outlier_vs_confirmed_counterexample():
    be = ExactBayesianBackend()
    syms = ["ka", "lu", "mo", "ne", "pi"]
    be.reset(syms, noise=0.05)

    # Confirm 4 of 5 diagonal idempotent cells (1 remains unobserved -> ~0.91)
    for s in syms[:4]:
        be.observe(ExperimentAction("combine", (s, s), 1.0), s)
    p_before = be.belief().law_probs["idempotent"]
    assert p_before > 0.88

    # Confirming the 5th cell too drives posterior > 0.99
    be_all = ExactBayesianBackend()
    be_all.reset(syms, noise=0.05)
    for s in syms:
        be_all.observe(ExperimentAction("combine", (s, s), 1.0), s)
    assert be_all.belief().law_probs["idempotent"] > 0.99

    # Single noisy mismatch on pi * pi -> ka lowers probability without collapsing to 0
    be.observe(ExperimentAction("combine", ("pi", "pi"), 1.0), "ka")
    p_after_one = be.belief().law_probs["idempotent"]
    assert 0.05 < p_after_one < p_before

    # If re-observation confirms a genuine counterexample (pi * pi -> ka again), falsifies cleanly
    be_counter = ExactBayesianBackend()
    be_counter.reset(syms, noise=0.05)
    for s in syms[:4]:
        be_counter.observe(ExperimentAction("combine", (s, s), 1.0), s)
    be_counter.observe(ExperimentAction("combine", ("pi", "pi"), 1.0), "ka")
    be_counter.observe(ExperimentAction("combine", ("pi", "pi"), 1.0), "ka")
    assert be_counter.belief().law_probs["idempotent"] < 0.05

    # Whereas if re-observation reveals the first mismatch was noise (pi * pi -> pi twice), recovers!
    be.observe(ExperimentAction("combine", ("pi", "pi"), 1.0), "pi")
    be.observe(ExperimentAction("combine", ("pi", "pi"), 1.0), "pi")
    assert be.belief().law_probs["idempotent"] > 0.90


def test_incremental_delta_matches_full_recompute_machine_precision():
    w = HiddenLawWorld(seed=17, noise=0.05, budget=500.0)
    be = ExactBayesianBackend()
    be.reset(w.symbols, noise=w.noise, combine_cost=w.combine_cost, transform_cost=w.transform_cost)
    rng = random.Random(99)
    actions = [ExperimentAction("transform", (a,), w.transform_cost) for a in w.symbols] + [
        ExperimentAction("combine", (a, b), w.combine_cost)
        for a in w.symbols
        for b in w.symbols
    ]
    for _ in range(30):
        a = rng.choice(actions)
        be.observe(a, a.execute(w))

    cache = be._build_cache()
    n = be._n
    max_diff = 0.0
    # Check both combine and transform counterfactuals
    for act in actions[:4] + actions[n : n + 8]:
        if act.kind == "combine":
            i, j = be._index[act.args[0]], be._index[act.args[1]]
            q_old, tot_old = be._op_q[i][j], be._op_totals[i][j]
            nc = list(be._op_counts[i][j])
            nc[0] += 1.0
            nt = tot_old + 1.0
            nq = be._cell_posterior_from_counts(nc, nt)
            t_set = set()
            for c in range(n):
                if be._op_totals[j][c] > 0.0 or (j == i and c == j):
                    t_set.add((i, j, c))
            for a in range(n):
                if be._op_totals[a][i] > 0.0 or (a == i and i == j):
                    t_set.add((a, i, j))
            for a, b in cache["inv_op"][i]:
                if be._op_totals[b][j] > 0.0 or (b == i and j == j):
                    t_set.add((a, b, j))
            for b, c in cache["inv_op"][j]:
                if be._op_totals[i][b] > 0.0 or (i == i and b == j):
                    t_set.add((i, b, c))
            h_set = set()
            if be._f_totals[i] > 0.0 and be._f_totals[j] > 0.0:
                h_set.add((i, j))
            for a in cache["inv_f"][i]:
                for b in cache["inv_f"][j]:
                    if be._op_totals[a][b] > 0.0 or (a == i and b == j):
                        h_set.add((a, b))
            inc = be._delta_lbfs_combine(
                cache, i, j, list(t_set), list(h_set), q_old, tot_old, nq, nt
            )
            be._op_q[i][j], be._op_totals[i][j] = nq, nt
            full = be._build_cache()["lbfs"]
            be._op_q[i][j], be._op_totals[i][j] = q_old, tot_old
        else:
            i = be._index[act.args[0]]
            q_old, tot_old = be._f_q[i], be._f_totals[i]
            nc = list(be._f_counts[i])
            nc[0] += 1.0
            nt = tot_old + 1.0
            nq = be._cell_posterior_from_counts(nc, nt)
            h_set = set()
            for b in range(n):
                if be._op_totals[i][b] > 0.0 and (be._f_totals[b] > 0.0 or b == i):
                    h_set.add((i, b))
            for a in range(n):
                if be._op_totals[a][i] > 0.0 and (be._f_totals[a] > 0.0 or a == i):
                    h_set.add((a, i))
            for a, b in cache["inv_op"][i]:
                if (be._f_totals[a] > 0.0 or a == i) and (be._f_totals[b] > 0.0 or b == i):
                    h_set.add((a, b))
            inc = be._delta_lbfs_transform(cache, i, list(h_set), q_old, tot_old, nq, nt)
            be._f_q[i], be._f_totals[i] = nq, nt
            full = be._build_cache()["lbfs"]
            be._f_q[i], be._f_totals[i] = q_old, tot_old
        for x, y in zip(inc, full):
            max_diff = max(max_diff, abs(x - y))
    assert max_diff < 1e-10


def test_efe_prefers_informative_over_redundant_and_respects_cost():
    be = ExactBayesianBackend()
    syms = ["ka", "lu", "mo"]
    be.reset(syms, noise=0.05, combine_cost=1.0, transform_cost=0.5)

    # Confirm (ka, ka) heavily so querying it again has near-zero epistemic value
    act_kaka = ExperimentAction("combine", ("ka", "ka"), 1.0)
    for _ in range(5):
        be.observe(act_kaka, "ka")

    scores = be.score_actions(remaining_budget=10.0)
    score_by_act = {s.action: s for s in scores}
    best = scores[0]
    assert best.action != act_kaka
    assert best.efe < score_by_act[act_kaka].efe
    assert score_by_act[act_kaka].law_eig_bits < 0.01

    # Respects tight remaining budget: when remaining=0.6, only transform (cost 0.5) is affordable
    tight_scores = be.score_actions(remaining_budget=0.6)
    assert tight_scores
    assert all(s.action.kind == "transform" for s in tight_scores)


def test_law_shift_causes_bayesian_surprise_spike_and_forgetting_adapts():
    w = HiddenLawWorld(seed=5, noise=0.0, budget=500.0)
    be_static = ExactBayesianBackend(EFEParams(noise_floor=1e-4, forgetting_factor=1.0))
    be_static.reset(w.symbols, noise=0.0)
    be_adapt = ExactBayesianBackend(EFEParams(noise_floor=1e-4, forgetting_factor=0.5))
    be_adapt.reset(w.symbols, noise=0.0)

    # Learn pre-shift table cleanly (2 passes)
    pre_surprises = []
    for _ in range(2):
        for a in w.symbols:
            for b in w.symbols:
                act = ExperimentAction("combine", (a, b), 1.0)
                y = act.execute(w)
                s = be_static.observe(act, y)
                be_adapt.observe(act, y)
                pre_surprises.append(s)

    settled_pre = pre_surprises[len(w.symbols) ** 2 :]
    assert max(settled_pre) < 0.01

    # Trigger Phase 3 law shift!
    w.shift_laws(shift_seed=77)
    post_surprises = []
    for a in w.symbols:
        for b in w.symbols:
            act = ExperimentAction("combine", (a, b), 1.0)
            y = act.execute(w)
            post_surprises.append(be_static.observe(act, y))
            be_adapt.observe(act, y)

    # Law shift must produce a clear Bayesian surprise spike
    assert max(post_surprises) > 5.0

    # With forgetting_factor=0.5, a few post-shift passes let the adaptive backend track the new truth
    for _ in range(3):
        for a in w.symbols:
            for b in w.symbols:
                act = ExperimentAction("combine", (a, b), 1.0)
                be_adapt.observe(act, act.execute(w))
        for a in w.symbols:
            act = ExperimentAction("transform", (a,), 0.5)
            be_adapt.observe(act, act.execute(w))
    new_truth = w.ground_truth()
    claims = be_adapt.belief().map_claims
    acc = sum(claims[k] == new_truth[k] for k in LAW_NAMES) / len(LAW_NAMES)
    assert acc >= 0.875


def test_bayes_optimal_claims_always_logically_consistent():
    rng = random.Random(7)
    be = ExactBayesianBackend()
    be.reset(["ka", "lu", "mo", "ne"], noise=0.05)
    for _ in range(20):
        weights = tuple(rng.random() for _ in range(NUM_VALID_PROFILES))
        be.set_prior(weights)
        claims = be.belief().map_claims
        if claims["has_inverses"]:
            assert claims["has_identity"]
            assert not claims["idempotent"]
        if claims["f_involution"]:
            assert claims["f_bijective"]
