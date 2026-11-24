"""Tests for Good's MDL & assembly-style structural memory."""

from __future__ import annotations

import math

from prometheus.core.blanket import ExperimentAction, LawBelief
from prometheus.core.inference import (
    NUM_VALID_PROFILES,
    PROFILE_INDEX,
    UNIFORM_JOINT_PRIOR,
    VALID_LAW_PROFILES,
    ExactBayesianBackend,
)
from prometheus.core.memory import AssemblyMemory, MemoryParams
from prometheus.lab.laws import LAW_NAMES


def _sharp_belief(law_flags: dict[str, bool], confidence: float = 0.96) -> LawBelief:
    key = "".join("1" if law_flags[name] else "0" for name in LAW_NAMES)
    target_idx = PROFILE_INDEX[key]
    rem = (1.0 - confidence) / (NUM_VALID_PROFILES - 1)
    joint = tuple(confidence if i == target_idx else rem for i in range(NUM_VALID_PROFILES))
    probs = {name: (confidence if law_flags[name] else 1.0 - confidence) for name in LAW_NAMES}
    ent = -sum(p * math.log2(p) for p in joint if p > 0)
    return LawBelief(
        law_probs=probs,
        joint_probs=joint,
        entropy_bits=ent,
        map_claims=dict(law_flags),
        identity_probs={},
    )


_GROUP_FLAGS = {
    "commutative": True,
    "associative": True,
    "has_identity": True,
    "has_inverses": True,
    "idempotent": False,
    "f_involution": True,
    "f_homomorphism": True,
    "f_bijective": True,
}

_RPS_FLAGS = {
    "commutative": True,
    "associative": False,
    "has_identity": False,
    "has_inverses": False,
    "idempotent": True,
    "f_involution": False,
    "f_homomorphism": False,
    "f_bijective": True,
}


def test_empty_memory_returns_uniform_prior():
    mem = AssemblyMemory()
    b = _sharp_belief(_GROUP_FLAGS)
    res = mem.recall(b, {}, n_symbols=5)
    assert res.activations == {}
    assert res.prior_joint == UNIFORM_JOINT_PRIOR
    assert res.compression_gain_bits == 0.0


def test_mdl_compression_gain_and_consolidation():
    mem = AssemblyMemory()
    b_group = _sharp_belief(_GROUP_FLAGS)
    feats = {"diag_fixpoint_rate": 0.2, "f_fixpoint_rate": 0.2, "row_uniqueness": 1.0, "identity_prob": 0.99}

    r1 = mem.consolidate(b_group, feats, n_symbols=5, episode_idx=0)
    assert r1.action == "spawned"
    assert len(mem.assemblies) == 1
    assert r1.compression_gain_bits > 0.0

    # A second similar episode reuses the same assembly and achieves strong MDL compression
    r2 = mem.consolidate(b_group, feats, n_symbols=5, episode_idx=1)
    assert r2.action == "updated"
    assert r2.assembly_id == r1.assembly_id
    assert len(mem.assemblies) == 1
    assert r2.compression_gain_bits > 1.5

    # Recall activates that assembly and shifts prior toward the learned profile
    rec = mem.recall(b_group, feats, n_symbols=5)
    assert rec.activations[r1.assembly_id] > 0.99
    target_idx = PROFILE_INDEX["".join("1" if _GROUP_FLAGS[k] else "0" for k in LAW_NAMES)]
    assert rec.prior_joint[target_idx] > UNIFORM_JOINT_PRIOR[target_idx] * 5.0


def test_distinct_structural_profiles_spawn_separate_assemblies():
    mem = AssemblyMemory(MemoryParams(spawn_mdl_threshold_bits=1.0, merge_kl_threshold=0.1))
    b_group = _sharp_belief(_GROUP_FLAGS)
    b_rps = _sharp_belief(_RPS_FLAGS)

    g_feats = {"diag_fixpoint_rate": 0.2, "f_fixpoint_rate": 0.2, "row_uniqueness": 1.0, "identity_prob": 0.99}
    r_feats = {"diag_fixpoint_rate": 1.0, "f_fixpoint_rate": 0.0, "row_uniqueness": 0.67, "identity_prob": 0.01}

    r1 = mem.consolidate(b_group, g_feats, n_symbols=5, episode_idx=0)
    r2 = mem.consolidate(b_rps, r_feats, n_symbols=3, episode_idx=1)
    assert r1.action == "spawned"
    assert r2.action == "spawned"
    assert len(mem.assemblies) == 2

    # Recalling with RPS-like belief activates the RPS assembly over the Group assembly
    rec = mem.recall(b_rps, r_feats, n_symbols=3)
    assert rec.activations[r2.assembly_id] > rec.activations[r1.assembly_id]


def test_low_confidence_episodes_skipped():
    mem = AssemblyMemory(MemoryParams(min_confidence_to_learn=0.80))
    uncertain = _sharp_belief(_GROUP_FLAGS, confidence=0.55)
    report = mem.consolidate(uncertain, {}, n_symbols=4, episode_idx=0)
    assert report.action == "skipped_low_confidence"
    assert len(mem.assemblies) == 0


def test_merge_and_prune_respect_max_assemblies():
    mem = AssemblyMemory(
        MemoryParams(
            spawn_mdl_threshold_bits=0.0,
            merge_kl_threshold=0.0,
            max_assemblies=3,
        )
    )
    # Create 5 distinct profiles; max_assemblies=3 must prune down to 3
    for idx, prof in enumerate(VALID_LAW_PROFILES[:5]):
        flags = {name: prof[k] for k, name in enumerate(LAW_NAMES)}
        mem.consolidate(_sharp_belief(flags), {}, n_symbols=4, episode_idx=idx)
    assert len(mem.assemblies) <= 3


def test_probabilistic_regeneration_overridden_by_strong_observations():
    # Train memory heavily on idempotent=False (_GROUP_FLAGS)
    mem = AssemblyMemory(MemoryParams(prior_blend=0.6))
    b_group = _sharp_belief(_GROUP_FLAGS)
    for ep in range(6):
        mem.consolidate(b_group, {}, n_symbols=3, episode_idx=ep)

    # Now observe a genuinely idempotent world (op[a, a] == a for all a, confirmed twice)
    be = ExactBayesianBackend()
    syms = ["ka", "lu", "mo"]
    be.reset(syms, noise=0.05)
    rec = mem.recall(be.belief(), be.feature_summary(), 3)
    be.set_prior(rec.prior_joint)
    assert be.belief().law_probs["idempotent"] < 0.4  # memory bias initially favors False

    for _ in range(2):
        for s in syms:
            be.observe(ExperimentAction("combine", (s, s), 1.0), s)
    # Concrete evidence must override the recalled memory prior
    assert be.belief().law_probs["idempotent"] > 0.90


def test_memory_checkpoint_roundtrip(tmp_path):
    mem = AssemblyMemory()
    mem.consolidate(_sharp_belief(_GROUP_FLAGS), {"diag_fixpoint_rate": 0.2}, n_symbols=5, episode_idx=0)
    mem.consolidate(_sharp_belief(_RPS_FLAGS), {"diag_fixpoint_rate": 1.0}, n_symbols=3, episode_idx=1)

    ckpt = tmp_path / "memory_state.json"
    mem.save(ckpt)
    loaded = AssemblyMemory.load(ckpt)
    assert loaded.to_dict() == mem.to_dict()
    assert len(loaded.assemblies) == 2
