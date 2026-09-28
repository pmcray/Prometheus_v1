"""Tests for the Phase 0 bake-off: harness, prompt, episode runner, models.

All tests mock the ModelClient / openai so no live server is required.
The tests verify:
  - Prompt construction and claim parsing (network-free)
  - EpisodeResult scoring logic
  - BakeoffConfig / BakeoffSummary construction
  - MODEL_REGISTRY integrity
  - Observation sampling fills the budget and respects max_obs
  - Error paths: malformed model reply, missing claims
"""

from __future__ import annotations

import json
import unittest.mock as mock

import pytest

from prometheus.bakeoff.models import BAKEOFF_MODELS, MODEL_REGISTRY, ModelSpec
from prometheus.bakeoff.runner import BakeoffConfig, BakeoffSummary, _summarise
from prometheus.harness.client import ModelClient, ModelClientError
from prometheus.harness.episode import EpisodeResult, _sample_observations, run_episode
from prometheus.harness.prompt import (
    SYSTEM_PROMPT,
    build_messages,
    parse_claims,
)
from prometheus.lab.laws import LAW_NAMES
from prometheus.lab.world import HiddenLawWorld


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ground_truth_claims(seed: int) -> dict[str, bool]:
    return HiddenLawWorld(seed=seed).ground_truth()


def _perfect_reply(seed: int) -> str:
    claims = _ground_truth_claims(seed)
    return json.dumps({"claims": claims, "reasoning": "test"})


def _mock_client(reply: str | Exception) -> ModelClient:
    """Return a ModelClient whose chat_json() returns *reply* (or raises)."""
    client = mock.MagicMock(spec=ModelClient)
    if isinstance(reply, Exception):
        client.chat_json.side_effect = reply
    else:
        client.chat_json.return_value = json.loads(reply)
    return client


# ---------------------------------------------------------------------------
# Prompt tests
# ---------------------------------------------------------------------------

def test_system_prompt_mentions_all_eight_laws():
    for name in LAW_NAMES:
        assert name in SYSTEM_PROMPT, f"Law '{name}' missing from system prompt"


def test_build_messages_returns_two_messages():
    world = HiddenLawWorld(seed=0)
    msgs = build_messages(world.symbols, [], world.remaining)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"


def test_build_messages_includes_symbols():
    world = HiddenLawWorld(seed=1)
    msgs = build_messages(world.symbols, [], world.remaining)
    for sym in world.symbols:
        assert sym in msgs[1]["content"]


def test_build_messages_includes_observations():
    world = HiddenLawWorld(seed=2, budget=20.0)
    s = world.symbols
    world.combine(s[0], s[1])
    world.transform(s[0])
    obs = [
        {"kind": "combine", "args": (s[0], s[1]), "result": s[0]},
        {"kind": "transform", "args": (s[0],), "result": s[1]},
    ]
    msgs = build_messages(world.symbols, obs, world.remaining)
    user = msgs[1]["content"]
    assert f"combine({s[0]}, {s[1]})" in user
    assert f"transform({s[0]})" in user


def test_parse_claims_returns_all_eight():
    raw = {"claims": {name: True for name in LAW_NAMES}}
    claims = parse_claims(raw)
    assert set(claims) == set(LAW_NAMES)


def test_parse_claims_raises_on_missing_key():
    raw = {"claims": {"commutative": True}}  # missing 7 laws
    with pytest.raises(ValueError, match="omitted claims"):
        parse_claims(raw)


def test_parse_claims_raises_on_non_dict():
    with pytest.raises(ValueError):
        parse_claims([1, 2, 3])


def test_parse_claims_raises_on_missing_claims_key():
    with pytest.raises(ValueError, match="Missing"):
        parse_claims({"reasoning": "hmm"})


def test_parse_claims_coerces_to_bool():
    raw = {"claims": {name: 1 for name in LAW_NAMES}}
    claims = parse_claims(raw)
    assert all(isinstance(v, bool) for v in claims.values())


# ---------------------------------------------------------------------------
# Observation sampler tests
# ---------------------------------------------------------------------------

def test_sample_observations_respects_max_obs():
    world = HiddenLawWorld(seed=5, budget=10_000.0)
    obs = _sample_observations(world, max_obs=10)
    assert len(obs) <= 10


def test_sample_observations_respects_budget():
    world = HiddenLawWorld(seed=5, budget=5.0)
    obs = _sample_observations(world, max_obs=1000)
    total_cost = sum(o["cost"] for o in obs)
    assert total_cost <= 5.0 + 1e-9


def test_sample_observations_returns_dicts_with_required_keys():
    world = HiddenLawWorld(seed=7, budget=50.0)
    obs = _sample_observations(world, max_obs=20)
    for o in obs:
        assert "kind" in o and "args" in o and "result" in o and "cost" in o
        assert o["kind"] in ("combine", "transform")


def test_sample_observations_cover_all_symbols():
    """Every symbol should appear in at least one observation."""
    world = HiddenLawWorld(seed=3, budget=10_000.0)
    obs = _sample_observations(world, max_obs=10_000)
    seen = set()
    for o in obs:
        seen.update(o["args"])
        seen.add(o["result"])
    for sym in world.symbols:
        assert sym in seen, f"Symbol {sym!r} never appeared"


# ---------------------------------------------------------------------------
# Episode runner tests
# ---------------------------------------------------------------------------

def test_run_episode_perfect_model():
    """A model that always returns ground truth scores 1.0."""
    world = HiddenLawWorld(seed=42)
    reply = _perfect_reply(42)
    client = _mock_client(reply)
    result = run_episode(world, client)
    assert result.error is None
    assert result.correct == len(LAW_NAMES)
    assert result.accuracy == 1.0
    assert result.seed == 42
    assert result.family == world.family


def test_run_episode_all_false_model():
    """A model that always claims False scores correctly only on False laws."""
    world = HiddenLawWorld(seed=42)
    truth = world.ground_truth()
    false_claims = {name: False for name in LAW_NAMES}
    client = _mock_client(json.dumps({"claims": false_claims}))
    result = run_episode(world, client)
    expected_correct = sum(1 for v in truth.values() if not v)
    assert result.correct == expected_correct


def test_run_episode_captures_model_error():
    world = HiddenLawWorld(seed=0)
    client = _mock_client(ModelClientError("server down"))
    result = run_episode(world, client)
    assert result.error is not None
    assert "server down" in result.error
    assert result.correct == 0


def test_run_episode_captures_parse_error():
    """Malformed JSON that parses but has wrong structure."""
    world = HiddenLawWorld(seed=1)
    client = _mock_client(json.dumps({"not_claims": {}}))
    # chat_json returns the dict directly in our mock
    result = run_episode(world, client)
    assert result.error is not None
    assert "Parse" in result.error or "Missing" in result.error


def test_run_episode_records_observations():
    world = HiddenLawWorld(seed=5)
    client = _mock_client(_perfect_reply(5))
    result = run_episode(world, client)
    assert result.observations_used > 0
    assert result.budget_spent > 0.0


def test_run_episode_result_to_dict():
    world = HiddenLawWorld(seed=10)
    client = _mock_client(_perfect_reply(10))
    result = run_episode(world, client)
    d = result.to_dict()
    assert "seed" in d and "accuracy" not in d   # accuracy is a property, not stored


# ---------------------------------------------------------------------------
# Model registry tests
# ---------------------------------------------------------------------------

def test_model_registry_has_required_candidates():
    for name in BAKEOFF_MODELS:
        assert name in MODEL_REGISTRY, f"BAKEOFF_MODELS entry {name!r} not in registry"


def test_model_registry_all_have_hf_id():
    for name, spec in MODEL_REGISTRY.items():
        assert spec.hf_id, f"ModelSpec {name!r} has empty hf_id"
        assert "/" in spec.hf_id, f"ModelSpec {name!r} hf_id looks wrong: {spec.hf_id!r}"


def test_model_registry_t4_feasibility_flags_set():
    # All bake-off candidates must claim T4 feasibility
    for name in BAKEOFF_MODELS:
        assert MODEL_REGISTRY[name].t4_feasible, (
            f"{name!r} is a bake-off candidate but t4_feasible=False"
        )


def test_bakeoff_config_model_spec_lookup():
    cfg = BakeoffConfig(model_name="qwen3-8b")
    spec = cfg.model_spec()
    assert spec.hf_id == "Qwen/Qwen3-8B"


def test_bakeoff_config_unknown_model_raises():
    cfg = BakeoffConfig(model_name="no-such-model")
    with pytest.raises(ValueError, match="Unknown model"):
        cfg.model_spec()


# ---------------------------------------------------------------------------
# Summarise helper tests
# ---------------------------------------------------------------------------

def _make_results(seeds: list[int], perfect: bool = True) -> list[EpisodeResult]:
    out = []
    for seed in seeds:
        w = HiddenLawWorld(seed=seed)
        truth = w.ground_truth()
        claims = truth if perfect else {k: False for k in truth}
        correct = sum(claims[k] == truth[k] for k in truth)
        out.append(EpisodeResult(
            seed=seed, family=w.family, n=len(w.symbols),
            claims=claims, truth=truth, correct=correct,
        ))
    return out


def test_summarise_perfect():
    results = _make_results(list(range(20)), perfect=True)
    summary = _summarise("test", results)
    assert summary.accuracy == pytest.approx(1.0)
    assert summary.n_errors == 0
    assert all(v == pytest.approx(1.0) for v in summary.per_law_accuracy.values())


def test_summarise_all_false():
    results = _make_results(list(range(20)), perfect=False)
    summary = _summarise("test", results)
    # All-False gets some laws right (those that are False in truth)
    assert 0.0 <= summary.accuracy <= 1.0


def test_summarise_empty():
    summary = _summarise("test", [])
    assert summary.accuracy == 0.0
    assert summary.n_episodes == 0


def test_summarise_per_family_coverage():
    # Use enough seeds to cover multiple families
    results = _make_results(list(range(50)), perfect=True)
    summary = _summarise("test", results)
    assert len(summary.per_family_accuracy) >= 2


def test_summarise_n_errors():
    results = _make_results(list(range(5)), perfect=True)
    results[2].error = "oops"
    summary = _summarise("test", results)
    assert summary.n_errors == 1


def test_bakeoff_summary_to_dict():
    results = _make_results(list(range(5)))
    summary = _summarise("qwen3-8b", results)
    d = summary.to_dict()
    assert d["model_name"] == "qwen3-8b"
    assert "accuracy" in d
    assert "per_law_accuracy" in d
    assert "per_family_accuracy" in d
