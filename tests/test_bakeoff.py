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

from prometheus.bakeoff.models import (
    BAKEOFF_MODELS,
    MODEL_REGISTRY,
    ModelSpec,
    vllm_flags_for_gpu,
)
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
    assert all(isinstance(v, bool) and v is True for v in claims.values())


def test_parse_claims_handles_string_booleans_and_rejects_garbage():
    raw = {"claims": {name: "false" for name in LAW_NAMES}}
    claims = parse_claims(raw)
    assert all(v is False for v in claims.values())

    raw_true = {"claims": {name: "True" for name in LAW_NAMES}}
    assert all(v is True for v in parse_claims(raw_true).values())

    bad = {"claims": {name: "maybe" for name in LAW_NAMES}}
    with pytest.raises(ValueError, match="Invalid boolean"):
        parse_claims(bad)


def test_build_messages_includes_noise_when_provided():
    world = HiddenLawWorld(seed=1, noise=0.05)
    msgs = build_messages(world.symbols, [], world.remaining, noise=world.noise)
    assert "5.0%" in msgs[1]["content"]


def test_model_client_strips_think_block_and_reuses_client():
    fake_resp = mock.MagicMock()
    fake_resp.choices = [mock.MagicMock()]
    fake_resp.choices[0].message.content = (
        "<think>Let me check associativity...</think>\n```json\n{\"ok\": true}\n```"
    )
    fake_openai = mock.MagicMock()
    fake_instance = fake_openai.OpenAI.return_value
    fake_instance.chat.completions.create.return_value = fake_resp

    with mock.patch("prometheus.harness.client._openai", return_value=fake_openai):
        client = ModelClient(base_url="http://localhost:8000/v1", model_id="Qwen/Qwen3-8B")
        assert client.chat_json([{"role": "user", "content": "hi"}]) == {"ok": True}
        assert client.chat_json([{"role": "user", "content": "hi2"}]) == {"ok": True}
        # OpenAI client should be instantiated once and reused across calls.
        assert fake_openai.OpenAI.call_count == 1


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


def test_sample_observations_repeats_when_noisy():
    """Noisy worlds get repeated passes (up to 3x) so outliers can be outvoted."""
    w_clean = HiddenLawWorld(seed=3, noise=0.0, budget=10_000.0)
    w_noisy = HiddenLawWorld(seed=3, noise=0.05, budget=10_000.0)
    n = len(w_clean.symbols)
    single_pass = n * n + n
    assert len(_sample_observations(w_clean, max_obs=500)) == single_pass
    assert len(_sample_observations(w_noisy, max_obs=500)) == 3 * single_pass


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
    results[2].correct = 0
    results[2].claims = {k: False for k in LAW_NAMES}
    summary = _summarise("test", results)
    assert summary.n_errors == 1
    # Errored episodes must count as 0 in BOTH overall accuracy AND per_law_accuracy.
    assert summary.accuracy == pytest.approx(4 / 5)
    assert all(v == pytest.approx(4 / 5) for v in summary.per_law_accuracy.values())


def test_bakeoff_config_seeds_auto_sized_and_validated():
    cfg = BakeoffConfig(model_name="qwen3-8b", n_episodes=250)
    assert len(cfg.seeds) == 250
    with pytest.raises(ValueError, match="seeds length"):
        BakeoffConfig(model_name="qwen3-8b", n_episodes=10, seeds=[0, 1, 2])


def test_bakeoff_runner_checkpoint_resume_and_artifact_name(tmp_path):
    """BakeoffRunner persists checkpoints, resumes the MLflow run, and logs results.json."""
    from prometheus.bakeoff.runner import BakeoffRunner

    ckpt = tmp_path / "ckpt.json"
    logged_artifacts: list[str] = []

    class FakeRun:
        class info:
            run_id = "run-abc-123"
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False

    fake_mlflow = mock.MagicMock()
    fake_mlflow.start_run.return_value = FakeRun()
    fake_mlflow.log_artifact.side_effect = lambda path, **kw: logged_artifacts.append(
        path.split("/")[-1]
    )

    def _reply_for_messages(messages, **kw):
        # Return all-False claims deterministically
        return {"claims": {k: False for k in LAW_NAMES}, "reasoning": "ok"}

    fake_client = mock.MagicMock(spec=ModelClient)
    fake_client.chat_json.side_effect = _reply_for_messages

    cfg = BakeoffConfig(
        model_name="qwen3-8b",
        n_episodes=4,
        seeds=[0, 1, 2, 3],
        checkpoint_path=str(ckpt),
    )

    with (
        mock.patch("prometheus.bakeoff.runner._get_mlflow", return_value=fake_mlflow),
        mock.patch.object(BakeoffRunner, "_make_client", return_value=fake_client),
    ):
        # Interrupt after 2 episodes via on_episode callback
        def _stop_at_two(idx, res):
            if idx == 1:
                raise RuntimeError("simulated restart")

        with pytest.raises(RuntimeError, match="simulated restart"):
            BakeoffRunner(cfg).run(on_episode=_stop_at_two)

        saved = json.loads(ckpt.read_text())
        assert saved["next_episode"] == 2
        assert saved["mlflow_run_id"] == "run-abc-123"

        # Resume and finish remaining 2 episodes
        summary = BakeoffRunner(cfg).run()
        assert summary.n_episodes == 4
        assert summary.n_errors == 0
        # Second start_run call must resume the same MLflow run_id
        assert fake_mlflow.start_run.call_args_list[-1].kwargs == {"run_id": "run-abc-123"}
        assert logged_artifacts == ["results.json"]


def test_bakeoff_summary_to_dict():
    results = _make_results(list(range(5)))
    summary = _summarise("qwen3-8b", results)
    d = summary.to_dict()
    assert d["model_name"] == "qwen3-8b"
    assert "accuracy" in d
    assert "per_law_accuracy" in d
    assert "per_family_accuracy" in d


# ---------------------------------------------------------------------------
# Prover harness & cheat-detection tests (Phase 0 step 5b)
# ---------------------------------------------------------------------------

def test_prover_validate_accepts_ground_truth_proof():
    from prometheus.harness.prover import validate_prover_source
    from prometheus.lab.lean_layer import world_lean_source

    w = HiddenLawWorld(seed=7)
    template = world_lean_source(w, include_proofs=False)
    proof = world_lean_source(w, include_proofs=True)
    assert validate_prover_source(template, proof) is None


@pytest.mark.parametrize("bad_tactic", ["sorry", "admit", "native_decide"])
def test_redteam_prover_rejects_cheat_primitives(bad_tactic):
    from prometheus.harness.prover import validate_prover_source
    from prometheus.lab.lean_layer import world_lean_source

    w = HiddenLawWorld(seed=7)
    template = world_lean_source(w, include_proofs=False)
    cheated = world_lean_source(w, include_proofs=True).replace("by decide", f"by {bad_tactic}", 1)
    err = validate_prover_source(template, cheated)
    assert err is not None and bad_tactic in err


def test_redteam_prover_rejects_altered_theorem_or_world_table():
    from prometheus.harness.prover import validate_prover_source
    from prometheus.lab.lean_layer import world_lean_source

    w = HiddenLawWorld(seed=7)
    template = world_lean_source(w, include_proofs=False)
    valid = world_lean_source(w, include_proofs=True)

    # Weakening a theorem statement
    weakened = valid.replace("theorem ", "theorem fake_", 1)
    assert validate_prover_source(template, weakened) is not None

    # Tampering with opTable
    tampered_table = valid.replace("def op (a b : Fin N) : Fin N := (opTable[a]!)[b]!",
                                   "def op (a b : Fin N) : Fin N := a")
    assert validate_prover_source(template, tampered_table) is not None


def test_run_prover_episode_rejects_cheat_without_calling_lean():
    from prometheus.harness.prover import run_prover_episode
    from prometheus.lab.lean_layer import world_lean_source

    w = HiddenLawWorld(seed=7)
    template = world_lean_source(w, include_proofs=False)
    fake_client = mock.MagicMock(spec=ModelClient)
    fake_client.chat.return_value = f"```lean\n{template}\n```"  # still contains sorry
    res = run_prover_episode(w, fake_client)
    assert res.proved is False
    assert res.error is not None and "sorry" in res.error
    assert res.to_dict()["seed"] == 7


def test_prover_strip_lean_fences_extracts_final_block_after_proof_plan():
    from prometheus.harness.prover import _strip_lean_fences

    raw = (
        "Proof plan:\nWe check each law using `decide`.\n"
        "```lean\n-- sketch snippet\n```\n"
        "Here is the complete Lean 4 proof:\n"
        "```lean4\nabbrev N := 3\ntheorem t : True := by decide\n```\n"
    )
    assert _strip_lean_fences(raw) == "abbrev N := 3\ntheorem t : True := by decide"


def test_model_client_health_check_uses_models_list():
    fake_openai = mock.MagicMock()
    fake_instance = fake_openai.OpenAI.return_value

    m_ok = mock.MagicMock()
    m_ok.id = "Qwen/Qwen3-8B"
    fake_instance.models.list.return_value = mock.MagicMock(data=[m_ok])

    with mock.patch("prometheus.harness.client._openai", return_value=fake_openai):
        client_ok = ModelClient(base_url="http://localhost:8000/v1", model_id="Qwen/Qwen3-8B")
        assert client_ok.health_check() is True
        # Wrong model served on localhost:8000 must fail health check
        client_wrong = ModelClient(base_url="http://localhost:8000/v1", model_id="openai/gpt-oss-20b")
        assert client_wrong.health_check() is False
        # Connection failure must return False
        fake_instance.models.list.side_effect = RuntimeError("Connection error.")
        assert client_ok.health_check() is False


def test_bakeoff_runner_fails_fast_on_unhealthy_server(tmp_path):
    from prometheus.bakeoff.runner import BakeoffRunner

    ckpt = tmp_path / "ckpt.json"
    fake_client = mock.MagicMock(spec=ModelClient)
    fake_client.health_check.return_value = False
    cfg = BakeoffConfig(
        model_name="qwen3-8b",
        n_episodes=5,
        checkpoint_path=str(ckpt),
    )
    with mock.patch.object(BakeoffRunner, "_make_client", return_value=fake_client):
        with pytest.raises(ModelClientError, match="failed health check"):
            BakeoffRunner(cfg).run()
    # Checkpoint must not be created or polluted with error episodes
    assert not ckpt.exists()


def test_bakeoff_runner_recovers_if_resumed_mlflow_run_missing(tmp_path):
    from prometheus.bakeoff.runner import BakeoffRunner

    ckpt = tmp_path / "ckpt.json"
    ckpt.write_text(
        json.dumps({
            "model_name": "qwen3-8b",
            "next_episode": 0,
            "mlflow_run_id": "stale-run-id",
            "results": {},
        })
    )

    class FakeRun:
        class info:
            run_id = "fresh-run-id"
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False

    fake_mlflow = mock.MagicMock()
    fake_mlflow.get_run.side_effect = RuntimeError("Run 'stale-run-id' not found")
    fake_mlflow.start_run.return_value = FakeRun()

    fake_client = mock.MagicMock(spec=ModelClient)
    fake_client.health_check.return_value = True
    fake_client.chat_json.return_value = {"claims": {k: False for k in LAW_NAMES}}

    cfg = BakeoffConfig(
        model_name="qwen3-8b",
        n_episodes=1,
        seeds=[0],
        checkpoint_path=str(ckpt),
    )
    with (
        mock.patch("prometheus.bakeoff.runner._get_mlflow", return_value=fake_mlflow),
        mock.patch.object(BakeoffRunner, "_make_client", return_value=fake_client),
    ):
        summary = BakeoffRunner(cfg).run()
    assert summary.n_episodes == 1
    assert fake_mlflow.start_run.call_args.kwargs.get("run_id") is None
    assert json.loads(ckpt.read_text())["mlflow_run_id"] == "fresh-run-id"


def test_vllm_flags_for_gpu_selects_bf16_on_l4_a100_g4_and_t4_fallback():
    spec = MODEL_REGISTRY["qwen3-8b"]
    t4_flags = vllm_flags_for_gpu(spec, compute_cap=(7, 5), vram_gb=15.0)
    assert "--quantization" in t4_flags and "bitsandbytes" in t4_flags

    l4_flags = vllm_flags_for_gpu(spec, compute_cap=(8, 9), vram_gb=24.0)
    assert "--dtype" in l4_flags and "bfloat16" in l4_flags
    assert "bitsandbytes" not in l4_flags

    pinned = ModelSpec(
        name="pinned-test",
        hf_id="Qwen/Qwen3-8B",
        revision="abc123def456",
        vllm_flags=("--dtype", "half"),
        bf16_vllm_flags=("--dtype", "bfloat16"),
    )
    flags_pinned = vllm_flags_for_gpu(pinned, compute_cap=(8, 0), vram_gb=40.0)
    assert flags_pinned == ("--dtype", "bfloat16", "--revision", "abc123def456")


