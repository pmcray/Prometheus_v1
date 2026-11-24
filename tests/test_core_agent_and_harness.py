"""Tests for CognitiveAgent, HarnessLoader, Governor->Gate->Core E2E, and red-team checks."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from conftest import Clock, PINS, RUN, claims as make_claims, make_proposal
from prometheus.common.eventlog import EventLog, verify_log
from prometheus.common.variant import write_bundle
from prometheus.core import (
    AssemblyMemory,
    CognitiveAgent,
    EFEParams,
    EssentialBounds,
    ExactBayesianBackend,
    ExperimentAction,
    HarnessValidationError,
    LoadedHarness,
    MeshConfig,
    TransferHint,
    load_harness,
    run_harness_evaluator,
    run_harness_tool,
)
from prometheus.governor.keys import generate_private_key, public_key_hex
from prometheus.governor.policy import Policy
from prometheus.governor.service import GENESIS_PARENT, Governor
from prometheus.harness.client import ModelClientError
from prometheus.lab.audit import score_claims
from prometheus.lab.laws import LAW_NAMES
from prometheus.lab.world import HiddenLawWorld
from prometheus.observatory import collect_snapshot
from prometheus.runtime.gate import Gate


# ---------------------------------------------------------------------------
# Quantitative benchmarks (Friston EFE vs random, budget savings, MDL)
# ---------------------------------------------------------------------------

def test_cognitive_agent_high_accuracy_and_budget_savings():
    seeds = list(range(20))
    agent = CognitiveAgent(rng_seed=42)
    hits = 0
    total = 0
    spent_list: list[float] = []
    for idx, s in enumerate(seeds):
        w = HiddenLawWorld(seed=s, noise=0.05, budget=500.0)
        res = agent.run_episode(w, learn=True, episode_idx=idx)
        truth = w.ground_truth()
        hits += sum(res.claims[k] == truth[k] for k in LAW_NAMES)
        total += len(LAW_NAMES)
        spent_list.append(res.budget_spent)

    acc = hits / total
    mean_spent = sum(spent_list) / len(spent_list)
    assert acc >= 0.95, f"expected >= 0.95 accuracy at budget=500, got {acc:.4f}"
    # Passive 3-pass sampler spends ~100-150+; EFE early stopping stays well below 100
    assert mean_spent < 100.0, f"expected early-stopping budget savings, mean_spent={mean_spent:.1f}"
    assert len(agent.memory.assemblies) >= 1


def test_cognitive_agent_beats_random_baseline_under_tight_budget():
    seeds = list(range(24))
    efe_agent = CognitiveAgent(rng_seed=7)
    efe_hits = 0
    rand_hits = 0
    total = len(seeds) * len(LAW_NAMES)
    rng = random.Random(123)

    for idx, s in enumerate(seeds):
        w_efe = HiddenLawWorld(seed=s, noise=0.05, budget=25.0)
        res = efe_agent.run_episode(w_efe, learn=True, episode_idx=idx)
        truth = w_efe.ground_truth()
        efe_hits += sum(res.claims[k] == truth[k] for k in LAW_NAMES)

        w_rand = HiddenLawWorld(seed=s, noise=0.05, budget=25.0)
        be = ExactBayesianBackend()
        be.reset(
            w_rand.symbols,
            noise=w_rand.noise,
            combine_cost=w_rand.combine_cost,
            transform_cost=w_rand.transform_cost,
        )
        all_acts = [ExperimentAction("transform", (a,), w_rand.transform_cost) for a in w_rand.symbols] + [
            ExperimentAction("combine", (a, b), w_rand.combine_cost)
            for a in w_rand.symbols
            for b in w_rand.symbols
        ]
        while True:
            aff = [a for a in all_acts if a.cost <= w_rand.remaining + 1e-9]
            if not aff:
                break
            act = rng.choice(aff)
            be.observe(act, act.execute(w_rand))
        rand_claims = be.belief().map_claims
        rand_hits += sum(rand_claims[k] == truth[k] for k in LAW_NAMES)

    efe_acc = efe_hits / total
    rand_acc = rand_hits / total
    assert efe_acc > rand_acc + 0.03, (
        f"expected EFE ({efe_acc:.3f}) to clearly outperform random ({rand_acc:.3f}) at budget=25"
    )


# ---------------------------------------------------------------------------
# Hybrid LLM advisor & Phase 2/3 extension seams
# ---------------------------------------------------------------------------

class _StubClient:
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls: list[list[dict[str, str]]] = []

    def chat_json(self, messages):
        self.calls.append(messages)
        item = self._replies[min(len(self.calls) - 1, len(self._replies) - 1)]
        if isinstance(item, Exception):
            raise item
        return item


def test_hybrid_llm_advisor_valid_suggestion_and_call_cap():
    w = HiddenLawWorld(seed=3, noise=0.05, budget=40.0)
    s0 = w.symbols[0]
    stub = _StubClient([
        {
            "next_experiment": {"kind": "transform", "args": [s0]},
            "claims": {k: True for k in LAW_NAMES},
            "reasoning": "Testing unary map first.",
        }
    ])
    harness = LoadedHarness(
        mesh=MeshConfig(
            use_llm_advisor=True,
            llm_entropy_trigger_bits=0.5,
            max_llm_calls_per_episode=2,
            llm_blend_weight=0.2,
            max_steps_per_episode=12,
        ),
        prompts={"advisor": "CUSTOM ADVISOR SYSTEM PROMPT"},
    )
    agent = CognitiveAgent(harness=harness, client=stub)  # type: ignore[arg-type]
    res = agent.run_episode(w, learn=False)
    assert res.metrics.llm_calls == 2
    assert len(stub.calls) == 2
    assert stub.calls[0][0]["content"] == "CUSTOM ADVISOR SYSTEM PROMPT"
    assert res.llm_notes == "Testing unary map first."

    # With use_llm_advisor=False, client is never called
    stub2 = _StubClient([{"reasoning": "unused"}])
    agent2 = CognitiveAgent(harness=LoadedHarness(mesh=MeshConfig(use_llm_advisor=False)), client=stub2)  # type: ignore[arg-type]
    agent2.run_episode(HiddenLawWorld(seed=3, noise=0.05, budget=20.0), learn=False)
    assert len(stub2.calls) == 0


def test_hybrid_llm_advisor_malformed_reply_falls_back_gracefully():
    w = HiddenLawWorld(seed=4, noise=0.05, budget=25.0)
    stub = _StubClient([ModelClientError("connection refused"), {"not_valid": 123}])
    harness = LoadedHarness(
        mesh=MeshConfig(use_llm_advisor=True, llm_entropy_trigger_bits=0.5, max_llm_calls_per_episode=2)
    )
    agent = CognitiveAgent(harness=harness, client=stub)  # type: ignore[arg-type]
    res = agent.run_episode(w, learn=False)
    assert res.error is None
    assert set(res.claims.keys()) == set(LAW_NAMES)


def test_phase2_transfer_and_phase3_monitor_seams():
    class _SpyTransfer:
        def __init__(self):
            self.called = 0

        def propose_transfer(self, belief, observations, assemblies):
            self.called += 1
            return TransferHint(source_assembly_id="asm-hint", confidence=0.5)

    class _SpyMonitor:
        def __init__(self):
            self.records = []

        def record(self, metrics):
            self.records.append(metrics)
            ok, _ = metrics.check_bounds(EssentialBounds())
            return not ok

        def violations(self):
            return []

    transfer = _SpyTransfer()
    monitor = _SpyMonitor()
    agent = CognitiveAgent(transfer=transfer, monitor=monitor)
    agent.run_episode(HiddenLawWorld(seed=2, noise=0.05, budget=30.0), learn=False)
    assert transfer.called >= 1
    assert len(monitor.records) == 1


# ---------------------------------------------------------------------------
# Harness loader + Governor -> Gate -> Core -> Observatory E2E
# ---------------------------------------------------------------------------

def test_harness_loader_defaults_and_valid_overrides(tmp_path):
    default_h = load_harness(None)
    assert default_h.loaded_files == ()
    assert default_h.efe_params == EFEParams()

    hdir = tmp_path / "harness"
    (hdir / "memory_params").mkdir(parents=True)
    (hdir / "mesh").mkdir(parents=True)
    (hdir / "prompts").mkdir(parents=True)
    (hdir / "tools").mkdir(parents=True)
    (hdir / "evaluators").mkdir(parents=True)

    (hdir / "memory_params" / "efe_params.json").write_text(
        json.dumps({"beta_law": 6.0, "lambda_cost": 0.12, "stop_entropy_bits": 0.2})
    )
    (hdir / "memory_params" / "memory_params.json").write_text(
        json.dumps({"prior_blend": 0.5, "max_assemblies": 12})
    )
    (hdir / "mesh" / "config.json").write_text(
        json.dumps({"use_llm_advisor": True, "max_steps_per_episode": 60})
    )
    (hdir / "prompts" / "advisor.md").write_text("Custom prompt body")
    (hdir / "tools" / "strategy.py").write_text(
        "cands = inputs['top_candidates']\nreturn_value = {'kind': cands[0]['kind'], 'args': cands[0]['args']}\n"
    )
    (hdir / "evaluators" / "score.py").write_text("return_value = 0.875\n")

    loaded = load_harness(hdir)
    assert loaded.efe_params.beta_law == 6.0
    assert loaded.memory_params.max_assemblies == 12
    assert loaded.mesh.use_llm_advisor is True
    assert loaded.prompts["advisor"] == "Custom prompt body"
    assert loaded.tool_source is not None
    assert loaded.evaluator_source is not None

    score, err = run_harness_evaluator(loaded.evaluator_source, {})
    assert err is None and score == pytest.approx(0.875)


def test_phase1_governor_gate_harness_reload_and_observatory_e2e(tmp_path):
    clock = Clock()
    gov_home = tmp_path / "gov_home"
    drive_root = tmp_path / "drive"
    exchange = drive_root / RUN
    install_root = tmp_path / "installed"
    gov_home.mkdir(parents=True)
    exchange.mkdir(parents=True)

    audit_seeds = [1000, 1001, 1002, 1003, 1004]
    Policy(
        run_id=RUN,
        audit_seeds=audit_seeds,
        model_pins=PINS,
        require_ready_marker=True,
    ).save(gov_home / "policy.json")
    key = generate_private_key()
    pub_hex = public_key_hex(key)
    (drive_root / "governor_public_key.txt").write_text(pub_hex + "\n")

    gov = Governor(gov_home, exchange, key, clock=clock)
    # Weak baseline (all False)
    base_claims = {str(s): {k: False for k in LAW_NAMES} for s in audit_seeds}
    baseline = make_proposal(
        exchange, "__baseline__", generation=0, parent=GENESIS_PARENT,
        claims_obj=base_claims, proxy=0.45,
    )
    gov.init_run(baseline.root)
    gov.heartbeat()
    baseline_hash = (exchange / "baseline_hash.txt").read_text().strip()

    gate = Gate(
        governor_public_key_hex=pub_hex,
        run_id=RUN,
        exchange=exchange,
        install_root=install_root,
        baseline_hash=baseline_hash,
        clock=clock,
        state_path=exchange / "gate_state.json",
    )

    # Gen-1 candidate harness files (all under allowed prefixes)
    gen1_files = {
        "harness/memory_params/efe_params.json": json.dumps({"beta_law": 5.0, "stop_entropy_bits": 0.15}),
        "harness/mesh/config.json": json.dumps({"use_tool_hook": True, "max_steps_per_episode": 80}),
        "harness/prompts/advisor.md": "Gen-1 advisor prompt",
        "harness/tools/strategy.py": (
            "cands = inputs['top_candidates']\n"
            "return_value = {'kind': cands[0]['kind'], 'args': cands[0]['args']} if cands else None\n"
        ),
        "harness/evaluators/score.py": "return_value = float(inputs['mean_confidence'])\n",
    }

    # Evaluate audit seeds using CognitiveAgent (read-only memory, learn=False)
    agent = CognitiveAgent(rng_seed=11)
    # Train on 2 training worlds first, then verify audit evaluation does NOT mutate memory
    for i in range(2):
        agent.run_episode(HiddenLawWorld(seed=i, noise=0.05, budget=200.0), learn=True, episode_idx=i)
    pre_audit_eps = agent.memory.episodes_consolidated
    audit_claims = agent.evaluate_audit_seeds(audit_seeds, noise=0.05, budget=300.0)
    assert agent.memory.episodes_consolidated == pre_audit_eps
    assert score_claims(audit_claims, audit_seeds) >= 0.90

    # Propose Gen-1 bundle -> Governor approves -> Gate installs -> load_harness reloads live files!
    prop_dir = exchange / "proposals" / "g1"
    bundle = write_bundle(
        prop_dir,
        {
            "run_id": RUN,
            "generation": 1,
            "parent_hash": gate.installed_hash,
            "model_pins": PINS,
            "proxy_score": 0.92,
        },
        gen1_files,
    )
    (prop_dir / "audit_claims.json").write_text(json.dumps(audit_claims))
    (prop_dir / ".ready").write_text(bundle.variant_hash + "\n")

    decisions = gov.step()
    assert len(decisions) == 1 and decisions[0]["body"]["decision"] == "approve", decisions
    gate.install(prop_dir)
    assert gate.generation == 1

    # Reload installed Gen-1 harness and run logged episode + save memory_state.json
    installed_harness = load_harness(install_root / "current" / "harness")
    assert installed_harness.efe_params.beta_law == 5.0
    assert installed_harness.prompts["advisor"] == "Gen-1 advisor prompt"

    log = EventLog(exchange / "logs" / "agent.jsonl", clock=clock)
    agent_g1 = CognitiveAgent(harness=installed_harness, memory=agent.memory, event_log=log)
    res = agent_g1.run_episode(HiddenLawWorld(seed=10, noise=0.05, budget=200.0), learn=True, episode_idx=2)
    assert res.metrics.sandbox_errors == 0
    agent_g1.memory.save(exchange / "memory_state.json")
    assert verify_log(exchange / "logs" / "agent.jsonl").ok

    # Observatory collects Phase 1 memory & cognitive telemetry cleanly
    snap = collect_snapshot(exchange, governor_home=gov_home, prometheus_dir=drive_root, now=clock())
    assert snap["overall_status"] == "HEALTHY"
    assert snap["memory"] is not None and snap["memory"]["n_assemblies"] >= 1
    assert snap["cognitive"] is not None and snap["cognitive"]["episodes"] == 1


# ---------------------------------------------------------------------------
# Red-team security tests
# ---------------------------------------------------------------------------

def test_redteam_core_never_touches_ground_truth():
    """Verify CognitiveAgent, ExactBayesianBackend, and AssemblyMemory never read forbidden attrs."""
    real_world = HiddenLawWorld(seed=12, noise=0.05, budget=60.0)
    forbidden_accesses: list[str] = []

    class _TrapWorldView:
        def __init__(self, w: HiddenLawWorld) -> None:
            self._w = w

        @property
        def symbols(self):
            return list(self._w.symbols)

        @property
        def noise(self):
            return self._w.noise

        @property
        def budget(self):
            return self._w.budget

        @property
        def remaining(self):
            return self._w.remaining

        @property
        def spent(self):
            return self._w.spent

        @property
        def combine_cost(self):
            return self._w.combine_cost

        @property
        def transform_cost(self):
            return self._w.transform_cost

        def combine(self, a: str, b: str) -> str:
            return self._w.combine(a, b)

        def transform(self, a: str) -> str:
            return self._w.transform(a)

        def __getattr__(self, name: str):
            forbidden_accesses.append(name)
            raise AttributeError(f"forbidden ground-truth access: {name}")

    agent = CognitiveAgent()
    res = agent.run_episode(_TrapWorldView(real_world), learn=True)  # type: ignore[arg-type]
    assert forbidden_accesses == []
    assert res.error is None


def test_redteam_harness_loader_rejects_nan_inf_bool_and_out_of_range(tmp_path):
    bad_payloads = [
        ("memory_params/efe_params.json", '{"beta_law": true}'),
        ("memory_params/efe_params.json", '{"beta_law": -1.0}'),
        ("memory_params/efe_params.json", '{"beta_law": NaN}'),
        ("memory_params/efe_params.json", '{"forgetting_factor": 1.5}'),
        ("memory_params/efe_params.json", '{"violation_grid": [0.0]}'),
        ("memory_params/memory_params.json", '{"prior_blend": 1.0}'),
        ("memory_params/memory_params.json", '{"max_assemblies": 9999}'),
        ("mesh/config.json", '{"max_steps_per_episode": true}'),
        ("mesh/config.json", '{"use_llm_advisor": "yes"}'),
    ]
    for i, (rel, content) in enumerate(bad_payloads):
        hdir = tmp_path / f"bad_{i}"
        p = hdir / rel
        p.parent.mkdir(parents=True)
        p.write_text(content)
        with pytest.raises(HarnessValidationError):
            load_harness(hdir)


def test_redteam_harness_loader_rejects_symlinks_and_path_escape(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP_SECRET")

    hdir = tmp_path / "harness"
    (hdir / "prompts").mkdir(parents=True)
    (hdir / "prompts" / "advisor.md").symlink_to(secret)
    with pytest.raises(HarnessValidationError, match="symlink"):
        load_harness(hdir)


def test_redteam_harness_tool_sandbox_escape_blocked(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("DO_NOT_LEAK")
    malicious_tool = f"""
import os
with open({str(secret)!r}) as fh:
    return_value = fh.read()
"""
    res = run_harness_tool(malicious_tool, {"top_candidates": []})
    assert res.error is not None
    assert res.return_value is None

    # Inside CognitiveAgent, a malicious tool is caught safely and EFE takes over cleanly
    harness = LoadedHarness(
        mesh=MeshConfig(use_tool_hook=True, max_steps_per_episode=8),
        tool_source=malicious_tool,
    )
    agent = CognitiveAgent(harness=harness)
    ep = agent.run_episode(HiddenLawWorld(seed=1, noise=0.05, budget=20.0), learn=False)
    assert ep.error is None
    assert ep.metrics.sandbox_errors >= 1
    assert ep.observations_used >= 1


def test_redteam_harness_evaluator_sandbox_escape_and_nan_blocked():
    score, err = run_harness_evaluator("import socket; return_value = 1.0", {})
    assert score is None and err is not None

    score2, err2 = run_harness_evaluator("return_value = float('nan')", {})
    assert score2 is None and err2 is not None

    score3, err3 = run_harness_evaluator("return_value = True", {})
    assert score3 is None and err3 is not None
