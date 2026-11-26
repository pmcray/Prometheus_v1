"""Phase 3 tests: law-shift stress recovery, Lean 4.9 prover bake-off (pass@k),
pinned multi-generation self-improvement (run-002), Observatory Phase 3 cards,
and red-team security checks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import Clock, make_proposal
from prometheus.bakeoff import (
    PROVER_BAKEOFF_MODELS,
    ProverBakeoffConfig,
    ProverBakeoffRunner,
)
from prometheus.common.eventlog import EventLog, verify_log
from prometheus.common.variant import write_bundle
from prometheus.core import (
    DEFAULT_RUN002_MODEL_PINS,
    AshbyConfig,
    AssemblyMemory,
    CognitiveAgent,
    EFEParams,
    HomeostaticMonitor,
    LoadedHarness,
    MultiGenConfig,
    MultiGenRunner,
    TransferParams,
    build_pinned_baseline_bundle,
    load_harness,
    run_law_shift_benchmark,
    validate_model_pins,
)
from prometheus.governor.keys import generate_private_key, public_key_hex
from prometheus.governor.policy import Policy
from prometheus.governor.service import Governor
from prometheus.harness.prover import run_prover_episode
from prometheus.lab.laws import LAW_NAMES
from prometheus.lab.lean_layer import (
    PROVER_LEAN_TOOLCHAIN,
    prover_lean_project_dir,
    verify_lean_source,
    world_lean_source,
)
from prometheus.lab.world import HiddenLawWorld
from prometheus.observatory import collect_snapshot, render_dashboard_html
from prometheus.runtime.gate import Gate


# ---------------------------------------------------------------------------
# 1. Phase 3 Law-Shift Stress Benchmarks & Recovery Loop
# ---------------------------------------------------------------------------

def test_phase3_law_shift_benchmark_ashby_adaptation_beats_static(tmp_path):
    """Under mid-run shift_laws(), Ashby's forgetting_adapt recovers strongly vs static Gen-0."""
    log = EventLog(tmp_path / "agent.jsonl")
    ckpt = tmp_path / "law_shift_benchmark.json"

    summary = run_law_shift_benchmark(
        seeds=[0, 1, 2, 3],
        shift_seed=777,
        noise=0.05,
        budget=500.0,
        event_log=log,
        checkpoint_path=ckpt,
    )
    assert summary.n_worlds == 4
    assert summary.adapted_label == "forgetting_adapt"
    assert summary.adapted_forgetting_factor < 1.0
    assert summary.adapted_post_accuracy >= 0.85, summary.to_dict()
    assert summary.adapted_post_accuracy > summary.static_post_accuracy + 0.15, summary.to_dict()
    assert summary.adapted_surprise_detection_rate >= 0.75
    assert ckpt.is_file()
    saved = json.loads(ckpt.read_text(encoding="utf-8"))
    assert saved["adapted"]["post_accuracy"] == summary.adapted_post_accuracy
    assert verify_log(tmp_path / "agent.jsonl").ok


def test_redteam_law_shift_continue_episode_respects_markov_blanket():
    """continue_episode() must never access ground_truth, family, _op, _f, or shift_laws."""
    w = HiddenLawWorld(seed=2, noise=0.05, budget=500.0)
    forbidden: list[str] = []

    class _TrapView:
        def __init__(self, world: HiddenLawWorld) -> None:
            self._w = world
        @property
        def symbols(self): return list(self._w.symbols)
        @property
        def noise(self): return self._w.noise
        @property
        def budget(self): return self._w.budget
        @property
        def remaining(self): return self._w.remaining
        @property
        def spent(self): return self._w.spent
        @property
        def combine_cost(self): return self._w.combine_cost
        @property
        def transform_cost(self): return self._w.transform_cost
        def combine(self, a: str, b: str) -> str: return self._w.combine(a, b)
        def transform(self, a: str) -> str: return self._w.transform(a)
        def __getattr__(self, name: str):
            forbidden.append(name)
            raise AttributeError(name)

    trap = _TrapView(w)
    har = LoadedHarness(efe_params=EFEParams(forgetting_factor=0.65))
    agent = CognitiveAgent(harness=har, rng_seed=0)
    agent.run_episode(trap, learn=False)  # type: ignore[arg-type]
    w.shift_laws(shift_seed=99)
    res = agent.continue_episode(trap, sentinel_probes=6, learn=False)  # type: ignore[arg-type]
    assert forbidden == []
    assert res.error is None


# ---------------------------------------------------------------------------
# 2. Phase 3 Lean 4.9 Prover Bake-Off (Goedel-Prover-V2-8B & pass@k)
# ---------------------------------------------------------------------------

def test_phase3_lean_49_compat_source_compiles_and_project_isolated():
    """Separate Lean 4.9 project exists without disturbing v4.34.1; 4.9 syntax compiles cleanly."""
    p49 = prover_lean_project_dir()
    assert p49 is not None
    assert (p49 / "lean-toolchain").read_text(encoding="utf-8").strip() == PROVER_LEAN_TOOLCHAIN
    laws_tc = (p49.parent / "PrometheusLaws" / "lean-toolchain").read_text(encoding="utf-8").strip()
    assert laws_tc == "leanprover/lean4:v4.34.1"

    w = HiddenLawWorld(seed=0)
    src49 = world_lean_source(w, include_proofs=True, lean_compat="4.9")
    assert "(opTable[a.val]!)[b.val]!" in src49
    assert "fTable[a.val]!" in src49
    assert "decideExistsFin" in src49
    # Verify on BOTH Lean 4.9.0 (PrometheusProver49) and Lean 4.34.1 (PrometheusLaws)
    ok_49, err_49 = verify_lean_source(src49, timeout=60, project_dir=p49)
    assert ok_49, err_49
    ok_34, err_34 = verify_lean_source(src49, timeout=60)
    assert ok_34, err_34


class _StubProverClient:
    """Fake ModelClient returning scripted responses per call."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls = 0
        self.temperatures: list[float] = []
        self.temperature = 0.0

    def health_check(self) -> bool:
        return True

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.temperatures.append(self.temperature)
        idx = min(self.calls, len(self._replies) - 1)
        self.calls += 1
        return self._replies[idx]


def test_phase3_prover_bakeoff_runner_pass_at_k_and_resume(tmp_path):
    """ProverBakeoffRunner supports pass@k, Lean 4.9 compat, and atomic checkpoint resume."""
    assert "goedel-prover-v2-8b" in PROVER_BAKEOFF_MODELS
    w0 = HiddenLawWorld(seed=0)
    w1 = HiddenLawWorld(seed=1)
    good0 = "```lean\n" + world_lean_source(w0, include_proofs=True, lean_compat="4.9") + "\n```"
    good1 = "```lean\n" + world_lean_source(w1, include_proofs=True, lean_compat="4.9") + "\n```"
    bad_cheat = world_lean_source(w0, include_proofs=False, lean_compat="4.9")  # contains sorry!

    # Seed 0: attempt 1 cheats (rejected!), attempt 2 succeeds at sample_temperature=0.6
    # Seed 1: attempt 1 succeeds immediately
    client = _StubProverClient([bad_cheat, good0, good1])
    ckpt = tmp_path / "prover_goedel-prover-v2-8b.json"

    # Point project_dir at PrometheusLaws (v4.34.1 installed locally) while emitting lean_compat="4.9"
    cfg = ProverBakeoffConfig(
        model_name="goedel-prover-v2-8b",
        n_worlds=2,
        seeds=[0, 1],
        attempts_per_world=3,
        lean_compat="4.9",
        lean_project_dir=str(prover_lean_project_dir().parent / "PrometheusLaws"),  # type: ignore[union-attr]
        checkpoint_path=str(ckpt),
    )
    runner = ProverBakeoffRunner(cfg, client=client)  # type: ignore[arg-type]
    summary = runner.run()

    assert summary.n_worlds == 2
    assert summary.n_proved == 2
    assert summary.pass_rate == 1.0
    assert summary.results[0].winning_attempt == 2
    assert summary.results[1].winning_attempt == 1
    assert client.temperatures == [0.0, 0.6, 0.0]
    assert ckpt.is_file()

    # Resuming from the completed checkpoint makes zero additional model calls
    client_resume = _StubProverClient([])
    runner2 = ProverBakeoffRunner(cfg, client=client_resume)  # type: ignore[arg-type]
    summary2 = runner2.run()
    assert summary2.n_proved == 2
    assert client_resume.calls == 0


def test_redteam_prover_pass_at_k_never_accepts_cheat_or_tampered_statement():
    """Cheat primitives and statement tampering are rejected across every pass@k attempt."""
    w = HiddenLawWorld(seed=3)
    tpl = world_lean_source(w, include_proofs=False, lean_compat="4.9")
    cheat_native = tpl.replace(":= by sorry", ":= by native_decide")
    tampered_stmt = world_lean_source(w, include_proofs=True, lean_compat="4.9").replace(
        "theorem associative", "theorem weak_associative"
    )

    c_cheat = _StubProverClient([cheat_native, cheat_native])
    r1 = run_prover_episode(w, c_cheat, lean_compat="4.9", attempts=2)  # type: ignore[arg-type]
    assert not r1.proved
    assert r1.error_kind == "cheat_rejected"
    assert r1.attempts == 2

    c_tamp = _StubProverClient([tampered_stmt])
    r2 = run_prover_episode(w, c_tamp, lean_compat="4.9", attempts=1)  # type: ignore[arg-type]
    assert not r2.proved
    assert r2.error_kind == "statement_altered"


# ---------------------------------------------------------------------------
# 3. Phase 3 Multi-Generation Pinned Self-Improvement (run-002) & Observatory
# ---------------------------------------------------------------------------

def test_phase3_multigen_pinned_run_002_governor_e2e(tmp_path):
    """Full run-002 pinned multi-gen E2E: baseline -> Gen 1 -> Gen 2 + Observatory Phase 3 cards."""
    clock = Clock()
    gov_home = tmp_path / "gov_home"
    drive_root = tmp_path / "drive"
    run_id = "run-002"
    exchange = drive_root / run_id
    install_root = tmp_path / "installed"
    gov_home.mkdir(parents=True)
    exchange.mkdir(parents=True)

    audit_seeds = [1000, 1001, 1002, 1003]
    pins = dict(DEFAULT_RUN002_MODEL_PINS)

    Policy(
        run_id=run_id,
        audit_seeds=audit_seeds,
        model_pins=pins,
        max_generations=10,
        require_ready_marker=True,
    ).save(gov_home / "policy.json")
    key = generate_private_key()
    pub_hex = public_key_hex(key)
    (drive_root / "governor_public_key.txt").write_text(pub_hex + "\n", encoding="utf-8")

    gov = Governor(gov_home, exchange, key, clock=clock)
    # Start with a weak baseline (all-False claims) that carries pinned model_pins
    base_claims = {str(s): {k: False for k in LAW_NAMES} for s in audit_seeds}
    baseline_dir = tmp_path / "baseline"
    build_pinned_baseline_bundle(
        baseline_dir,
        run_id=run_id,
        model_pins=pins,
        audit_seeds=audit_seeds,
        evaluate_audit=False,
        initial_claims=base_claims,
    )
    gov.init_run(baseline_dir)
    gov.heartbeat()
    baseline_hash = (exchange / "baseline_hash.txt").read_text(encoding="utf-8").strip()

    gate = Gate(
        governor_public_key_hex=pub_hex,
        run_id=run_id,
        exchange=exchange,
        install_root=install_root,
        baseline_hash=baseline_hash,
        clock=clock,
        state_path=exchange / "gate_state.json",
    )

    log = EventLog(exchange / "logs" / "agent.jsonl", clock=clock)
    mem = AssemblyMemory()
    # Pre-seed memory with 3 worlds for transfer
    seeder = CognitiveAgent(memory=mem, rng_seed=0)
    for ep in range(3):
        seeder.run_episode(HiddenLawWorld(seed=ep, noise=0.05, budget=200.0), learn=True, episode_idx=ep)

    mg_ckpt = exchange / "multigen_checkpoint.json"
    cfg = MultiGenConfig(
        run_id=run_id,
        model_pins=pins,
        target_generations=2,
        audit_seeds=audit_seeds,
        dev_seed_start=300,
        dev_window_size=4,
        dev_budget=30.0,
        audit_budget=250.0,
        stress_modes=("tight_budget", "law_shift"),
        wait_timeout_s=30.0,
        poll_interval_s=0.0,
    )

    def _tick_gov() -> None:
        clock.advance(5.0)
        gov.step()
        gov.heartbeat()

    runner = MultiGenRunner(
        cfg,
        gate=gate,
        exchange=exchange,
        install_root=install_root,
        memory=mem,
        monitor=HomeostaticMonitor(
            AshbyConfig(window_size=4, min_episodes_before_trigger=2, min_breach_episodes=2, cooldown_episodes=2),
            event_log=log,
        ),
        event_log=log,
        checkpoint_path=mg_ckpt,
        on_wait_tick=_tick_gov,
        sleep_fn=lambda _s: None,
        clock=clock,
    )
    summary = runner.run()

    assert summary.end_generation == 2, summary.to_dict()
    assert summary.approvals == 2
    assert summary.vetoes == 0
    assert summary.model_pins == pins
    assert gate.generation == 2
    assert mg_ckpt.is_file()
    assert verify_log(exchange / "logs" / "agent.jsonl").ok
    reloaded = load_harness(install_root / "current" / "harness")
    assert reloaded.loaded_files

    # Also drop a prover checkpoint & law-shift summary to verify Observatory Phase 3 cards
    (drive_root / "prover_goedel-prover-v2-8b.json").write_text(
        json.dumps({
            "model_name": "goedel-prover-v2-8b",
            "lean_toolchain": PROVER_LEAN_TOOLCHAIN,
            "lean_compat": "4.9",
            "attempts_per_world": 4,
            "results": {
                "0": {"seed": 0, "family": "cyclic", "proved": True, "latency_s": 2.1},
                "1": {"seed": 1, "family": "dihedral", "proved": False, "latency_s": 3.5},
            },
        }),
        encoding="utf-8",
    )
    (exchange / "law_shift_benchmark.json").write_text(
        json.dumps({
            "n_worlds": 4,
            "static": {"post_accuracy": 0.625, "ashby_triggers": 2},
            "adapted": {
                "post_accuracy": 0.9688,
                "recovery_gain": 0.3438,
                "forgetting_factor": 0.65,
                "surprise_detection_rate": 1.0,
            },
        }),
        encoding="utf-8",
    )

    snap = collect_snapshot(exchange, governor_home=gov_home, prometheus_dir=drive_root, now=clock())
    assert snap["overall_status"] == "HEALTHY"
    assert snap["multigen"] is not None
    assert snap["multigen"]["end_generation"] == 2
    assert snap["law_shift"] is not None
    assert snap["law_shift"]["adapted_post_accuracy"] == 0.9688
    assert len(snap["prover_bakeoff"]) == 1
    assert snap["prover_bakeoff"][0]["model_name"] == "goedel-prover-v2-8b"

    html_out = render_dashboard_html(snap)
    assert "Phase 3: Law-Shift &amp; Pinned Multi-Gen" in html_out
    assert "Lean 4 Prover Bake-off Leaderboard" in html_out
    assert "goedel-prover-v2-8b" in html_out


def test_redteam_multigen_rejects_unpinned_or_mismatched_model_pins(tmp_path):
    """Pinned runs refuse empty/floating model_pins, and Governor vetoes pin mismatches."""
    with pytest.raises(ValueError, match="non-empty model_pins"):
        validate_model_pins({})
    with pytest.raises(ValueError, match="40-hex"):
        validate_model_pins({"cortex": "qwen3-8b@HEAD"})
    with pytest.raises(ValueError, match="non-empty model_pins"):
        MultiGenConfig(run_id="run-002", model_pins={})

    # Verify Governor strictly vetoes a proposal with wrong model_pins
    clock = Clock()
    gov_home = tmp_path / "gov_home"
    exchange = tmp_path / "exchange"
    gov_home.mkdir()
    exchange.mkdir()
    audit_seeds = [1000, 1001]
    pins = dict(DEFAULT_RUN002_MODEL_PINS)
    Policy(
        run_id="run-002",
        audit_seeds=audit_seeds,
        model_pins=pins,
        require_ready_marker=True,
    ).save(gov_home / "policy.json")
    key = generate_private_key()
    gov = Governor(gov_home, exchange, key, clock=clock)
    perfect = {str(s): HiddenLawWorld(seed=s).ground_truth() for s in audit_seeds}
    base_dir = tmp_path / "baseline"
    build_pinned_baseline_bundle(
        base_dir, run_id="run-002", model_pins=pins, audit_seeds=audit_seeds,
        evaluate_audit=False, initial_claims=perfect,
    )
    gov.init_run(base_dir)

    # Proposal tries to sneak in an unpinned / different commit hash
    bad_dir = exchange / "proposals" / "g1-wrong-pin"
    bundle = write_bundle(
        bad_dir,
        {
            "run_id": "run-002",
            "generation": 1,
            "parent_hash": gov.state.installed_hash,
            "model_pins": {"cortex": "qwen3-8b@" + "f" * 40},
        },
        {"harness/memory_params/transfer_params.json": json.dumps({"enabled": True})},
    )
    (bad_dir / "audit_claims.json").write_text(json.dumps(perfect), encoding="utf-8")
    (bad_dir / ".ready").write_text(bundle.variant_hash + "\n", encoding="utf-8")

    decisions = gov.step()
    assert len(decisions) == 1
    assert decisions[0]["body"]["decision"] == "veto"
    assert any("pinned hashes" in r for r in decisions[0]["body"]["reasons"])
