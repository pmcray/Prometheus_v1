"""End-to-end Phase 0 exit-criteria integration test.

Exercises the full contract between workstation governor, Drive exchange folder,
Colab gate, sandboxed episode loop, forced runtime restart recovery, kill-switch
halt, and Observatory verification.
"""

from __future__ import annotations

import json
import shutil

import pytest

from conftest import Clock, PINS, RUN, claims, make_proposal
from prometheus.common.eventlog import EventLog, verify_log
from prometheus.common.variant import write_bundle
from prometheus.governor.keys import generate_private_key, public_key_hex
from prometheus.governor.policy import Policy
from prometheus.governor.service import GENESIS_PARENT, Governor
from prometheus.lab.laws import LAW_NAMES
from prometheus.lab.world import HiddenLawWorld
from prometheus.observatory import collect_snapshot
from prometheus.runtime.gate import Gate, Halted
from prometheus.runtime.sandbox import run as sandbox_run


def test_phase0_roundtrip_restart_and_kill_switch(tmp_path):
    clock = Clock()
    gov_home = tmp_path / "workstation_governor"
    drive_root = tmp_path / "gdrive_prometheus"
    exchange = drive_root / RUN
    gov_home.mkdir(parents=True)
    exchange.mkdir(parents=True)

    # 1. Governor workstation setup (policy with --require-ready, keygen, init, heartbeat)
    audit_seeds = list(range(1000, 1010))
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
    baseline = make_proposal(
        exchange, "__baseline__", generation=0, parent=GENESIS_PARENT,
        claims_obj=claims(3), proxy=0.5,
    )
    gov.init_run(baseline.root)
    gov.heartbeat()
    baseline_hash = (exchange / "baseline_hash.txt").read_text().strip()
    assert baseline_hash == baseline.variant_hash

    # 2. Colab runtime boot with persistent Drive state_path & checkpoint
    ephemeral_colab = tmp_path / "colab_local_ephemeral"
    gate_state_path = exchange / "gate_state.json"
    checkpoint_path = exchange / "checkpoint.json"
    agent_log_path = exchange / "logs" / "agent.jsonl"

    gate = Gate(
        governor_public_key_hex=pub_hex,
        run_id=RUN,
        exchange=exchange,
        install_root=ephemeral_colab / "harness",
        baseline_hash=baseline_hash,
        clock=clock,
        state_path=gate_state_path,
    )
    agent_log = EventLog(agent_log_path, clock=clock)

    # Run 3 sandboxed episodes and checkpoint after each
    state = {"next_episode": 0, "results": {}}
    agent_code = "return_value = {k: False for k in inputs['law_names']}"
    for idx in range(3):
        gate.check_heartbeat()
        w = HiddenLawWorld(seed=idx)
        res = sandbox_run(agent_code, inputs={"law_names": list(LAW_NAMES), "symbols": w.symbols})
        assert res.error is None
        truth = w.ground_truth()
        state["results"][str(idx)] = {
            "claims": res.return_value,
            "truth": truth,
            "correct": sum(res.return_value[k] == truth[k] for k in LAW_NAMES),
            "family": w.family,
            "spent": w.spent,
        }
        state["next_episode"] = idx + 1
        checkpoint_path.write_text(json.dumps(state, sort_keys=True))
        agent_log.append("episode", {"seed": idx, "correct": state["results"][str(idx)]["correct"]})

    # 3. Propose generation 1 with .ready marker -> Governor approves -> Gate installs
    prop_dir = exchange / "proposals" / "g1"
    bundle = write_bundle(
        prop_dir,
        {
            "run_id": RUN,
            "generation": 1,
            "parent_hash": gate.installed_hash,
            "model_pins": PINS,
            "proxy_score": 0.7,
        },
        {"harness/prompts/theorist.md": "improved prompt v1"},
    )
    (prop_dir / "audit_claims.json").write_text(json.dumps(claims(7)))
    # Before .ready is written, governor ignores the half-uploaded bundle
    assert gov.pending_proposals() == []
    (prop_dir / ".ready").write_text(bundle.variant_hash + "\n")

    decisions = gov.step()
    assert len(decisions) == 1 and decisions[0]["body"]["decision"] == "approve"
    gate.install(prop_dir)
    assert gate.generation == 1
    assert gate.installed_hash == bundle.variant_hash

    # 4. Simulate a forced Colab runtime restart (wipe ephemeral local disk!)
    shutil.rmtree(ephemeral_colab)
    gate_restarted = Gate(
        governor_public_key_hex=pub_hex,
        run_id=RUN,
        exchange=exchange,
        install_root=ephemeral_colab / "harness",
        baseline_hash=baseline_hash,
        clock=clock,
        state_path=gate_state_path,
    )
    assert gate_restarted.generation == 1
    assert gate_restarted.installed_hash == bundle.variant_hash
    resumed_state = json.loads(checkpoint_path.read_text())
    assert resumed_state["next_episode"] == 3
    resumed_log = EventLog(agent_log_path, clock=clock)
    gate_restarted.check_heartbeat()
    resumed_log.append("episode", {"seed": 3, "resumed": True})
    assert verify_log(agent_log_path).ok and verify_log(agent_log_path).records == 4

    # 5. Stopping governor (heartbeat expires) halts live run
    clock.advance(20 * 60)
    with pytest.raises(Halted, match="expired"):
        gate_restarted.check_heartbeat()

    # Governor resumes heartbeat -> run healthy again; then pull kill switch -> halted
    gov.heartbeat()
    gate_restarted.check_heartbeat()
    snap_healthy = collect_snapshot(exchange, governor_home=gov_home, prometheus_dir=drive_root, now=clock())
    assert snap_healthy["overall_status"] == "HEALTHY"
    assert snap_healthy["state"]["generation"] == 1
    assert all(log["ok"] for log in snap_healthy["logs"])

    gov.kill("phase0 e2e drill")
    with pytest.raises(Halted, match="kill switch"):
        gate_restarted.check_heartbeat()
    snap_halted = collect_snapshot(exchange, governor_home=gov_home, prometheus_dir=drive_root, now=clock())
    assert snap_halted["overall_status"] == "HALTED"
    assert all(log["ok"] for log in snap_halted["logs"])
