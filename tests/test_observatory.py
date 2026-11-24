"""Tests for the Prometheus v1 Observatory dashboard (collector, HTML, HTTP, CLI)."""

from __future__ import annotations

import json
import sqlite3
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from conftest import claims, make_proposal
from prometheus.common.eventlog import EventLog
from prometheus.governor.keys import public_key_hex
from prometheus.lab.laws import LAW_NAMES
from prometheus.lab.world import HiddenLawWorld
from prometheus.observatory import collect_snapshot, render_dashboard_html
from prometheus.observatory.cli import main as observatory_main
from prometheus.observatory.server import make_handler


def test_collector_handles_empty_exchange(tmp_path):
    ex = tmp_path / "run-empty"
    ex.mkdir()
    snap = collect_snapshot(ex, now=1_800_000_000.0)
    assert snap["overall_status"] == "NO_HEARTBEAT"
    assert snap["heartbeat"]["present"] is False
    assert snap["approvals"] == []
    assert snap["logs"] == []


def test_collector_full_run_and_leaderboard(setup):
    s = setup
    gov = s["gov"]
    ex = s["exchange"]
    prom_dir = ex.parent
    (gov.home / "governor_public_key.txt").write_text(public_key_hex(s["key"]) + "\n")

    # Approve one generation and veto one
    b1 = make_proposal(ex, "g1", generation=1, parent=s["gate"].installed_hash,
                       claims_obj=claims(6), proxy=0.6)
    gov.step()
    s["gate"].install(b1.root)
    make_proposal(ex, "bad", generation=2, parent=s["gate"].installed_hash,
                  files={"governor/p.py": "x=1"}, claims_obj=claims(7))
    gov.step()

    # Pending proposal (not reviewed yet)
    make_proposal(ex, "pending-g2", generation=2, parent=s["gate"].installed_hash,
                  claims_obj=claims(8))

    # Runtime agent event log on exchange (including Phase 1 cognitive telemetry)
    agent_log = EventLog(ex / "logs" / "agent.jsonl", clock=s["clock"])
    agent_log.append("episode", {"seed": 0, "correct": 7})
    agent_log.append("cognitive_episode", {
        "seed": 0, "budget_spent": 42.5, "final_entropy_bits": 0.08,
        "mdl_compression_bits": 2.4, "stopped_early": True,
    })
    (ex / "memory_state.json").write_text(json.dumps({
        "episodes_consolidated": 5,
        "assemblies": [{"assembly_id": "asm-001", "salience": 0.92}],
    }))

    # Runtime checkpoint + two bake-off checkpoints
    w0 = HiddenLawWorld(seed=0)
    truth0 = w0.ground_truth()
    (ex / "checkpoint.json").write_text(json.dumps({
        "next_episode": 1,
        "results": {
            "0": {"claims": truth0, "truth": truth0, "correct": 8,
                  "family": w0.family, "n": len(w0.symbols), "spent": 40.0},
        },
    }))
    (prom_dir / "bakeoff_qwen3-8b.json").write_text(json.dumps({
        "model_name": "qwen3-8b",
        "next_episode": 1,
        "results": {"0": {"claims": truth0, "truth": truth0, "family": w0.family, "latency_s": 1.2}},
    }))
    (prom_dir / "bakeoff_gemma4-4b.json").write_text(json.dumps({
        "model_name": "gemma4-4b",
        "next_episode": 1,
        "results": {"0": {"claims": {k: False for k in LAW_NAMES}, "truth": truth0,
                          "family": w0.family, "latency_s": 0.5, "error": "boom"}},
    }))

    # Optional MLflow SQLite DB
    db = prom_dir / "mlflow.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE runs (run_uuid TEXT, name TEXT, status TEXT, start_time INT, end_time INT, lifecycle_stage TEXT)")
    conn.execute("CREATE TABLE latest_metrics (run_uuid TEXT, key TEXT, value REAL)")
    conn.execute("INSERT INTO runs VALUES ('r1', 'qwen3-8b', 'FINISHED', 100, 200, 'active')")
    conn.execute("INSERT INTO latest_metrics VALUES ('r1', 'accuracy', 0.91)")
    conn.commit()
    conn.close()

    snap = collect_snapshot(ex, governor_home=gov.home, prometheus_dir=prom_dir, now=s["clock"]())
    assert snap["overall_status"] == "HEALTHY"
    assert snap["heartbeat"]["sig_valid"] is True
    assert snap["heartbeat"]["fresh"] is True
    assert snap["state"]["generation"] == 1
    assert snap["state"]["approvals"] == 1
    assert snap["state"]["vetoes"] == 1
    assert len(snap["pending_proposals"]) == 1
    assert all(entry["ok"] for entry in snap["logs"])
    assert len(snap["logs"]) == 2
    assert snap["checkpoint"]["accuracy"] == 1.0
    assert [b["model_name"] for b in snap["bakeoff"]] == ["qwen3-8b", "gemma4-4b"]
    assert snap["bakeoff"][0]["accuracy"] == 1.0
    assert snap["bakeoff"][1]["accuracy"] == 0.0  # errored episode scores 0
    assert snap["mlflow_runs"][0]["metrics"]["accuracy"] == 0.91

    html_doc = render_dashboard_html(snap)
    assert "Prometheus v1 Observatory" in html_doc
    assert "HEALTHY" in html_doc
    assert "qwen3-8b" in html_doc
    assert "Cognitive Core &amp; MDL Memory" in html_doc
    assert "42.5" in html_doc
    assert "1 / 5" in html_doc


def test_collector_detects_corrupted_log_and_halted_state(setup):
    s = setup
    gov = s["gov"]
    ex = s["exchange"]
    (gov.home / "governor_public_key.txt").write_text(public_key_hex(s["key"]) + "\n")

    # Tamper with an agent log on exchange
    alog = EventLog(ex / "logs" / "agent.jsonl", clock=s["clock"])
    alog.append("ep", {"v": 1})
    alog.append("ep", {"v": 2})
    lines = alog.path.read_text().splitlines()
    rec = json.loads(lines[0])
    rec["payload"]["v"] = 999
    lines[0] = json.dumps(rec)
    alog.path.write_text("\n".join(lines) + "\n")

    snap = collect_snapshot(ex, governor_home=gov.home, now=s["clock"]())
    assert snap["overall_status"] == "LOG_CORRUPTED"
    assert any(not entry["ok"] for entry in snap["logs"])

    # Pulling kill switch surfaces HALTED
    gov.kill("observatory test")
    snap2 = collect_snapshot(ex, governor_home=gov.home, now=s["clock"]())
    assert snap2["overall_status"] == "HALTED"
    assert snap2["heartbeat"]["halt"] is True


def test_render_html_escapes_untrusted_text(tmp_path):
    snap = collect_snapshot(tmp_path, now=1_800_000_000.0)
    snap["decisions"] = [{
        "proposal": "<script>alert(1)</script>",
        "generation": 1,
        "decision": "veto",
        "audit_score": 0.5,
        "proxy_score": None,
        "reasons": ["<b>bad</b>"],
    }]
    out = render_dashboard_html(snap)
    assert "<script>" not in out
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in out


def test_http_server_and_cli_snapshot(setup, capsys):
    s = setup
    gov = s["gov"]
    ex = s["exchange"]
    (gov.home / "governor_public_key.txt").write_text(public_key_hex(s["key"]) + "\n")

    handler = make_handler(ex, governor_home=gov.home, prometheus_dir=ex.parent)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=5) as resp:
            assert json.loads(resp.read().decode()) == {"ok": True}
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/snapshot", timeout=5) as resp:
            data = json.loads(resp.read().decode())
            assert data["heartbeat"]["sig_valid"] is True
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as resp:
            body = resp.read().decode()
            assert "Prometheus v1 Observatory" in body
    finally:
        httpd.shutdown()
        httpd.server_close()

    rc = observatory_main([
        "snapshot", "--exchange", str(ex), "--governor-home", str(gov.home), "--json"
    ])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["state"]["generation"] == 0
