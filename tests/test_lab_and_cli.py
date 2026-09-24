import json
import os

import pytest

from conftest import claims, make_proposal
from prometheus.governor import cli
from prometheus.lab import LAWS, HiddenLawWorld
from prometheus.lab.audit import AuditError, score_claims
from prometheus.lab.world import BudgetExhausted
from prometheus.runtime.gate import InstallRefused


# ----- Hidden-Law Lab -------------------------------------------------------------

def test_worlds_are_deterministic():
    a, b = HiddenLawWorld(seed=42), HiddenLawWorld(seed=42)
    assert a.symbols == b.symbols and a.ground_truth() == b.ground_truth()
    s = a.symbols
    assert [a.combine(s[0], s[1]) for _ in range(20)] == [b.combine(s[0], s[1]) for _ in range(20)]


def test_noise_free_observations_match_truth():
    w = HiddenLawWorld(seed=3, noise=0.0, budget=10_000)
    op, f = w.true_tables()
    for (x, y), z in op.items():
        assert w.combine(x, y) == z
    for x, y in f.items():
        assert w.transform(x) == y


def test_known_families_have_expected_laws():
    seen = {}
    for seed in range(400):
        w = HiddenLawWorld(seed=seed)
        seen.setdefault(w.family, w.ground_truth())
    assert seen["cyclic"]["associative"] and seen["cyclic"]["has_inverses"]
    assert not seen["symmetric3"]["commutative"] and seen["symmetric3"]["has_inverses"]
    assert seen["max_semilattice"]["idempotent"] and seen["max_semilattice"]["has_identity"]
    rps = seen["rock_paper_scissors"]
    assert rps["commutative"] and rps["idempotent"] and not rps["associative"]
    assert not seen["left_zero"]["commutative"] and seen["left_zero"]["associative"]


def test_every_law_varies_across_worlds():
    truths = [HiddenLawWorld(seed=s).ground_truth() for s in range(300)]
    for law in LAWS:
        values = {t[law.name] for t in truths}
        assert values == {True, False}, law.name


def test_budget_enforced():
    w = HiddenLawWorld(seed=1, budget=2.0)
    s = w.symbols
    w.combine(s[0], s[0]); w.combine(s[0], s[1])
    with pytest.raises(BudgetExhausted):
        w.combine(s[1], s[1])


def test_law_shift_changes_structure_keeps_symbols():
    w = HiddenLawWorld(seed=11)
    before_syms, before_family = list(w.symbols), w.family
    w.shift_laws(1)
    assert w.symbols == before_syms and w.family != before_family


def test_audit_scoring():
    seeds = list(range(1000, 1010))
    assert score_claims(claims(10), seeds) == 1.0
    assert score_claims({}, seeds) == 0.0
    with pytest.raises(AuditError):
        score_claims([], seeds)


# ----- gate hardening --------------------------------------------------------------

def test_redteam_symlink_in_bundle(setup):
    s = setup
    bundle = make_proposal(s["exchange"], "sym", generation=1, parent=s["gate"].installed_hash,
                           claims_obj=claims(6))
    os.symlink("/etc/passwd", bundle.root / "files" / "harness" / "prompts" / "leak.md")
    s["gov"].step()
    body = json.loads((s["exchange"] / "approvals" / "sym.json").read_text())["body"]
    assert body["decision"] == "veto"


def test_redteam_file_added_after_approval_is_refused(setup):
    s = setup
    bundle = make_proposal(s["exchange"], "late", generation=1, parent=s["gate"].installed_hash,
                           claims_obj=claims(6))
    s["gov"].step()
    (bundle.root / "files" / "harness" / "tools").mkdir(parents=True)
    (bundle.root / "files" / "harness" / "tools" / "x.py").write_text("import socket")
    with pytest.raises(InstallRefused):
        s["gate"].install(bundle.root)
    assert s["gate"].generation == 0


# ----- CLI smoke test ------------------------------------------------------------

def test_cli_end_to_end(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PROMETHEUS_GOVERNOR_PASSPHRASE", "correct horse battery staple")
    home, exchange = tmp_path / "home", tmp_path / "drive" / "run-cli"
    assert cli.main(["keygen", "--home", str(home)]) == 0
    assert oct((home / "governor_key.pem").stat().st_mode & 0o777) == "0o600"
    assert cli.main(["policy", "--home", str(home), "--run-id", "run-cli",
                     "--audit-seeds", "1000-1009", "--model-pin", "cortex=qwen3.8-27b@abc123",
                     "--model-pin", "prover=goedel-v2-8b@def456"]) == 0
    base = make_proposal(exchange, "__baseline__", generation=0, parent="genesis",
                         run_id="run-cli", claims_obj=claims(3))
    assert cli.main(["init", "--home", str(home), "--exchange", str(exchange),
                     "--baseline", str(base.root)]) == 0
    make_proposal(exchange, "g1", generation=1, parent=base.variant_hash, run_id="run-cli",
                  claims_obj=claims(5))
    assert cli.main(["run", "--home", str(home), "--exchange", str(exchange), "--once"]) == 0
    assert "approve" in capsys.readouterr().out
    assert cli.main(["kill", "--home", str(home), "--exchange", str(exchange),
                     "--reason", "test"]) == 0
    assert json.loads((exchange / "heartbeat.json").read_text())["body"]["halt"] is True
    assert cli.main(["verify-log", str(home / "decisions.jsonl")]) == 0
