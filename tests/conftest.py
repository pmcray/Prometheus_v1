import json
from pathlib import Path

import pytest

from prometheus.common.variant import write_bundle
from prometheus.governor.keys import generate_private_key, public_key_hex
from prometheus.governor.policy import Policy
from prometheus.governor.service import GENESIS_PARENT, Governor
from prometheus.lab.world import HiddenLawWorld
from prometheus.runtime.gate import Gate

RUN = "run-test"
PINS = {"cortex": "qwen3.8-27b@abc123", "prover": "goedel-v2-8b@def456"}
AUDIT_SEEDS = list(range(1000, 1010))


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


def claims(accuracy_worlds: int):
    """Correct claims for the first `accuracy_worlds` audit worlds, all-False elsewhere."""
    out = {}
    for i, seed in enumerate(AUDIT_SEEDS):
        truth = HiddenLawWorld(seed=seed).ground_truth()
        out[str(seed)] = truth if i < accuracy_worlds else {k: False for k in truth}
    return out


def make_proposal(exchange: Path, name: str, *, generation: int, parent: str,
                  files=None, claims_obj=None, proxy=None, pins=PINS, run_id=RUN):
    root = exchange / "proposals" / name if name != "__baseline__" else exchange / "baseline"
    fields = {"run_id": run_id, "generation": generation, "parent_hash": parent,
              "model_pins": pins, "description": f"variant {name}"}
    if proxy is not None:
        fields["proxy_score"] = proxy
    bundle = write_bundle(root, fields, files or {"harness/prompts/theorist.md": f"prompt {name}"})
    (root / "audit_claims.json").write_text(json.dumps(claims_obj if claims_obj is not None else claims(3)))
    return bundle


@pytest.fixture
def setup(tmp_path):
    clock = Clock()
    home, exchange = tmp_path / "gov_home", tmp_path / "drive" / RUN
    home.mkdir()
    Policy(run_id=RUN, audit_seeds=AUDIT_SEEDS, model_pins=PINS).save(home / "policy.json")
    key = generate_private_key()
    gov = Governor(home, exchange, key, clock=clock)
    baseline = make_proposal(exchange, "__baseline__", generation=0, parent=GENESIS_PARENT,
                             claims_obj=claims(3), proxy=0.5)
    gov.init_run(baseline.root)
    gov.heartbeat()
    gate = Gate(governor_public_key_hex=public_key_hex(key), run_id=RUN, exchange=exchange,
                install_root=tmp_path / "colab" / "harness_installs",
                baseline_hash=baseline.variant_hash, clock=clock)
    return {"gov": gov, "gate": gate, "exchange": exchange, "clock": clock, "key": key,
            "baseline": baseline, "tmp": tmp_path}
