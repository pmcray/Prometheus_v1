"""The Internal Alignment Governor service.

Runs on the Chief Scientist's Linux workstation. It shares one folder with the
Colab runtime (synced through Google Drive with rclone):

    exchange/proposals/<name>/        variant bundle + audit_claims.json   (runtime writes)
    exchange/approvals/<name>.json    signed approval or veto              (governor writes)
    exchange/heartbeat.json           signed heartbeat, halt flag          (governor writes)

and keeps its own private home, which never leaves the workstation:

    home/policy.json  home/state.json  home/governor_key.pem  home/decisions.jsonl
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ..common.eventlog import EventLog
from ..common.variant import BundleError, load_bundle
from ..lab.audit import AuditError, score_claims
from .policy import Policy
from .rules import check_variant
from .tokens import approval_body, heartbeat_body, sign

GENESIS_PARENT = "genesis"


@dataclass
class GovernorState:
    installed_hash: str
    generation: int
    audit_score: float
    proxy_score: float | None
    heartbeat_counter: int = 0
    killed: bool = False
    approvals: int = 0
    vetoes: int = 0


def atomic_write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, sort_keys=True, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


class Governor:
    def __init__(self, home: Path, exchange: Path, key: Ed25519PrivateKey,
                 clock: Callable[[], float] = time.time):
        self.home, self.exchange, self.key, self.clock = Path(home), Path(exchange), key, clock
        self.policy = Policy.load(self.home / "policy.json")
        self.log = EventLog(self.home / "decisions.jsonl", clock=clock)
        self.state = self._load_state()

    # ----- state -------------------------------------------------------------

    @property
    def _state_path(self) -> Path:
        return self.home / "state.json"

    def _load_state(self) -> GovernorState | None:
        if not self._state_path.exists():
            return None
        return GovernorState(**json.loads(self._state_path.read_text()))

    def _save_state(self) -> None:
        atomic_write_json(self._state_path, asdict(self.state))

    def _require_state(self) -> GovernorState:
        if self.state is None:
            raise RuntimeError("run not initialised: call init_run with the baseline bundle")
        return self.state

    # ----- run lifecycle -----------------------------------------------------

    def init_run(self, baseline_dir: Path) -> GovernorState:
        """Register the human-authored generation-0 harness and its audit claims."""
        if self.state is not None:
            raise RuntimeError("run already initialised")
        bundle = load_bundle(baseline_dir)
        reasons = check_variant(bundle, self.policy, expected_generation=0,
                                expected_parent=GENESIS_PARENT)
        claims = _read_claims(Path(baseline_dir))
        score = score_claims(claims, self.policy.audit_seeds)
        if reasons:
            raise ValueError(f"baseline fails the rules: {reasons}")
        self.state = GovernorState(
            installed_hash=bundle.variant_hash, generation=0, audit_score=score,
            proxy_score=bundle.manifest.get("proxy_score"),
        )
        self._save_state()
        self.log.append("init_run", {"run_id": self.policy.run_id,
                                     "variant_hash": bundle.variant_hash, "audit_score": score})
        return self.state

    def kill(self, reason: str) -> None:
        st = self._require_state()
        st.killed = True
        self._save_state()
        self.log.append("kill", {"reason": reason})
        self.heartbeat()

    # ----- heartbeat ---------------------------------------------------------

    def heartbeat(self) -> dict[str, Any]:
        st = self._require_state()
        st.heartbeat_counter += 1
        self._save_state()
        token = sign(heartbeat_body(run_id=self.policy.run_id, counter=st.heartbeat_counter,
                                    now=self.clock(), ttl_s=self.policy.heartbeat_ttl_s,
                                    halt=st.killed), self.key)
        atomic_write_json(self.exchange / "heartbeat.json", token)
        return token

    # ----- proposals ---------------------------------------------------------

    def pending_proposals(self) -> list[Path]:
        pdir = self.exchange / "proposals"
        if not pdir.exists():
            return []
        return [p for p in sorted(pdir.iterdir())
                if p.is_dir() and not p.name.startswith(".")
                and not (self.exchange / "approvals" / f"{p.name}.json").exists()]

    def review(self, proposal_dir: Path) -> dict[str, Any]:
        """Decide one proposal, write the signed decision, and log it."""
        st = self._require_state()
        proposal_dir = Path(proposal_dir)
        reasons: list[str] = []
        variant_hash, generation = "invalid", st.generation + 1
        audit = proxy = None

        if st.killed:
            reasons.append("run has been halted by the kill switch")
        try:
            bundle = load_bundle(proposal_dir)
            variant_hash = bundle.variant_hash
            generation = bundle.manifest.get("generation", generation)
            reasons += check_variant(bundle, self.policy, expected_generation=st.generation + 1,
                                     expected_parent=st.installed_hash)
            proxy = bundle.manifest.get("proxy_score")
        except BundleError as exc:
            reasons.append(f"bundle rejected: {exc}")

        try:
            audit = score_claims(_read_claims(proposal_dir), self.policy.audit_seeds)
        except (AuditError, OSError, ValueError) as exc:
            reasons.append(f"audit claims unusable: {exc}")

        if audit is not None:
            if audit < st.audit_score:
                reasons.append(f"audit score fell from {st.audit_score:.3f} to {audit:.3f}")
            proxy_gain = (proxy is not None and st.proxy_score is not None
                          and proxy > st.proxy_score)
            if proxy_gain and audit <= st.audit_score:
                reasons.append("proxy-only gain: evaluator score rose but audit score did not")

        approve = not reasons
        body = approval_body(run_id=self.policy.run_id,
                             generation=generation if isinstance(generation, int) else -1,
                             variant_hash=variant_hash, approve=approve,
                             reasons=reasons or ["all rules passed"], now=self.clock(),
                             ttl_s=self.policy.approval_ttl_s)
        token = sign(body, self.key)
        atomic_write_json(self.exchange / "approvals" / f"{proposal_dir.name}.json", token)

        if approve:
            st.installed_hash, st.generation = variant_hash, generation
            st.audit_score = audit
            if proxy is not None:
                st.proxy_score = proxy
            st.approvals += 1
        else:
            st.vetoes += 1
        self._save_state()
        self.log.append("decision", {"proposal": proposal_dir.name, "decision": body["decision"],
                                     "generation": body["generation"],
                                     "variant_hash": variant_hash, "audit_score": audit,
                                     "proxy_score": proxy, "reasons": body["reasons"]})
        return token

    def step(self) -> list[dict[str, Any]]:
        """One service tick: decide all pending proposals, then send a heartbeat."""
        decisions = [self.review(p) for p in self.pending_proposals()]
        self.heartbeat()
        return decisions


def _read_claims(proposal_dir: Path) -> Any:
    path = Path(proposal_dir) / "audit_claims.json"
    if not path.is_file():
        raise AuditError("audit_claims.json missing")
    return json.loads(path.read_text(encoding="utf-8"))
