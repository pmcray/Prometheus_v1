"""Colab-side gate: the runtime half of the governor contract.

Protected code (variants may not touch anything outside harness/). It holds only
the governor's public key. Two duties:

1. `check_heartbeat` must pass before every step of the run loop. A missing,
   forged, stale, replayed or halting heartbeat raises `Halted`.
2. `install` copies an approved variant into place. It re-derives the variant hash
   from the bundle on disk and requires a matching, unexpired, signed approval for
   exactly that hash, run and generation; otherwise it raises `InstallRefused`.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from ..common.variant import BundleError, load_bundle
from ..governor.keys import load_public_key_hex
from ..governor.tokens import TokenError, verify

GENESIS_PARENT = "genesis"


class Halted(RuntimeError):
    """The run must stop now (checkpoint and exit)."""


class InstallRefused(RuntimeError):
    pass


class Gate:
    def __init__(self, *, governor_public_key_hex: str, run_id: str, exchange: Path,
                 install_root: Path, baseline_hash: str,
                 clock: Callable[[], float] = time.time, max_clock_skew_s: float = 60.0):
        self.pub = load_public_key_hex(governor_public_key_hex)
        self.run_id = run_id
        self.exchange, self.install_root = Path(exchange), Path(install_root)
        self.clock, self.skew = clock, max_clock_skew_s
        self._state_path = self.install_root / ".gate_state.json"
        self.install_root.mkdir(parents=True, exist_ok=True)
        if self._state_path.exists():
            self._state = json.loads(self._state_path.read_text())
        else:
            self._state = {"installed_hash": baseline_hash, "generation": 0,
                           "last_heartbeat_counter": 0}
            self._save()

    def _save(self) -> None:
        self._state_path.write_text(json.dumps(self._state, sort_keys=True))

    @property
    def installed_hash(self) -> str:
        return self._state["installed_hash"]

    @property
    def generation(self) -> int:
        return self._state["generation"]

    def _read_token(self, path: Path) -> dict[str, Any]:
        try:
            envelope = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TokenError(f"cannot read {path.name}: {exc}") from exc
        return verify(envelope, self.pub)

    def _fresh(self, body: dict[str, Any], now: float) -> bool:
        return (body["issued_at"] <= now + self.skew) and (now <= body["expires_at"] + self.skew)

    # ----- heartbeat -----------------------------------------------------------

    def check_heartbeat(self) -> dict[str, Any]:
        now = self.clock()
        try:
            body = self._read_token(self.exchange / "heartbeat.json")
        except TokenError as exc:
            raise Halted(f"no valid heartbeat: {exc}") from exc
        if body.get("type") != "heartbeat" or body.get("run_id") != self.run_id:
            raise Halted("heartbeat is for a different run")
        if body.get("halt"):
            raise Halted("governor pulled the kill switch")
        if not self._fresh(body, now):
            raise Halted("heartbeat expired")
        counter = body.get("counter")
        if not isinstance(counter, int) or counter < self._state["last_heartbeat_counter"]:
            raise Halted("heartbeat replayed (counter went backwards)")
        if counter > self._state["last_heartbeat_counter"]:
            self._state["last_heartbeat_counter"] = counter
            self._save()
        return body

    # ----- install -------------------------------------------------------------

    def install(self, proposal_dir: Path) -> Path:
        self.check_heartbeat()
        proposal_dir = Path(proposal_dir)
        now = self.clock()
        try:
            body = self._read_token(self.exchange / "approvals" / f"{proposal_dir.name}.json")
        except TokenError as exc:
            raise InstallRefused(f"no valid approval: {exc}") from exc
        if body.get("type") != "approval" or body.get("decision") != "approve":
            raise InstallRefused(f"governor vetoed: {'; '.join(body.get('reasons', []))}")
        if body.get("run_id") != self.run_id:
            raise InstallRefused("approval is for a different run")
        if not self._fresh(body, now):
            raise InstallRefused("approval expired")
        # Copy first, then verify the private copy: files changed on Drive after the
        # check could otherwise slip in between verification and installation.
        staging = self.install_root / ".staging"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
        try:
            shutil.copy2(proposal_dir / "manifest.json", staging / "manifest.json")
            if (proposal_dir / "files").exists():
                shutil.copytree(proposal_dir / "files", staging / "files", symlinks=True)
            bundle = load_bundle(staging)
        except (BundleError, OSError) as exc:
            shutil.rmtree(staging, ignore_errors=True)
            raise InstallRefused(f"bundle changed or invalid: {exc}") from exc
        refusal = None
        m = bundle.manifest
        if bundle.variant_hash != body.get("variant_hash"):
            refusal = "bundle on disk is not the variant that was approved"
        elif m.get("generation") != body.get("generation") or m.get("generation") != self.generation + 1:
            refusal = "approval is not for the next generation"
        elif m.get("parent_hash") != self.installed_hash:
            refusal = "variant does not descend from the installed variant"
        if refusal:
            shutil.rmtree(staging, ignore_errors=True)
            raise InstallRefused(refusal)

        dest = self.install_root / f"gen-{m['generation']:04d}"
        tmp = self.install_root / f".tmp-gen-{m['generation']:04d}"
        if tmp.exists():
            shutil.rmtree(tmp)
        (staging / "files").rename(tmp) if (staging / "files").exists() else tmp.mkdir()
        (tmp / "manifest.json").write_text(json.dumps(m, sort_keys=True, indent=2))
        shutil.rmtree(staging, ignore_errors=True)
        if dest.exists():
            shutil.rmtree(dest)
        tmp.rename(dest)
        current = self.install_root / "current"
        link_tmp = self.install_root / ".current.tmp"
        if link_tmp.is_symlink() or link_tmp.exists():
            link_tmp.unlink()
        link_tmp.symlink_to(dest.name)
        link_tmp.replace(current)
        self._state.update(installed_hash=bundle.variant_hash, generation=m["generation"])
        self._save()
        return dest
