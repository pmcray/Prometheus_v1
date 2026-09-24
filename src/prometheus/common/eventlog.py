"""Hash-chained, append-only event log (JSON Lines).

Each record carries the hash of the previous record, so any edit, deletion or
reordering of past entries is detectable by `verify_log`. One writer per file:
the Colab runtime writes the agent log, the governor writes its own decision log.

Record fields: seq, ts, kind, payload, prev_hash, hash.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .canonical import hash_obj

GENESIS_HASH = "0" * 64


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    records: int
    first_bad_seq: int | None = None
    reason: str = ""


def _record_hash(record: dict[str, Any]) -> str:
    body = {k: v for k, v in record.items() if k != "hash"}
    return hash_obj(body)


class EventLog:
    """Append-only log. Reopening an existing file resumes the chain."""

    def __init__(self, path: str | os.PathLike[str], clock=time.time):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._seq, self._last_hash = self._tail()

    def _tail(self) -> tuple[int, str]:
        if not self.path.exists():
            return 0, GENESIS_HASH
        result = verify_log(self.path)
        if not result.ok:
            raise ValueError(
                f"refusing to append to a corrupted log {self.path}: "
                f"record {result.first_bad_seq}: {result.reason}"
            )
        last = None
        for last in read_log(self.path):
            pass
        if last is None:
            return 0, GENESIS_HASH
        return last["seq"] + 1, last["hash"]

    def append(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        record = {
            "seq": self._seq,
            "ts": round(float(self._clock()), 6),
            "kind": kind,
            "payload": payload,
            "prev_hash": self._last_hash,
        }
        record["hash"] = _record_hash(record)
        line = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self._seq += 1
        self._last_hash = record["hash"]
        return record

    @property
    def head_hash(self) -> str:
        return self._last_hash

    def __len__(self) -> int:
        return self._seq


def read_log(path: str | os.PathLike[str]) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def verify_log(path: str | os.PathLike[str]) -> VerifyResult:
    """Check sequence numbers, prev-hash links and each record's own hash."""
    expected_prev = GENESIS_HASH
    count = 0
    try:
        for i, rec in enumerate(read_log(path)):
            if rec.get("seq") != i:
                return VerifyResult(False, count, i, "sequence gap or reorder")
            if rec.get("prev_hash") != expected_prev:
                return VerifyResult(False, count, i, "broken chain link")
            if rec.get("hash") != _record_hash(rec):
                return VerifyResult(False, count, i, "record content altered")
            expected_prev = rec["hash"]
            count += 1
    except json.JSONDecodeError as exc:
        return VerifyResult(False, count, count, f"unparseable line: {exc}")
    return VerifyResult(True, count)
