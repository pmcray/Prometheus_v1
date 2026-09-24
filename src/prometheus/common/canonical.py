"""Canonical JSON encoding and hashing.

Everything that is signed or hash-chained goes through these two functions, so the
Colab runtime and the governor workstation always compute identical bytes.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_bytes(obj: Any) -> bytes:
    """Deterministic JSON: sorted keys, no whitespace, UTF-8, no NaN/Infinity."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_bytes(obj))
