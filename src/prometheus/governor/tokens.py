"""Signed governor tokens: approvals, vetoes and heartbeats.

Envelope: {"body": {...}, "key_id": "...", "sig": "<hex ed25519 over canonical(body)>"}

Approval/veto body binds: type, run_id, generation, variant_hash, decision, reasons,
issued_at, expires_at, nonce. It cannot be replayed onto another variant, run or
generation, and it expires.

Heartbeat body: type, run_id, counter (strictly increasing), issued_at, expires_at,
halt (True when the kill switch has been pulled).
"""

from __future__ import annotations

import secrets
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from ..common.canonical import canonical_bytes
from .keys import key_id, public_key_hex


class TokenError(ValueError):
    pass


def sign(body: dict[str, Any], key: Ed25519PrivateKey) -> dict[str, Any]:
    return {
        "body": body,
        "key_id": key_id(public_key_hex(key)),
        "sig": key.sign(canonical_bytes(body)).hex(),
    }


def verify(envelope: Any, pub: Ed25519PublicKey) -> dict[str, Any]:
    """Return the body if the signature is valid; raise TokenError otherwise."""
    if not isinstance(envelope, dict) or set(envelope) != {"body", "key_id", "sig"}:
        raise TokenError("malformed token envelope")
    body, sig = envelope["body"], envelope["sig"]
    if not isinstance(body, dict) or not isinstance(sig, str):
        raise TokenError("malformed token envelope")
    if envelope["key_id"] != key_id(public_key_hex(pub)):
        raise TokenError("token signed by an unknown key")
    try:
        pub.verify(bytes.fromhex(sig), canonical_bytes(body))
    except (InvalidSignature, ValueError) as exc:
        raise TokenError("bad signature") from exc
    return body


def approval_body(
    *,
    run_id: str,
    generation: int,
    variant_hash: str,
    approve: bool,
    reasons: list[str],
    now: float,
    ttl_s: float,
) -> dict[str, Any]:
    return {
        "type": "approval" if approve else "veto",
        "run_id": run_id,
        "generation": generation,
        "variant_hash": variant_hash,
        "decision": "approve" if approve else "veto",
        "reasons": reasons,
        "issued_at": round(now, 3),
        "expires_at": round(now + ttl_s, 3),
        "nonce": secrets.token_hex(8),
    }


def heartbeat_body(
    *, run_id: str, counter: int, now: float, ttl_s: float, halt: bool
) -> dict[str, Any]:
    return {
        "type": "heartbeat",
        "run_id": run_id,
        "counter": counter,
        "issued_at": round(now, 3),
        "expires_at": round(now + ttl_s, 3),
        "halt": halt,
    }
