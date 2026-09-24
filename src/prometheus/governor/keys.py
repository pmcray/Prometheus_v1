"""Ed25519 keys for the governor.

The private key stays on the Chief Scientist's workstation (ideally on a hardware
token; this module handles the file-based fallback: PKCS#8 PEM, encrypted with a
passphrase when one is given, mode 0600). The Colab runtime only ever receives the
public key.
"""

from __future__ import annotations

import os
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from ..common.canonical import sha256_hex


def generate_private_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


def save_private_key(key: Ed25519PrivateKey, path: Path, passphrase: bytes | None) -> None:
    enc = (
        serialization.BestAvailableEncryption(passphrase)
        if passphrase
        else serialization.NoEncryption()
    )
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, enc)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(pem)


def load_private_key(path: Path, passphrase: bytes | None) -> Ed25519PrivateKey:
    key = serialization.load_pem_private_key(Path(path).read_bytes(), password=passphrase)
    if not isinstance(key, Ed25519PrivateKey):
        raise TypeError("governor key must be Ed25519")
    return key


def public_key_hex(key: Ed25519PrivateKey | Ed25519PublicKey) -> str:
    pub = key.public_key() if isinstance(key, Ed25519PrivateKey) else key
    raw = pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return raw.hex()


def load_public_key_hex(hex_str: str) -> Ed25519PublicKey:
    return Ed25519PublicKey.from_public_bytes(bytes.fromhex(hex_str.strip()))


def key_id(pub_hex: str) -> str:
    return sha256_hex(bytes.fromhex(pub_hex))[:16]
