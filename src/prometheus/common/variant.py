"""Harness variant bundles, shared by the governor and the Colab runtime.

A variant is a directory:

    <bundle>/manifest.json   run_id, generation, parent_hash, files{relpath: sha256}, ...
    <bundle>/files/<relpath> the harness files the variant installs

The variant hash is the canonical hash of the manifest. Because the manifest lists
the SHA-256 of every file, one hash pins the whole bundle. `load_bundle` refuses
bundles whose files do not match the manifest exactly (missing, extra, altered,
symlinked or path-escaping files).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .canonical import hash_obj, sha256_hex


class BundleError(ValueError):
    pass


@dataclass(frozen=True)
class Bundle:
    root: Path
    manifest: dict[str, Any]
    variant_hash: str

    def file_path(self, rel: str) -> Path:
        return self.root / "files" / rel

    def read_text(self, rel: str) -> str:
        return self.file_path(rel).read_text(encoding="utf-8")


def safe_relpath(rel: str) -> str:
    """Normalise a relative POSIX path and reject anything that could escape."""
    if not rel or "\\" in rel or "\x00" in rel:
        raise BundleError(f"invalid path {rel!r}")
    p = PurePosixPath(rel)
    if p.is_absolute() or any(part in ("..", "") for part in p.parts):
        raise BundleError(f"path escapes bundle: {rel!r}")
    return p.as_posix()


def scan_files(files_dir: Path) -> dict[str, str]:
    """Map each regular file under files_dir to its SHA-256. Symlinks are refused."""
    out: dict[str, str] = {}
    if not files_dir.exists():
        return out
    for dirpath, dirnames, filenames in os.walk(files_dir, followlinks=False):
        for d in dirnames:
            if (Path(dirpath) / d).is_symlink():
                raise BundleError(f"symlinked directory not allowed: {d}")
        for name in filenames:
            full = Path(dirpath) / name
            if full.is_symlink() or not full.is_file():
                raise BundleError(f"not a regular file: {full}")
            rel = safe_relpath(full.relative_to(files_dir).as_posix())
            out[rel] = sha256_hex(full.read_bytes())
    return out


def write_bundle(root: Path, manifest_fields: dict[str, Any], files: dict[str, str]) -> Bundle:
    """Create a bundle on disk (used by the runtime to propose, and by tests)."""
    root = Path(root)
    files_dir = root / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        rel = safe_relpath(rel)
        dest = files_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
    manifest = dict(manifest_fields)
    manifest["files"] = scan_files(files_dir)
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8"
    )
    return Bundle(root, manifest, hash_obj(manifest))


def load_bundle(root: Path) -> Bundle:
    root = Path(root)
    mpath = root / "manifest.json"
    if not mpath.is_file() or mpath.is_symlink():
        raise BundleError("manifest.json missing")
    try:
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BundleError(f"manifest is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        raise BundleError("manifest must be an object with a 'files' map")
    listed = {safe_relpath(k): v for k, v in manifest["files"].items()}
    actual = scan_files(root / "files")
    if listed != actual:
        missing = sorted(set(listed) - set(actual))
        extra = sorted(set(actual) - set(listed))
        changed = sorted(k for k in set(listed) & set(actual) if listed[k] != actual[k])
        raise BundleError(
            f"bundle does not match manifest (missing={missing}, extra={extra}, changed={changed})"
        )
    return Bundle(root, manifest, hash_obj(manifest))
