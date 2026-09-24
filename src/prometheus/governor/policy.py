"""The governor's fixed rules (Good's docility clause, made enforceable).

The policy lives on the workstation. Nothing in a variant can change it: the
runtime never sees this file, and any variant touching a path outside the allowed
harness prefixes is vetoed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Only these parts of the harness may be rewritten by the Ashby loop.
DEFAULT_ALLOWED_PREFIXES = (
    "harness/prompts/",
    "harness/mesh/",
    "harness/memory_params/",
    "harness/tools/",
    "harness/evaluators/",
)

DEFAULT_ALLOWED_SUFFIXES = (".py", ".json", ".yaml", ".yml", ".md", ".txt")

# Modules a tool script may not import (defence in depth; the sandbox is primary).
DEFAULT_BANNED_MODULES = (
    "socket", "ssl", "urllib", "http", "requests", "httpx", "aiohttp", "ftplib",
    "smtplib", "telnetlib", "xmlrpc", "subprocess", "multiprocessing", "ctypes",
    "cffi", "importlib", "pty", "signal", "shutil", "webbrowser", "asyncio",
)

DEFAULT_BANNED_CALLS = ("eval", "exec", "compile", "__import__", "breakpoint")

DEFAULT_BANNED_ATTRS = (
    "system", "popen", "spawnl", "spawnv", "spawnve", "execv", "execve", "execl",
    "fork", "kill", "remove", "unlink", "rmdir", "rename", "chmod", "chown",
)


@dataclass
class Policy:
    run_id: str
    audit_seeds: list[int]
    model_pins: dict[str, str]
    max_generations: int = 10
    allowed_prefixes: tuple[str, ...] = DEFAULT_ALLOWED_PREFIXES
    allowed_suffixes: tuple[str, ...] = DEFAULT_ALLOWED_SUFFIXES
    banned_modules: tuple[str, ...] = DEFAULT_BANNED_MODULES
    banned_calls: tuple[str, ...] = DEFAULT_BANNED_CALLS
    banned_attrs: tuple[str, ...] = DEFAULT_BANNED_ATTRS
    max_file_bytes: int = 256_000
    max_bundle_bytes: int = 2_000_000
    approval_ttl_s: float = 2 * 3600
    heartbeat_ttl_s: float = 15 * 60
    heartbeat_interval_s: float = 5 * 60
    notes: dict[str, str] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: Path) -> "Policy":
        data = json.loads(Path(path).read_text())
        for key in ("allowed_prefixes", "allowed_suffixes", "banned_modules",
                    "banned_calls", "banned_attrs"):
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data)
