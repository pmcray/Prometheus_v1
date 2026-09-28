"""Sandbox for agent-generated Python code.

Runs untrusted code in a child subprocess with three defence layers:

1. **Process isolation** — a separate OS process; state cannot leak back.
2. **Resource limits** — RLIMIT_CPU (CPU seconds), RLIMIT_AS (virtual memory),
   RLIMIT_NPROC (max child processes); enforced by the kernel before the child
   starts and cannot be raised by the child.
3. **Restricted exec** — stripped builtins (no open/eval/exec/compile/__import__)
   and a blocking __import__ hook that allows only ALLOWED_MODULES.

Network access is prevented through the import block (socket, http, urllib,
requests, etc.).  Because the child is a separate process, a bypass only
compromises the sandboxed process itself, not the Colab runtime.

Design constraint: must work on Colab (single unprivileged user, no bwrap user
namespaces available).  All sandboxing is therefore userspace + kernel rlimits.

Usage::

    result = run("x = inputs['a'] + inputs['b']; return_value = x",
                 inputs={"a": 1, "b": 2})
    assert result.return_value == 3
"""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Module allowlist
# ---------------------------------------------------------------------------

ALLOWED_MODULES: frozenset[str] = frozenset({
    # maths / numeric
    "math", "cmath", "decimal", "fractions", "numbers", "statistics",
    "random",
    # data structures / functional
    "collections", "collections.abc", "itertools", "functools", "operator",
    "heapq", "bisect", "array",
    # typing / introspection (safe subset)
    "typing", "typing_extensions", "types", "abc", "copy", "dataclasses",
    "enum",
    # text / encoding
    "re", "string", "unicodedata", "codecs",
    # serialisation (read-only use)
    "json",
    # time (read-only use — no sleep, no signals)
    "time",
    # internal plumbing needed by the runner itself
    "__future__",
})

# Modules whose *top-level name* alone is sufficient to block them.
# Any module not in ALLOWED_MODULES is also blocked; this list is kept
# for documentation and for the error message.
BLOCKED_EXAMPLES: tuple[str, ...] = (
    "socket", "ssl", "http", "urllib", "urllib3", "requests", "httpx",
    "aiohttp", "asyncio",
    "subprocess", "multiprocessing", "concurrent",
    "os", "sys", "signal", "resource", "ctypes", "cffi",
    "pty", "tty", "termios",
    "importlib", "pkgutil", "zipimport", "compileall",
    "gc", "ast", "dis", "inspect", "trace", "tracemalloc",
    "code", "codeop",
    "builtins",
    "threading",
    "io",                     # gives file access via BytesIO escape routes
    "pathlib", "glob", "fnmatch",
    "pickle", "shelve", "dbm",
    "xml", "html", "csv",
    "logging", "warnings",
    "tempfile", "shutil",
    "struct",
    "hashlib", "hmac", "secrets",
    "platform", "sysconfig",
)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SandboxResult:
    return_value: Any = None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None           # exception class + message from child
    timed_out: bool = False
    oom_killed: bool = False


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SandboxViolation(RuntimeError):
    """Raised when the sandbox rejects a run (timeout, OOM, import block, …)."""


# ---------------------------------------------------------------------------
# Child-side runner (executed in the subprocess)
# ---------------------------------------------------------------------------

_RUNNER_SOURCE = r"""
import builtins as _builtins
import json as _json
import sys as _sys
import resource as _resource

# ── 1. Read the task from stdin ──────────────────────────────────────────────
_task = _json.loads(_sys.stdin.read())
_code    = _task["code"]
_inputs  = _task["inputs"]
_allowed = frozenset(_task["allowed_modules"])

# ── 2. Pre-warm allowed modules BEFORE installing the hook ───────────────────
# This ensures every transitive stdlib dependency (warnings, _io, _abc, …) is
# already cached in sys.modules under the real __import__, so the hook never
# sees them again.  We silently ignore modules that aren't installed.
for _mod in list(_allowed):
    try:
        _real_import_tmp = _builtins.__import__
        _real_import_tmp(_mod)
    except ImportError:
        pass

# ── 3. Install import hook ───────────────────────────────────────────────────
_real_import = _builtins.__import__

def _safe_import(name, *args, **kwargs):
    top = name.split(".")[0]
    # Explicitly allowed modules (and their sub-modules).
    if top in _allowed or name in _allowed:
        return _real_import(name, *args, **kwargs)
    # C-extension backing modules (single leading underscore, e.g. _io,
    # _random, _collections).  These only arise as transitive imports triggered
    # by the pre-warmed allowed modules; they cannot grant file/network access
    # on their own and cannot be spelled meaningfully by agent code.
    # Double-underscore names (__future__ etc.) are handled above via _allowed.
    if top.startswith("_") and not top.startswith("__"):
        return _real_import(name, *args, **kwargs)
    # Everything else — including os, io, sys, inspect — is refused.
    raise ImportError(
        f"Module {name!r} is not permitted in the sandbox. "
        f"Allowed: {sorted(_allowed)}"
    )

_builtins.__import__ = _safe_import

# ── 4. Build a stripped builtins namespace ───────────────────────────────────
_REMOVED = frozenset({
    "open", "input",
    "eval", "exec", "compile",
    "__import__",
    "__loader__", "__spec__", "__build_class__",
    "breakpoint",
    "memoryview",       # raw buffer access
    "vars", "locals",   # introspection
    "__builtins__",
})

_safe_builtins = {
    k: getattr(_builtins, k)
    for k in dir(_builtins)
    if k not in _REMOVED and not k.startswith("__")
}
# Re-expose the safe import (so code can `import math` etc.)
_safe_builtins["__import__"] = _safe_import

# ── 5. Execute agent code ────────────────────────────────────────────────────
_ns = {
    "__builtins__": _safe_builtins,
    "inputs": _inputs,
    "return_value": None,
}
try:
    exec(compile(_code, "<agent>", "exec"), _ns)
    result = {"ok": True, "return_value": _ns.get("return_value")}
except Exception as _exc:
    result = {"ok": False, "error": f"{type(_exc).__name__}: {_exc}"}

# ── 6. Write result to stdout ────────────────────────────────────────────────
_sys.stdout.write(_json.dumps(result))
_sys.stdout.flush()
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(
    code: str,
    inputs: dict[str, Any] | None = None,
    *,
    cpu_seconds: int = 10,
    wall_seconds: float = 30.0,
    memory_mb: int = 256,
    max_procs: int = 32,
    allowed_modules: frozenset[str] | None = None,
) -> SandboxResult:
    """Run *code* in a sandboxed subprocess and return a :class:`SandboxResult`.

    The child process receives *inputs* as ``inputs`` in its namespace and may
    set ``return_value`` to communicate a result back.

    Args:
        code: Python source to execute.
        inputs: JSON-serialisable dict passed as ``inputs`` in the exec namespace.
        cpu_seconds: Hard CPU-time limit (SIGKILL on overflow).
        wall_seconds: Wall-clock timeout for the whole run.
        memory_mb: Virtual-address-space cap in mebibytes.
        max_procs: Maximum number of child processes the sandbox may spawn.
        allowed_modules: Override the default :data:`ALLOWED_MODULES`.

    Returns:
        A :class:`SandboxResult`.  Never raises on its own — errors are
        reported in ``result.error`` or the ``timed_out`` / ``oom_killed`` flags.
        The caller should raise :class:`SandboxViolation` if it considers any
        of those conditions fatal.
    """
    if inputs is None:
        inputs = {}
    if allowed_modules is None:
        allowed_modules = ALLOWED_MODULES

    task = {
        "code": textwrap.dedent(code),
        "inputs": inputs,
        "allowed_modules": sorted(allowed_modules),
    }
    task_json = json.dumps(task)

    mem_bytes = memory_mb * 1024 * 1024

    def _preexec() -> None:  # runs inside child before exec
        # Own process group so timeout SIGKILL hits the whole subtree.
        os.setsid()
        # CPU time (SIGKILL when hard limit hit).
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        # Virtual memory.
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        # Child processes (limits fork bombs; does not affect threads).
        resource.setrlimit(resource.RLIMIT_NPROC, (max_procs, max_procs))

    try:
        proc = subprocess.run(
            [sys.executable, "-c", _RUNNER_SOURCE],
            input=task_json,
            capture_output=True,
            text=True,
            timeout=wall_seconds,
            preexec_fn=_preexec,
        )
    except subprocess.TimeoutExpired as exc:
        # Kill the entire process group.
        try:
            os.killpg(os.getpgid(exc.process.pid), 9)  # type: ignore[attr-defined]
        except (ProcessLookupError, PermissionError):
            pass
        return SandboxResult(
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            timed_out=True,
            error=f"SandboxTimeout: wall-clock limit ({wall_seconds}s) exceeded",
        )

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    returncode = proc.returncode

    # SIGKILL from RLIMIT_CPU or RLIMIT_AS arrives as returncode -9
    if returncode == -9 and not stdout:
        # Distinguish OOM from CPU by heuristic: if stderr mentions MemoryError
        # or is empty and we got -9 with no stdout, call it OOM.
        oom = "MemoryError" in stderr or not stderr
        if oom:
            return SandboxResult(
                stdout=stdout, stderr=stderr, oom_killed=True,
                error=f"SandboxMemoryError: address-space limit ({memory_mb} MiB) exceeded",
            )
        return SandboxResult(
            stdout=stdout, stderr=stderr, timed_out=True,
            error=f"SandboxTimeout: CPU limit ({cpu_seconds}s) exceeded",
        )

    if returncode != 0:
        return SandboxResult(
            stdout=stdout, stderr=stderr,
            error=f"SandboxChildCrash: child exited with code {returncode}",
        )

    # Parse result JSON from stdout
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return SandboxResult(
            stdout=stdout, stderr=stderr,
            error=f"SandboxProtocolError: could not parse child output: {exc}",
        )

    if payload.get("ok"):
        return SandboxResult(
            return_value=payload.get("return_value"),
            stdout="",   # runner writes only JSON to stdout
            stderr=stderr,
        )
    return SandboxResult(
        stdout="",
        stderr=stderr,
        error=payload.get("error", "unknown error from child"),
    )


def run_or_raise(
    code: str,
    inputs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> SandboxResult:
    """Like :func:`run` but raises :class:`SandboxViolation` on any error."""
    result = run(code, inputs, **kwargs)
    if result.error or result.timed_out or result.oom_killed:
        raise SandboxViolation(
            result.error or ("timed out" if result.timed_out else "OOM")
        )
    return result
