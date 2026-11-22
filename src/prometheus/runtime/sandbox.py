"""Sandbox for agent-generated Python code.

Runs untrusted code in a child subprocess with four defence layers:

1. **Process isolation** — a separate isolated OS process (``python -I``); state
   cannot leak back to the runtime.
2. **Resource limits** — ``RLIMIT_CPU`` (soft ``SIGXCPU`` / hard ``SIGKILL``),
   ``RLIMIT_AS`` (virtual memory), ``RLIMIT_NPROC`` (max child processes), plus
   a wall-clock timeout that kills the entire process group.
3. **Static AST screen** — rejects forbidden builtins, dunder/frame escape
   attributes, and blocked imports before execution starts.
4. **Runtime import hook + CPython audit hook** — ``sys.addaudithook`` blocks
   C-level file I/O (``open``), networking (``socket.*``), OS/process execution
   (``os.*``, ``subprocess.*``), FFI (``ctypes.*``), GC/frame introspection, and
   non-allowlisted imports even if Python object-graph gadgets are attempted.

Design constraint: must work on Colab (single unprivileged user, no bwrap user
namespaces available).  All sandboxing is therefore userspace + CPython audit
hooks + kernel rlimits.

Usage::

    result = run("x = inputs['a'] + inputs['b']; return_value = x",
                 inputs={"a": 1, "b": 2})
    assert result.return_value == 3
"""

from __future__ import annotations

import ast
import json
import os
import resource
import signal
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
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
    # text / encoding (note: codecs is intentionally excluded — codecs.open() reads files)
    "re", "string", "unicodedata",
    # serialisation (in-memory only)
    "json",
    # time (read-only proxy — no sleep, no signals)
    "time",
    # internal plumbing needed by the runner itself
    "__future__",
})

# Modules whose *top-level name* alone is sufficient to block them.
# Any module not in ALLOWED_MODULES is also blocked; this list is kept
# for documentation and for diagnostics.
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
    "io", "_io", "_socket", "_posixsubprocess", "_ctypes",
    "codecs",
    "pathlib", "glob", "fnmatch",
    "pickle", "shelve", "dbm",
    "xml", "html", "csv",
    "logging", "warnings",
    "tempfile", "shutil",
    "struct",
    "hashlib", "hmac", "secrets",
    "platform", "sysconfig",
)

# Static AST blocklists (defence in depth before spawning the child).
_AST_BLOCKED_NAMES: frozenset[str] = frozenset({
    "open", "input",
    "eval", "exec", "compile",
    "__import__", "__loader__", "__spec__", "__builtins__",
    "breakpoint", "memoryview",
    "vars", "locals", "globals", "dir",
    "getattr", "setattr", "delattr",
})

_AST_BLOCKED_ATTRS: frozenset[str] = frozenset({
    "__subclasses__", "__globals__", "__builtins__", "__code__",
    "__closure__", "__loader__", "__spec__", "__import__",
    "__mro__", "__bases__", "__base__", "__dict__",
    "__reduce__", "__reduce_ex__", "__getattribute__",
    "__class__", "__self__", "__func__",
    "f_back", "f_globals", "f_locals", "f_builtins", "f_code",
    "tb_frame", "gi_frame", "cr_frame", "ag_frame",
    "_os", "_io", "_sys", "_builtins", "_posixsubprocess",
})


# ---------------------------------------------------------------------------
# Result type & exceptions
# ---------------------------------------------------------------------------

@dataclass
class SandboxResult:
    return_value: Any = None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None           # exception class + message from child
    timed_out: bool = False
    oom_killed: bool = False


class SandboxViolation(RuntimeError):
    """Raised when the sandbox rejects a run (timeout, OOM, import block, …)."""


# ---------------------------------------------------------------------------
# Pre-execution static AST screen
# ---------------------------------------------------------------------------

def _screen_ast(code: str, allowed_modules: frozenset[str]) -> str | None:
    """Return an error string if *code* fails syntax or static safety checks."""
    try:
        tree = ast.parse(code, filename="<agent>")
    except SyntaxError as exc:
        return f"SyntaxError: {exc}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in allowed_modules and alias.name not in allowed_modules:
                    return (
                        f"ImportError: Module {alias.name!r} is not permitted in the sandbox. "
                        f"Allowed: {sorted(allowed_modules)}"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                return "ImportError: relative imports are not permitted in the sandbox"
            mod = node.module or ""
            top = mod.split(".")[0]
            if top not in allowed_modules and mod not in allowed_modules:
                return (
                    f"ImportError: Module {mod!r} is not permitted in the sandbox. "
                    f"Allowed: {sorted(allowed_modules)}"
                )
        elif isinstance(node, ast.Name) and node.id in _AST_BLOCKED_NAMES:
            return f"SandboxPolicyError: builtin or name {node.id!r} is not permitted in the sandbox"
        elif isinstance(node, ast.Attribute) and node.attr in _AST_BLOCKED_ATTRS:
            return f"SandboxPolicyError: attribute {node.attr!r} is not permitted in the sandbox"
    return None


# ---------------------------------------------------------------------------
# Child-side runner (executed in the subprocess)
# ---------------------------------------------------------------------------

_RUNNER_SOURCE = r"""
import builtins as _builtins
import json as _json
import os as _os
import sys as _sys
import time as _real_time
import types as _types

_result_fd = int(_sys.argv[1])
_os_write = _os.write
_os_close = _os.close

# ── 1. Read the task from stdin ──────────────────────────────────────────────
_task = _json.loads(_sys.stdin.read())
_code    = _task["code"]
_inputs  = _task["inputs"]
_allowed = frozenset(_task["allowed_modules"])

# ── 2. Pre-warm allowed modules BEFORE installing hooks ──────────────────────
# Every transitive stdlib dependency is cached in sys.modules while the real
# __import__ is active.  Silently ignore optional modules not installed.
_real_import = _builtins.__import__
for _mod in list(_allowed):
    try:
        _real_import(_mod)
    except ImportError:
        pass

# Build a read-only `time` proxy (no sleep, no tzset).
_safe_time = _types.ModuleType("time")
for _attr in (
    "time", "monotonic", "perf_counter", "process_time",
    "time_ns", "monotonic_ns", "perf_counter_ns", "process_time_ns",
    "gmtime", "localtime", "strftime", "timezone", "tzname",
):
    if hasattr(_real_time, _attr):
        setattr(_safe_time, _attr, getattr(_real_time, _attr))

# ── 3. Compile agent code before locking down with audit hook ────────────────
_compiled = compile(_code, "<agent>", "exec")

# ── 4. Install strict import hook (NO single-underscore wildcard) ────────────
def _safe_import(name, *args, **kwargs):
    top = name.split(".")[0]
    if name == "time" and "time" in _allowed:
        return _safe_time
    if top in _allowed or name in _allowed:
        return _real_import(name, *args, **kwargs)
    raise ImportError(
        f"Module {name!r} is not permitted in the sandbox. "
        f"Allowed: {sorted(_allowed)}"
    )

_builtins.__import__ = _safe_import

# ── 5. Build a stripped builtins namespace ───────────────────────────────────
# Note: __build_class__ is retained so agent code may define classes,
# dataclasses, and enums.
_REMOVED = frozenset({
    "open", "input",
    "eval", "exec", "compile",
    "__import__",
    "__loader__", "__spec__",
    "breakpoint",
    "memoryview",
    "vars", "locals", "globals", "dir",
    "getattr", "setattr", "delattr",
    "__builtins__",
})

_safe_builtins = {
    k: getattr(_builtins, k)
    for k in dir(_builtins)
    if k not in _REMOVED and not k.startswith("__")
}
_safe_builtins["__import__"] = _safe_import
_safe_builtins["__build_class__"] = _builtins.__build_class__

# ── 6. Install irreversible CPython audit hook ───────────────────────────────
_BLOCKED_EXACT = frozenset({
    "open",
    "sys._getframe",
    "sys._current_frames",
    "sys.settrace",
    "sys.setprofile",
    "sys.addaudithook",
    "sys.excepthook",
    "sys.unraisablehook",
    "code.__new__",
    "_thread.start_new_thread",
})

_BLOCKED_PREFIXES = (
    "socket.", "os.", "subprocess.", "ctypes.", "shutil.",
    "urllib.", "http.", "ftplib.", "smtplib.", "webbrowser.",
    "fcntl.", "resource.", "signal.", "syslog.", "tempfile.",
    "gc.", "mmap.", "sqlite3.", "pickle.", "marshal.",
    "pty.", "posix.", "nt.", "glob.", "pathlib.",
    "winreg.", "msvcrt.", "threading.", "_thread.",
)

def _audit_hook(event, args):
    if event in _BLOCKED_EXACT or event.startswith(_BLOCKED_PREFIXES):
        raise PermissionError(f"Sandbox blocked audit event: {event}")
    if event == "import":
        mod_name = args[0] if args else ""
        if isinstance(mod_name, str):
            top = mod_name.split(".")[0]
            if top not in _allowed and mod_name not in _allowed:
                raise ImportError(f"Module {mod_name!r} is not permitted in the sandbox")

_sys.addaudithook(_audit_hook)

# ── 7. Execute agent code ────────────────────────────────────────────────────
_ns = {
    "__name__": "__agent__",
    "__builtins__": _safe_builtins,
    "inputs": _inputs,
    "return_value": None,
}
try:
    exec(_compiled, _ns)
    _payload = {"ok": True, "return_value": _ns.get("return_value")}
except MemoryError as _exc:
    _payload = {"ok": False, "oom": True, "error": f"MemoryError: {_exc}"}
except BaseException as _exc:
    _payload = {"ok": False, "error": f"{type(_exc).__name__}: {_exc}"}

# ── 8. Write result JSON over the dedicated result pipe ──────────────────────
try:
    _encoded = _json.dumps(_payload).encode("utf-8")
except (TypeError, ValueError) as _exc:
    _encoded = _json.dumps({
        "ok": False,
        "error": f"TypeError: return_value is not JSON-serialisable: {_exc}",
    }).encode("utf-8")

_sys.stdout.flush()
_sys.stderr.flush()
_view = _encoded
while _view:
    _n = _os_write(_result_fd, _view)
    _view = _view[_n:]
_os_close(_result_fd)
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_clamped_rlimit(kind: int, soft: int, hard: int) -> None:
    """Apply an rlimit without attempting to raise the current hard limit."""
    cur_soft, cur_hard = resource.getrlimit(kind)
    if cur_hard != resource.RLIM_INFINITY:
        hard = min(hard, cur_hard)
        soft = min(soft, hard)
    resource.setrlimit(kind, (soft, hard))


def _read_all_fd(fd: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = os.read(fd, 65536)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


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
    set ``return_value`` to communicate a result back.  Any ``print()`` output
    is captured in ``result.stdout`` without disturbing the result protocol.
    """
    if inputs is None:
        inputs = {}
    if allowed_modules is None:
        allowed_modules = ALLOWED_MODULES

    dedented = textwrap.dedent(code)
    ast_err = _screen_ast(dedented, allowed_modules)
    if ast_err is not None:
        return SandboxResult(error=ast_err)

    task = {
        "code": dedented,
        "inputs": inputs,
        "allowed_modules": sorted(allowed_modules),
    }
    task_json = json.dumps(task)
    mem_bytes = memory_mb * 1024 * 1024

    def _preexec() -> None:  # runs inside child before exec
        os.setsid()
        # Soft limit triggers SIGXCPU (-24); hard limit sends SIGKILL 1s later.
        _set_clamped_rlimit(resource.RLIMIT_CPU, cpu_seconds, cpu_seconds + 1)
        _set_clamped_rlimit(resource.RLIMIT_AS, mem_bytes, mem_bytes)
        _set_clamped_rlimit(resource.RLIMIT_NPROC, max_procs, max_procs)

    r_fd, w_fd = os.pipe()
    os.set_inheritable(w_fd, True)
    t0 = time.monotonic()

    # Minimal sanitised environment: never leak host/Colab secrets into child.
    child_env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LC_ALL": "C.UTF-8",
        "LANG": "C.UTF-8",
    }

    try:
        proc = subprocess.Popen(
            [sys.executable, "-I", "-c", _RUNNER_SOURCE, str(w_fd)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=child_env,
            preexec_fn=_preexec,
            pass_fds=(w_fd,),
        )
    except Exception:
        os.close(r_fd)
        os.close(w_fd)
        raise

    # Close the write end in the parent so read() sees EOF when the child exits.
    os.close(w_fd)

    try:
        try:
            stdout, stderr = proc.communicate(input=task_json, timeout=wall_seconds)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            stdout, stderr = proc.communicate()
            return SandboxResult(
                stdout=stdout or "",
                stderr=stderr or "",
                timed_out=True,
                error=f"SandboxTimeout: wall-clock limit ({wall_seconds}s) exceeded",
            )
        raw_result = _read_all_fd(r_fd).decode("utf-8", errors="replace")
    finally:
        os.close(r_fd)

    elapsed = time.monotonic() - t0
    stdout = stdout or ""
    stderr = stderr or ""
    returncode = proc.returncode

    # Soft RLIMIT_CPU delivers SIGXCPU cleanly.
    if returncode == -signal.SIGXCPU:
        return SandboxResult(
            stdout=stdout, stderr=stderr, timed_out=True,
            error=f"SandboxTimeout: CPU limit ({cpu_seconds}s) exceeded",
        )

    # Hard SIGKILL without a JSON payload: distinguish CPU vs OOM.
    if returncode == -signal.SIGKILL and not raw_result:
        if "MemoryError" not in stderr and elapsed >= cpu_seconds * 0.9:
            return SandboxResult(
                stdout=stdout, stderr=stderr, timed_out=True,
                error=f"SandboxTimeout: CPU limit ({cpu_seconds}s) exceeded",
            )
        return SandboxResult(
            stdout=stdout, stderr=stderr, oom_killed=True,
            error=f"SandboxMemoryError: address-space limit ({memory_mb} MiB) exceeded",
        )

    if returncode != 0 and not raw_result:
        if "MemoryError" in stderr:
            return SandboxResult(
                stdout=stdout, stderr=stderr, oom_killed=True,
                error=f"SandboxMemoryError: address-space limit ({memory_mb} MiB) exceeded",
            )
        return SandboxResult(
            stdout=stdout, stderr=stderr,
            error=f"SandboxChildCrash: child exited with code {returncode}",
        )

    # Parse JSON result from the dedicated pipe.
    try:
        payload = json.loads(raw_result)
    except json.JSONDecodeError as exc:
        return SandboxResult(
            stdout=stdout, stderr=stderr,
            error=f"SandboxProtocolError: could not parse child output: {exc}",
        )

    if payload.get("ok"):
        return SandboxResult(
            return_value=payload.get("return_value"),
            stdout=stdout,
            stderr=stderr,
        )
    return SandboxResult(
        stdout=stdout,
        stderr=stderr,
        oom_killed=bool(payload.get("oom", False)),
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
