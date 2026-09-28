"""Tests for prometheus.runtime.sandbox.

Normal tests: correct behaviour for allowed code.
Redteam tests (test_redteam_*): escape attempts that must be blocked.
"""

import time

import pytest

from prometheus.runtime.sandbox import (
    ALLOWED_MODULES,
    SandboxResult,
    SandboxViolation,
    run,
    run_or_raise,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ok(result: SandboxResult) -> SandboxResult:
    """Assert no error and return result."""
    assert result.error is None, f"Unexpected error: {result.error}"
    assert not result.timed_out
    assert not result.oom_killed
    return result


# ---------------------------------------------------------------------------
# Normal / positive tests
# ---------------------------------------------------------------------------

def test_simple_arithmetic():
    r = ok(run("return_value = 2 + 2"))
    assert r.return_value == 4


def test_inputs_are_accessible():
    r = ok(run("return_value = inputs['x'] * inputs['y']", {"x": 6, "y": 7}))
    assert r.return_value == 42


def test_allowed_math_import():
    r = ok(run("import math; return_value = math.sqrt(inputs['n'])", {"n": 144}))
    assert abs(r.return_value - 12.0) < 1e-9


def test_allowed_itertools():
    r = ok(run(
        "import itertools; return_value = list(itertools.islice(itertools.count(1), 5))",
    ))
    assert r.return_value == [1, 2, 3, 4, 5]


def test_allowed_collections():
    r = ok(run(
        "from collections import Counter; return_value = dict(Counter('abracadabra'))",
    ))
    assert r.return_value["a"] == 5


def test_allowed_json():
    r = ok(run(
        "import json; return_value = json.loads(inputs['s'])",
        {"s": '{"k": 99}'},
    ))
    assert r.return_value == {"k": 99}


def test_allowed_re():
    r = ok(run(
        "import re; return_value = bool(re.match(r'\\d+', inputs['s']))",
        {"s": "42abc"},
    ))
    assert r.return_value is True


def test_allowed_random():
    r = ok(run(
        "import random; random.seed(0); return_value = random.randint(0, 100)",
    ))
    assert isinstance(r.return_value, int)


def test_all_allowed_modules_importable():
    """Every module in ALLOWED_MODULES must be importable without error."""
    SKIP = {"typing_extensions"}   # may not be installed in minimal venv
    for mod in sorted(ALLOWED_MODULES - SKIP):
        r = run(f"import {mod}")
        assert r.error is None or "No module named" not in (r.error or ""), (
            f"ALLOWED_MODULE {mod!r} failed to import: {r.error}"
        )


def test_return_value_none_by_default():
    r = ok(run("x = 1 + 1"))
    assert r.return_value is None


def test_exception_in_code_is_reported():
    r = run("raise ValueError('oops')")
    assert r.error is not None
    assert "ValueError" in r.error
    assert r.return_value is None


def test_syntax_error_in_code_is_reported():
    r = run("def broken(: pass")
    assert r.error is not None


def test_complex_computation_returns_correctly():
    code = """
import math, functools, operator
primes = [n for n in range(2, inputs['limit']) if all(n % i for i in range(2, int(math.sqrt(n))+1))]
return_value = functools.reduce(operator.add, primes)
"""
    r = ok(run(code, {"limit": 50}))
    # sum of primes below 50: 2+3+5+7+11+13+17+19+23+29+31+37+41+43+47 = 328
    assert r.return_value == 328


def test_run_or_raise_succeeds_on_good_code():
    r = run_or_raise("return_value = 7")
    assert r.return_value == 7


def test_run_or_raise_raises_on_bad_code():
    with pytest.raises(SandboxViolation):
        run_or_raise("import socket")


# ---------------------------------------------------------------------------
# Redteam tests — every attempt to escape must be blocked
# ---------------------------------------------------------------------------

def test_redteam_network_socket_blocked():
    """import socket must be refused."""
    r = run("import socket")
    assert r.error is not None
    assert "socket" in r.error or "not permitted" in r.error


def test_redteam_network_urllib_blocked():
    r = run("import urllib.request")
    assert r.error is not None


def test_redteam_network_http_blocked():
    r = run("import http.client")
    assert r.error is not None


def test_redteam_subprocess_blocked():
    r = run("import subprocess; subprocess.run(['id'])")
    assert r.error is not None


def test_redteam_os_blocked():
    """Direct 'import os' must be refused."""
    r = run("import os; os.system('id')")
    assert r.error is not None


def test_redteam_sys_blocked():
    r = run("import sys; sys.exit(0)")
    assert r.error is not None


def test_redteam_open_builtin_removed():
    """open() must not be available."""
    r = run("f = open('/etc/passwd'); return_value = f.read(10)")
    assert r.error is not None


def test_redteam_eval_removed():
    r = run("return_value = eval('1+1')")
    assert r.error is not None


def test_redteam_exec_removed():
    r = run("exec('x=1')")
    assert r.error is not None


def test_redteam_compile_removed():
    r = run("compile('pass','<x>','exec')")
    assert r.error is not None


def test_redteam_import_escape_via_builtins():
    """Trying to recover __import__ through __builtins__ dict must fail."""
    r = run(
        "import builtins as b; b.__import__('socket')"
    )
    assert r.error is not None


def test_redteam_import_escape_via_globals():
    """__builtins__['__import__'] escape must be blocked."""
    r = run(
        "bi = __builtins__; orig = bi['__import__'] if isinstance(bi, dict) else bi.__import__; "
        "return_value = orig('socket')"
    )
    # builtins is not accessible (removed), but even if it were, our hook intercepts
    assert r.error is not None


def test_redteam_ctypes_blocked():
    r = run("import ctypes")
    assert r.error is not None


def test_redteam_multiprocessing_blocked():
    r = run("import multiprocessing; multiprocessing.Process(target=lambda: None).start()")
    assert r.error is not None


def test_redteam_threading_blocked():
    r = run("import threading")
    assert r.error is not None


def test_redteam_pathlib_blocked():
    r = run("import pathlib; pathlib.Path('/etc/passwd').read_text()")
    assert r.error is not None


def test_redteam_io_blocked():
    """io module gives file access via open() equivalents."""
    r = run("import io; io.open('/etc/passwd')")
    assert r.error is not None


def test_redteam_cpu_limit_kills_infinite_loop():
    """An infinite loop must be killed within cpu_seconds + a small margin."""
    start = time.time()
    r = run("while True: pass", cpu_seconds=2, wall_seconds=15)
    elapsed = time.time() - start
    assert r.timed_out or r.oom_killed or (r.error is not None and "limit" in r.error.lower()), (
        f"Expected sandbox kill, got: {r}"
    )
    assert elapsed < 10, f"Took too long to kill ({elapsed:.1f}s)"


def test_redteam_memory_limit_kills_allocation():
    """Allocating more RAM than the limit must terminate the child."""
    # Allocate 512 MB in one shot against a 64 MB limit.
    code = "x = b'A' * (512 * 1024 * 1024)"
    r = run(code, memory_mb=64, wall_seconds=15)
    assert r.oom_killed or r.timed_out or r.error is not None


def test_redteam_fork_bomb_limited():
    """A fork bomb must be contained by RLIMIT_NPROC."""
    code = """
import os
def bomb():
    while True:
        try:
            os.fork()
        except Exception:
            break
bomb()
"""
    # os is blocked by import hook, so this should fail at the import stage.
    r = run(code, max_procs=4, wall_seconds=10)
    assert r.error is not None


def test_redteam_gc_introspection_blocked():
    """gc gives access to all live objects — must be blocked."""
    r = run("import gc; return_value = len(gc.get_objects())")
    assert r.error is not None


def test_redteam_inspect_blocked():
    r = run("import inspect")
    assert r.error is not None


def test_redteam_importlib_blocked():
    """importlib can bypass the __import__ hook."""
    r = run("import importlib; importlib.import_module('socket')")
    assert r.error is not None


def test_redteam_pickle_blocked():
    """pickle can execute arbitrary code via __reduce__."""
    r = run("import pickle")
    assert r.error is not None
