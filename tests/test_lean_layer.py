"""Tests for prometheus.lab.lean_layer.

Compilation tests call lean --stdin and are marked ``lean`` so they can be
skipped when Lean is not installed::

    pytest -q -k "not lean"   # fast, no lean required
    pytest -q -k lean          # full round-trip (needs lean binary)

Each test that calls lean is also marked ``slow`` because a cold-start lean
process takes a few seconds.
"""

import re
import shutil

import pytest

from prometheus.lab import HiddenLawWorld, world_lean_source, verify_lean_source
from prometheus.lab.lean_layer import _lean_bin, verify_world


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lean_available() -> bool:
    try:
        _lean_bin()
        return True
    except FileNotFoundError:
        return False


lean = pytest.mark.skipif(not _lean_available(), reason="lean binary not found")
slow = pytest.mark.slow


# ---------------------------------------------------------------------------
# Source generation (no lean required)
# ---------------------------------------------------------------------------

class TestSourceGeneration:
    def test_source_is_string(self):
        w = HiddenLawWorld(seed=42)
        src = world_lean_source(w)
        assert isinstance(src, str) and len(src) > 0

    def test_source_contains_seed_comment(self):
        w = HiddenLawWorld(seed=99)
        src = world_lean_source(w)
        assert "World seed : 99" in src

    def test_source_contains_abbrev_N(self):
        w = HiddenLawWorld(seed=42)
        src = world_lean_source(w)
        n = len(w.symbols)
        assert f"abbrev N := {n}" in src

    def test_source_has_one_theorem_per_law(self):
        w = HiddenLawWorld(seed=42)
        src = world_lean_source(w)
        # Each of the 8 laws produces exactly one theorem (or two for f_bijective)
        # The positive f_bijective case splits into f_injective + f_surjective.
        theorem_count = len(re.findall(r"^theorem ", src, re.MULTILINE))
        # Minimum 8 (one per law), maximum 9 (f_bijective splits into two)
        assert 8 <= theorem_count <= 9

    def test_positive_theorems_match_ground_truth(self):
        """Laws that hold produce positive theorems; failing laws produce negations."""
        w = HiddenLawWorld(seed=42)
        src = world_lean_source(w)
        truth = w.ground_truth()
        for law, holds in truth.items():
            if law == "f_bijective":
                if holds:
                    assert "theorem f_injective" in src
                    assert "theorem f_surjective" in src
                else:
                    assert "theorem not_f_bijective" in src
            elif holds:
                assert f"theorem {law}" in src
                assert f"theorem not_{law}" not in src
            else:
                assert f"theorem not_{law}" in src
                assert f"theorem {law} " not in src

    def test_all_theorems_use_decide(self):
        w = HiddenLawWorld(seed=7)
        src = world_lean_source(w)
        theorem_blocks = re.findall(r"theorem .+? by decide", src)
        # Every theorem must end in "by decide"
        assert len(theorem_blocks) >= 8

    def test_op_table_dimensions(self):
        """The opTable array literal has exactly n rows of n elements."""
        for seed in [0, 1, 7, 42]:
            w = HiddenLawWorld(seed=seed)
            src = world_lean_source(w)
            n = len(w.symbols)
            # Count inner arrays in opTable line
            op_line = next(l for l in src.splitlines() if "opTable" in l)
            inner_arrays = re.findall(r"#\[[\d, ]+\]", op_line)
            assert len(inner_arrays) == n, f"seed={seed}: expected {n} rows"
            for arr in inner_arrays:
                elems = [x.strip() for x in arr[2:-1].split(",")]
                assert len(elems) == n

    def test_different_seeds_give_different_sources(self):
        src_a = world_lean_source(HiddenLawWorld(seed=10))
        src_b = world_lean_source(HiddenLawWorld(seed=11))
        assert src_a != src_b

    def test_same_seed_is_deterministic(self):
        src_a = world_lean_source(HiddenLawWorld(seed=55))
        src_b = world_lean_source(HiddenLawWorld(seed=55))
        assert src_a == src_b


# ---------------------------------------------------------------------------
# Lean compilation (requires lean binary)
# ---------------------------------------------------------------------------

# Seeds chosen to cover all 8 structure families.
_FAMILY_SEEDS = [0, 1, 2, 3, 6, 7, 12, 16]


@lean
@slow
@pytest.mark.parametrize("seed", _FAMILY_SEEDS)
def test_lean_compiles_per_family_seed(seed):
    """Generated source compiles without errors for one seed per family."""
    w = HiddenLawWorld(seed=seed)
    ok, err = verify_world(w, timeout=60)
    assert ok, f"seed={seed} family={w.family}: lean error:\n{err}"


@lean
@slow
def test_lean_source_string_compiles():
    """verify_lean_source accepts a raw string and returns (True, '')."""
    src = world_lean_source(HiddenLawWorld(seed=42))
    ok, err = verify_lean_source(src, timeout=60)
    assert ok, f"lean error:\n{err}"


@lean
@slow
def test_lean_rejects_bad_source():
    """verify_lean_source returns (False, ...) for invalid Lean."""
    ok, err = verify_lean_source("this is not valid lean code", timeout=30)
    assert not ok
    assert err  # some error message present


@lean
@slow
def test_lean_toolchain_is_v4_34():
    """The active lean binary reports version 4.34.x."""
    import subprocess
    lean_path = _lean_bin()
    result = subprocess.run([lean_path, "--version"], capture_output=True, text=True, timeout=15)
    assert "4.34" in result.stdout, f"unexpected version: {result.stdout.strip()}"
