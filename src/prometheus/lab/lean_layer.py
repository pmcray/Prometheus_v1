"""Lean 4 statement and proof generator for Hidden-Law Lab worlds.

Each Hidden-Law Lab world has a finite carrier {0..n-1}, a binary operation (op
table) and a unary map (f).  All eight laws are first-order over this finite
type, so every closed proposition is decidable and can be proved (or refuted)
by ``decide``.

Usage
-----
Single world::

    from prometheus.lab.lean_layer import world_lean_source, verify_lean_source
    src = world_lean_source(HiddenLawWorld(seed=42))
    ok, err = verify_lean_source(src)
    assert ok, err

Batch verification (e.g. 1,000 worlds for Phase 0 exit criterion)::

    prometheus-verify-worlds --seeds 0-999 --workers 8 --out lean/verified_1000_worlds.json

No import is needed: ``Fin``, ``Array``, and ``decide`` are all in Lean core.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import textwrap
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .world import HiddenLawWorld

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_LEAN_BINARY: str | None = None


PROVER_LEAN_TOOLCHAIN = "leanprover/lean4:v4.9.0"


def _lean_project_dir() -> Path | None:
    """Locate ``lean/PrometheusLaws`` so elan honours its ``lean-toolchain`` pin."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "lean" / "PrometheusLaws"
        if (candidate / "lean-toolchain").is_file():
            return candidate
    return None


def prover_lean_project_dir() -> Path | None:
    """Locate the separate Lean 4.9 prover project (``lean/PrometheusProver49``)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "lean" / "PrometheusProver49"
        if (candidate / "lean-toolchain").is_file():
            return candidate
    return None


def _lean_bin() -> str:
    """Return the path to the ``lean`` executable, searching PATH and ``~/.elan``."""
    global _LEAN_BINARY
    if _LEAN_BINARY is not None:
        return _LEAN_BINARY
    found = shutil.which("lean")
    if found:
        _LEAN_BINARY = found
        return _LEAN_BINARY
    elan_lean = Path.home() / ".elan" / "bin" / "lean"
    if elan_lean.exists():
        _LEAN_BINARY = str(elan_lean)
        return _LEAN_BINARY
    raise FileNotFoundError(
        "lean binary not found on PATH or in ~/.elan/bin. "
        "Install Lean 4 via elan: https://github.com/leanprover/elan"
    )


def lean_version(
    timeout: int = 15,
    *,
    project_dir: Path | str | None = None,
) -> str:
    """Return the active Lean version string inside *project_dir* (default: ``lean/PrometheusLaws``)."""
    cwd = Path(project_dir) if project_dir is not None else _lean_project_dir()
    res = subprocess.run(
        [_lean_bin(), "--version"],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=cwd,
    )
    return res.stdout.strip()


def _op_array(op: list[list[int]], n: int) -> str:
    """Render the op table as a nested Lean Array literal."""
    rows = ", ".join(
        "#[" + ", ".join(str(op[a][b]) for b in range(n)) + "]"
        for a in range(n)
    )
    return f"#[{rows}]"


def _f_array(f: list[int], n: int) -> str:
    """Render the f map as a Lean Array literal."""
    return "#[" + ", ".join(str(f[a]) for a in range(n)) + "]"


def _world_theorems(truth: dict[str, bool], *, include_proofs: bool = True) -> list[str]:
    """Build the 8 (or 9 when f_bijective holds) Lean theorems for *truth*."""
    proof = " := by decide" if include_proofs else " := by sorry"
    theorems: list[str] = []

    if truth["commutative"]:
        theorems.append(f"theorem commutative : ∀ a b : Fin N, op a b = op b a{proof}")
    else:
        theorems.append(f"theorem not_commutative : ¬∀ a b : Fin N, op a b = op b a{proof}")

    if truth["associative"]:
        theorems.append(f"theorem associative : ∀ a b c : Fin N, op (op a b) c = op a (op b c){proof}")
    else:
        theorems.append(f"theorem not_associative : ¬∀ a b c : Fin N, op (op a b) c = op a (op b c){proof}")

    if truth["has_identity"]:
        theorems.append(f"theorem has_identity : ∃ e : Fin N, ∀ a : Fin N, op e a = a ∧ op a e = a{proof}")
    else:
        theorems.append(f"theorem not_has_identity : ¬∃ e : Fin N, ∀ a : Fin N, op e a = a ∧ op a e = a{proof}")

    if truth["has_inverses"]:
        theorems.append(
            "theorem has_inverses : "
            "(∃ e : Fin N, (∀ a : Fin N, op e a = a ∧ op a e = a) ∧ "
            f"∀ a : Fin N, ∃ b : Fin N, op a b = e ∧ op b a = e){proof}"
        )
    else:
        theorems.append(
            "theorem not_has_inverses : "
            "¬(∃ e : Fin N, (∀ a : Fin N, op e a = a ∧ op a e = a) ∧ "
            f"∀ a : Fin N, ∃ b : Fin N, op a b = e ∧ op b a = e){proof}"
        )

    if truth["idempotent"]:
        theorems.append(f"theorem idempotent : ∀ a : Fin N, op a a = a{proof}")
    else:
        theorems.append(f"theorem not_idempotent : ¬∀ a : Fin N, op a a = a{proof}")

    if truth["f_involution"]:
        theorems.append(f"theorem f_involution : ∀ a : Fin N, f (f a) = a{proof}")
    else:
        theorems.append(f"theorem not_f_involution : ¬∀ a : Fin N, f (f a) = a{proof}")

    if truth["f_homomorphism"]:
        theorems.append(f"theorem f_homomorphism : ∀ a b : Fin N, f (op a b) = op (f a) (f b){proof}")
    else:
        theorems.append(f"theorem not_f_homomorphism : ¬∀ a b : Fin N, f (op a b) = op (f a) (f b){proof}")

    if truth["f_bijective"]:
        theorems.append(
            f"theorem f_injective : ∀ a b : Fin N, f a = f b → a = b{proof}\n"
            f"theorem f_surjective : ∀ b : Fin N, ∃ a : Fin N, f a = b{proof}"
        )
    else:
        theorems.append(
            "theorem not_f_bijective : "
            f"¬((∀ a b : Fin N, f a = f b → a = b) ∧ ∀ b : Fin N, ∃ a : Fin N, f a = b){proof}"
        )
    return theorems


def _world_defs(world: HiddenLawWorld, *, lean_compat: str = "4.34") -> str:
    n = len(world.symbols)
    op = world._op   # noqa: SLF001  (governor/ground-truth use only)
    f = world._f     # noqa: SLF001
    if lean_compat == "4.9":
        op_expr = "(opTable[a.val]!)[b.val]!"
        f_expr = "fTable[a.val]!"
    elif lean_compat == "4.34":
        op_expr = "(opTable[a]!)[b]!"
        f_expr = "fTable[a]!"
    else:
        raise ValueError(f"unsupported lean_compat {lean_compat!r}; expected '4.34' or '4.9'")
    return textwrap.dedent(f"""\
        abbrev N := {n}

        private def opTable : Array (Array (Fin N)) := {_op_array(op, n)}
        def op (a b : Fin N) : Fin N := {op_expr}

        private def fTable : Array (Fin N) := {_f_array(f, n)}
        def f (a : Fin N) : Fin N := {f_expr}
    """)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def world_lean_source(
    world: HiddenLawWorld,
    *,
    include_proofs: bool = True,
    lean_compat: str = "4.34",
) -> str:
    """Return a self-contained Lean 4 source string for *world*.

    The source contains one ``theorem`` per law (two when ``f_bijective``
    holds), each proved by ``decide`` when *include_proofs* is True.
    Pass ``lean_compat="4.9"`` for Lean 4.9.0 / Goedel-Prover-V2 compatibility
    (uses explicit ``.val`` indexing into ``Array``, valid on both 4.9 and 4.34).
    """
    n = len(world.symbols)
    truth = world.ground_truth()
    header = textwrap.dedent(f"""\
        -- Auto-generated by prometheus.lab.lean_layer
        -- World seed : {world.seed}
        -- Family     : {world.family}
        -- Map kind   : {world.map_kind}
        -- Carrier    : Fin {n}
        -- Laws       : {', '.join(f'{k}={v}' for k, v in truth.items())}

        -- No imports needed: Fin, Array, and `decide` are in Lean 4 core.

    """)
    defs = _world_defs(world, lean_compat=lean_compat)
    theorems = _world_theorems(truth, include_proofs=include_proofs)
    return header + defs + "\n" + "\n".join(theorems) + "\n"


def batch_lean_source(worlds: Sequence[HiddenLawWorld]) -> str:
    """Combine multiple worlds into one Lean file using isolated namespaces."""
    blocks: list[str] = ["-- Auto-generated batch by prometheus.lab.lean_layer\n"]
    for w in worlds:
        truth = w.ground_truth()
        body = _world_defs(w) + "\n" + "\n".join(_world_theorems(truth, include_proofs=True))
        indented = textwrap.indent(body, "  ")
        blocks.append(f"namespace World_{w.seed}\n{indented}\nend World_{w.seed}\n")
    return "\n".join(blocks)


def verify_lean_source(
    source: str,
    timeout: int = 60,
    *,
    project_dir: Path | str | None = None,
) -> tuple[bool, str]:
    """Check *source* by piping it to ``lean --stdin`` inside *project_dir* (default: ``lean/PrometheusLaws``)."""
    lean = _lean_bin()
    cwd = Path(project_dir) if project_dir is not None else _lean_project_dir()
    try:
        result = subprocess.run(
            [lean, "--stdin"],
            input=source,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
    except subprocess.TimeoutExpired:
        return False, f"lean --stdin timed out after {timeout}s"
    if result.returncode != 0:
        return False, result.stderr or result.stdout
    return True, ""


def verify_world(
    world: HiddenLawWorld,
    timeout: int = 60,
    *,
    lean_compat: str = "4.34",
    project_dir: Path | str | None = None,
) -> tuple[bool, str]:
    """Generate and verify the Lean source for *world* in one call."""
    return verify_lean_source(
        world_lean_source(world, lean_compat=lean_compat),
        timeout=timeout,
        project_dir=project_dir,
    )


@dataclass(frozen=True)
class BatchVerifyResult:
    total_worlds: int
    verified_worlds: int
    failed_seeds: dict[int, str]
    total_theorems: int
    family_counts: dict[str, int]
    lean_version: str
    elapsed_s: float
    digest_sha256: str

    @property
    def ok(self) -> bool:
        return self.verified_worlds == self.total_worlds and not self.failed_seeds

    def to_dict(self) -> dict:
        return {**asdict(self), "ok": self.ok}


def verify_world_batch(
    seeds: Iterable[int],
    *,
    chunk_size: int = 25,
    workers: int = 4,
    timeout: int = 120,
) -> BatchVerifyResult:
    """Verify ground-truth Lean proofs for all *seeds* in parallel chunks."""
    seed_list = list(seeds)
    t0 = time.monotonic()
    worlds = [HiddenLawWorld(seed=s) for s in seed_list]
    family_counts = dict(sorted(Counter(w.family for w in worlds).items()))
    total_theorems = sum(9 if w.ground_truth()["f_bijective"] else 8 for w in worlds)

    hasher = hashlib.sha256()
    for w in worlds:
        hasher.update(world_lean_source(w).encode("utf-8"))

    chunks = [worlds[i:i + chunk_size] for i in range(0, len(worlds), chunk_size)]
    failed: dict[int, str] = {}

    def _verify_chunk(chunk: list[HiddenLawWorld]) -> dict[int, str]:
        ok, err = verify_lean_source(batch_lean_source(chunk), timeout=timeout)
        if ok:
            return {}
        # Pinpoint exact failing seeds if a chunk fails
        chunk_failures: dict[int, str] = {}
        for w in chunk:
            w_ok, w_err = verify_world(w, timeout=timeout)
            if not w_ok:
                chunk_failures[w.seed] = w_err
        return chunk_failures or {chunk[0].seed: err}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = [pool.submit(_verify_chunk, c) for c in chunks]
        for fut in as_completed(futs):
            failed.update(fut.result())

    return BatchVerifyResult(
        total_worlds=len(worlds),
        verified_worlds=len(worlds) - len(failed),
        failed_seeds=dict(sorted(failed.items())),
        total_theorems=total_theorems,
        family_counts=family_counts,
        lean_version=lean_version(),
        elapsed_s=round(time.monotonic() - t0, 2),
        digest_sha256=hasher.hexdigest(),
    )


def _parse_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="prometheus-verify-worlds")
    p.add_argument("--seeds", default="0-999", help="Seed range, e.g. 0-999")
    p.add_argument("--chunk-size", type=int, default=25)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--out", default=None, help="Optional JSON certificate output path")
    args = p.parse_args(argv)

    seeds = _parse_seeds(args.seeds)
    res = verify_world_batch(
        seeds, chunk_size=args.chunk_size, workers=args.workers, timeout=args.timeout
    )
    payload = {"seeds": args.seeds, **res.to_dict()}
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"Verified {res.verified_worlds}/{res.total_worlds} worlds "
        f"({res.total_theorems} theorems) in {res.elapsed_s:.2f}s — "
        f"{'PASS' if res.ok else 'FAIL'}"
    )
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
