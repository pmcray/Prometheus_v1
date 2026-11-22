"""Lean 4 prover evaluation harness for Phase 0 (step 5b).

Given a Hidden-Law Lab world whose ground-truth statements are emitted with
``sorry`` placeholders, asks a prover model (e.g. Goedel-Prover-V2-8B or the
cortex candidate) to supply tactic proofs, checks that:
  1. No cheating primitives appear (``sorry``, ``admit``, ``axiom``, ``unsafe``).
  2. Every required theorem header matches the target statement verbatim (the
     prover cannot weaken or delete a theorem).
  3. Lean 4 compiles the completed source cleanly.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from ..lab.lean_layer import verify_lean_source, world_lean_source
from ..lab.world import HiddenLawWorld
from .client import ModelClient, ModelClientError

_CHEAT_PATTERN = re.compile(
    r"\b(?:sorry|admit|axiom|unsafe|implemented_by|extern|native_decide)\b"
)

PROVER_SYSTEM_PROMPT = """\
You are a Lean 4 theorem prover. You will be given a self-contained Lean 4 file
defining a finite algebraic structure (carrier `Fin N`, binary `op`, unary `f`)
and theorems whose proofs are replaced by `sorry`.

Replace every `sorry` with a valid Lean 4 proof (without using `sorry`, `admit`,
`axiom`, `unsafe`, or `native_decide`), keeping every `abbrev`, `def`, and
`theorem` statement identical. Reply ONLY with the complete Lean 4 file.
"""


@dataclass
class ProverResult:
    seed: int
    family: str
    proved: bool
    latency_s: float = 0.0
    error: str | None = None


def _strip_lean_fences(raw: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner).strip()
    return text


def _theorem_signatures(source: str) -> list[str]:
    """Extract every `theorem <name> : <prop>` signature (up to `:=`)."""
    sigs: list[str] = []
    for line in source.splitlines():
        s = line.strip()
        if s.startswith("theorem ") and ":=" in s:
            sigs.append(s.split(":=", 1)[0].strip())
    return sigs


def _definition_lines(source: str) -> list[str]:
    """Extract world definition lines (abbrev/def) so the prover cannot alter op/f."""
    prefixes = ("abbrev ", "def ", "private def ")
    return [
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith(prefixes)
    ]


def validate_prover_source(target_template: str, candidate_source: str) -> str | None:
    """Return an error reason if *candidate_source* cheats or alters definitions/statements."""
    # Strip line comments before checking for cheat tokens
    no_comments = "\n".join(
        line.split("--", 1)[0] for line in candidate_source.splitlines()
    )
    m = _CHEAT_PATTERN.search(no_comments)
    if m:
        return f"forbidden proof primitive: {m.group(0)!r}"

    if _definition_lines(candidate_source) != _definition_lines(target_template):
        return "world definitions (N/op/f) altered"

    expected = _theorem_signatures(target_template)
    actual = _theorem_signatures(candidate_source)
    if actual != expected:
        return (
            f"theorem statements altered (expected {len(expected)}, got {len(actual)})"
        )
    return None


def run_prover_episode(
    world: HiddenLawWorld,
    client: ModelClient,
    *,
    lean_timeout: int = 60,
) -> ProverResult:
    """Run one Lean proof-generation episode against *client* and verify with Lean."""
    template = world_lean_source(world, include_proofs=False)
    messages = [
        {"role": "system", "content": PROVER_SYSTEM_PROMPT},
        {"role": "user", "content": template},
    ]
    t0 = time.monotonic()
    try:
        raw = client.chat(messages)
    except ModelClientError as exc:
        return ProverResult(
            seed=world.seed, family=world.family, proved=False,
            latency_s=time.monotonic() - t0, error=str(exc),
        )
    latency = time.monotonic() - t0
    candidate = _strip_lean_fences(raw)

    reason = validate_prover_source(template, candidate)
    if reason is not None:
        return ProverResult(
            seed=world.seed, family=world.family, proved=False,
            latency_s=latency, error=reason,
        )

    ok, err = verify_lean_source(candidate, timeout=lean_timeout)
    return ProverResult(
        seed=world.seed,
        family=world.family,
        proved=ok,
        latency_s=latency,
        error=None if ok else err.strip()[:500],
    )
