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

import dataclasses
import re
import time
from dataclasses import dataclass
from typing import Any

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
    attempts: int = 1
    winning_attempt: int | None = None
    error_kind: str | None = None
    lean_compat: str = "4.34"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _strip_lean_fences(raw: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    # Extract the last fenced Lean block if prose (e.g. a proof plan) precedes it.
    fenced = re.findall(r"```(?:lean4?|)\s*\n(.*?)```", text, flags=re.DOTALL)
    if fenced:
        return fenced[-1].strip()
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


def _classify_validation_error(reason: str) -> str:
    if "forbidden proof primitive" in reason:
        return "cheat_rejected"
    return "statement_altered"


def run_prover_episode(
    world: HiddenLawWorld,
    client: ModelClient,
    *,
    lean_timeout: int = 60,
    lean_compat: str = "4.34",
    project_dir: str | Any | None = None,
    attempts: int = 1,
    sample_temperature: float = 0.6,
) -> ProverResult:
    """Run one Lean proof-generation episode against *client* and verify with Lean.

    Supports ``pass@k`` via *attempts* (attempt 1 uses ``client.temperature``,
    typically ``0.0``; subsequent attempts sample at *sample_temperature*).
    Every attempt is validated through :func:`validate_prover_source` before Lean
    verification; cheating primitives or altered theorems are always rejected.
    """
    n_attempts = max(1, int(attempts))
    template = world_lean_source(world, include_proofs=False, lean_compat=lean_compat)
    messages = [
        {"role": "system", "content": PROVER_SYSTEM_PROMPT},
        {"role": "user", "content": template},
    ]

    t0 = time.monotonic()
    last_error: str | None = None
    last_kind: str | None = None
    orig_temp = getattr(client, "temperature", 0.0)

    try:
        for attempt in range(1, n_attempts + 1):
            if attempt > 1 and hasattr(client, "temperature"):
                client.temperature = sample_temperature
            try:
                raw = client.chat(messages)
            except ModelClientError as exc:
                last_error = str(exc)
                last_kind = "client_error"
                continue

            candidate = _strip_lean_fences(raw)
            reason = validate_prover_source(template, candidate)
            if reason is not None:
                last_error = reason
                last_kind = _classify_validation_error(reason)
                continue

            ok, err = verify_lean_source(
                candidate, timeout=lean_timeout, project_dir=project_dir
            )
            if ok:
                return ProverResult(
                    seed=world.seed,
                    family=world.family,
                    proved=True,
                    latency_s=time.monotonic() - t0,
                    error=None,
                    attempts=attempt,
                    winning_attempt=attempt,
                    error_kind=None,
                    lean_compat=lean_compat,
                )
            last_error = err.strip()[:500]
            last_kind = "lean_compile_error"
    finally:
        if hasattr(client, "temperature"):
            client.temperature = orig_temp

    return ProverResult(
        seed=world.seed,
        family=world.family,
        proved=False,
        latency_s=time.monotonic() - t0,
        error=last_error,
        attempts=n_attempts,
        winning_attempt=None,
        error_kind=last_kind,
        lean_compat=lean_compat,
    )
