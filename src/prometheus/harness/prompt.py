"""Prompt templates for the Hidden-Law Lab cortex model.

The cortex model is a general-purpose instruction-tuned LM (Qwen3-8B,
gpt-oss-20b, etc.) that acts as the "scientist" in each episode.

Protocol
--------
The model receives:
  - A system prompt explaining its role and the JSON output format.
  - A user message containing the world description and the experiment
    transcript (all combine / transform observations so far).

The model must reply with a JSON object::

    {
      "claims": {
        "commutative":    true,
        "associative":    false,
        ...
      },
      "reasoning": "..."   // optional, ignored by scorer
    }

This module is pure-Python and network-free.
"""

from __future__ import annotations

from typing import Any

from ..lab.laws import LAW_NAMES


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a scientist exploring an unknown finite algebraic structure.
You have access to two operations on a set of opaque symbols:
  • combine(a, b) → symbol   (a binary operation, written a ⋆ b)
  • transform(a)  → symbol   (a unary map, written f(a))

You will be shown the results of several experiments on the structure.
IMPORTANT: observations are noisy — a small fraction of experiment results may
be randomly corrupted to another symbol. Look for global consistency across
observations (and majority values when an experiment is repeated) rather than
letting a single isolated outlier falsify a law that holds everywhere else.

Your task is to decide which of the following eight laws hold:

  1. commutative   – a ⋆ b = b ⋆ a  for all a, b
  2. associative   – (a ⋆ b) ⋆ c = a ⋆ (b ⋆ c)  for all a, b, c
  3. has_identity  – some e satisfies e ⋆ a = a ⋆ e = a  for all a
  4. has_inverses  – identity exists and every element has a two-sided inverse
  5. idempotent    – a ⋆ a = a  for all a
  6. f_involution  – f(f(a)) = a  for all a
  7. f_homomorphism – f(a ⋆ b) = f(a) ⋆ f(b)  for all a, b
  8. f_bijective   – f is a bijection (one-to-one and onto)

Reply ONLY with a JSON object in this exact format (no extra keys, no markdown):
{
  "claims": {
    "commutative":    true or false,
    "associative":    true or false,
    "has_identity":   true or false,
    "has_inverses":   true or false,
    "idempotent":     true or false,
    "f_involution":   true or false,
    "f_homomorphism": true or false,
    "f_bijective":    true or false
  },
  "reasoning": "brief explanation (1-3 sentences)"
}
"""


# ---------------------------------------------------------------------------
# User message builder
# ---------------------------------------------------------------------------

def build_messages(
    symbols: list[str],
    observations: list[dict[str, Any]],
    budget_remaining: float,
    *,
    noise: float | None = None,
) -> list[dict[str, str]]:
    """Build the OpenAI message list for one decision step.

    Args:
        symbols:          The world's symbol list (opaque names).
        observations:     List of dicts from ``world.history``; each has
                          ``kind`` (``"combine"`` or ``"transform"``),
                          ``args`` (tuple of symbol names) and ``result``
                          (symbol name).
        budget_remaining: Remaining budget (informational for the model).
        noise:            Optional observation noise rate in [0, 1).

    Returns:
        An OpenAI message list: ``[{"role": "system", ...}, {"role": "user", ...}]``.
    """
    sym_list = ", ".join(symbols)
    n = len(symbols)

    lines = [
        f"The symbol set has {n} elements: {{{sym_list}}}.",
        f"Budget remaining: {budget_remaining:.1f} units.",
    ]
    if noise is not None:
        lines.append(f"Observation noise rate: {noise * 100:.1f}% per query.")
    lines += [
        "",
        f"Experiment transcript ({len(observations)} observations):",
    ]

    if not observations:
        lines.append("  (no experiments run yet)")
    else:
        for obs in observations:
            if obs["kind"] == "combine":
                a, b = obs["args"]
                lines.append(f"  combine({a}, {b}) = {obs['result']}")
            else:
                (a,) = obs["args"]
                lines.append(f"  transform({a}) = {obs['result']}")

    lines += [
        "",
        "Based on these observations, state your claims about the eight laws.",
        "Remember: reply with ONLY the JSON object, no other text.",
    ]

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


def _parse_bool(value: Any, law_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low == "true":
            return True
        if low == "false":
            return False
    raise ValueError(f"Invalid boolean claim for {law_name!r}: {value!r}")


def parse_claims(raw: Any) -> dict[str, bool]:
    """Extract and validate the ``claims`` dict from a parsed model response.

    Args:
        raw: Parsed JSON (should be a dict with a ``"claims"`` key).

    Returns:
        A ``{law_name: bool}`` dict for all eight laws.

    Raises:
        ValueError: If the response is malformed, missing required keys, or
                    contains non-boolean values.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object, got {type(raw).__name__}")
    claims_raw = raw.get("claims")
    if not isinstance(claims_raw, dict):
        raise ValueError("Missing or non-dict 'claims' key in model response")

    claims: dict[str, bool] = {}
    missing = []
    for name in LAW_NAMES:
        if name not in claims_raw:
            missing.append(name)
        else:
            claims[name] = _parse_bool(claims_raw[name], name)

    if missing:
        raise ValueError(f"Model omitted claims for: {', '.join(missing)}")

    return claims


# ---------------------------------------------------------------------------
# Phase 1 hybrid advisor prompt & parser
# ---------------------------------------------------------------------------

ADVISOR_SYSTEM_PROMPT = """\
You are the cortex advisor assisting an active-inference Bayesian scientist
exploring an unknown finite algebraic structure with noisy observations.

You will be given:
  - The symbol set and remaining query budget.
  - Current posterior probabilities P(law=True) for all eight laws.
  - Top candidate next experiments ranked by Expected Free Energy (lower is better).
  - Recent experiment observations and activated structural memory assemblies.

Reply ONLY with a JSON object in this format (no markdown, no extra text):
{
  "next_experiment": {"kind": "combine", "args": ["a", "b"]} or {"kind": "transform", "args": ["a"]} or null,
  "claims": {
    "commutative": true or false,
    "associative": true or false,
    "has_identity": true or false,
    "has_inverses": true or false,
    "idempotent": true or false,
    "f_involution": true or false,
    "f_homomorphism": true or false,
    "f_bijective": true or false
  },
  "reasoning": "brief 1-2 sentence scientific note"
}
"""


def build_advisor_messages(
    *,
    symbols: list[str],
    law_probs: dict[str, float],
    observations: list[dict[str, Any]],
    top_candidates: list[dict[str, Any]],
    memory_activations: dict[str, float] | None = None,
    budget_remaining: float,
    noise: float | None = None,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    """Build OpenAI messages for the optional Phase 1 Cortex LLM advisor."""
    lines = [
        f"Symbols ({len(symbols)}): {{{', '.join(symbols)}}}",
        f"Budget remaining: {budget_remaining:.1f}",
    ]
    if noise is not None:
        lines.append(f"Observation noise rate: {noise * 100:.1f}%")
    lines.append("")
    lines.append("Current Bayesian law posteriors:")
    for name in LAW_NAMES:
        lines.append(f"  {name:16s}: {law_probs.get(name, 0.5):.3f}")

    if memory_activations:
        top_mem = sorted(memory_activations.items(), key=lambda kv: -kv[1])[:4]
        mem_str = ", ".join(f"{k}={v:.2f}" for k, v in top_mem)
        lines += ["", f"Activated memory assemblies: {mem_str}"]

    if top_candidates:
        lines += ["", "Top EFE-ranked candidate experiments (lowest EFE first):"]
        for c in top_candidates:
            args_str = ", ".join(c["args"])
            lines.append(
                f"  {c['kind']}({args_str})  cost={c['cost']}  "
                f"EFE={c['efe']}  law_EIG={c['law_eig_bits']} bits"
            )

    lines += ["", f"Recent observations ({len(observations)} shown):"]
    if not observations:
        lines.append("  (none yet)")
    else:
        for obs in observations:
            if obs["kind"] == "combine":
                a, b = obs["args"]
                lines.append(f"  combine({a}, {b}) = {obs['result']}")
            else:
                (a,) = obs["args"]
                lines.append(f"  transform({a}) = {obs['result']}")

    lines += ["", "Reply ONLY with the JSON object."]
    return [
        {"role": "system", "content": system_prompt or ADVISOR_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


def parse_advisor_reply(
    raw: Any,
) -> tuple[dict[str, Any] | None, dict[str, bool] | None, str | None]:
    """Parse optional ``next_experiment``, ``claims``, and ``reasoning`` from advisor JSON."""
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object from advisor, got {type(raw).__name__}")

    next_exp: dict[str, Any] | None = None
    raw_exp = raw.get("next_experiment")
    if isinstance(raw_exp, dict):
        kind = raw_exp.get("kind")
        args = raw_exp.get("args")
        if kind in ("combine", "transform") and isinstance(args, list) and all(
            isinstance(a, str) for a in args
        ):
            next_exp = {"kind": kind, "args": args}

    claims: dict[str, bool] | None = None
    if isinstance(raw.get("claims"), dict):
        claims = parse_claims(raw)

    note = raw.get("reasoning")
    reasoning = str(note).strip() if isinstance(note, str) and note.strip() else None
    return (next_exp, claims, reasoning)
