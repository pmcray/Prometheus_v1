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
) -> list[dict[str, str]]:
    """Build the OpenAI message list for one decision step.

    Args:
        symbols:          The world's symbol list (opaque names).
        observations:     List of dicts from ``world.history``; each has
                          ``kind`` (``"combine"`` or ``"transform"``),
                          ``args`` (tuple of symbol names) and ``result``
                          (symbol name).
        budget_remaining: Remaining budget (informational for the model).

    Returns:
        An OpenAI message list: ``[{"role": "system", ...}, {"role": "user", ...}]``.
    """
    sym_list = ", ".join(symbols)
    n = len(symbols)

    lines = [
        f"The symbol set has {n} elements: {{{sym_list}}}.",
        f"Budget remaining: {budget_remaining:.1f} units.",
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


def parse_claims(raw: Any) -> dict[str, bool]:
    """Extract and validate the ``claims`` dict from a parsed model response.

    Args:
        raw: Parsed JSON (should be a dict with a ``"claims"`` key).

    Returns:
        A ``{law_name: bool}`` dict for all eight laws.

    Raises:
        ValueError: If the response is malformed or missing required keys.
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
            claims[name] = bool(claims_raw[name])

    if missing:
        raise ValueError(f"Model omitted claims for: {', '.join(missing)}")

    return claims
