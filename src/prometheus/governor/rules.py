"""Rule checks applied to every proposed harness variant.

`check_variant` returns a list of plain-language violation reasons; an empty list
means the variant passed every structural rule. Score rules (audit non-regression
and proxy-only gains) are applied by the service, which owns the running scores.
"""

from __future__ import annotations

import ast
from typing import Any

from ..common.variant import Bundle
from .policy import Policy


def check_variant(bundle: Bundle, policy: Policy, *, expected_generation: int,
                  expected_parent: str) -> list[str]:
    m = bundle.manifest
    reasons: list[str] = []

    if m.get("run_id") != policy.run_id:
        reasons.append(f"wrong run: {m.get('run_id')!r}")
    gen = m.get("generation")
    if isinstance(gen, bool) or not isinstance(gen, int) or gen != expected_generation:
        reasons.append(f"generation {gen!r} is not the next generation ({expected_generation})")
    if not isinstance(gen, bool) and isinstance(gen, int) and gen > policy.max_generations:
        reasons.append(f"generation cap of {policy.max_generations} reached")
    if m.get("parent_hash") != expected_parent:
        reasons.append("parent is not the currently installed variant")
    if m.get("model_pins") != policy.model_pins:
        reasons.append("model weights do not match the pinned hashes")

    total = 0
    for rel in sorted(m["files"]):
        if not rel.startswith(policy.allowed_prefixes):
            reasons.append(f"touches protected path {rel}")
            continue
        if not rel.endswith(policy.allowed_suffixes):
            reasons.append(f"file type not allowed: {rel}")
            continue
        size = bundle.file_path(rel).stat().st_size
        total += size
        if size > policy.max_file_bytes:
            reasons.append(f"file too large: {rel} ({size} bytes)")
            continue
        if rel.endswith(".py"):
            reasons.extend(_scan_python(rel, bundle.read_text(rel), policy))
    if total > policy.max_bundle_bytes:
        reasons.append(f"bundle too large ({total} bytes)")
    return reasons


def _scan_python(rel: str, source: str, policy: Policy) -> list[str]:
    """Static screen for network, process and self-modification primitives.

    This is defence in depth: the sandbox (no network, unprivileged user) is the
    primary control. The scan is deliberately strict; a false veto costs one
    generation, a false approval could cost the run.
    """
    try:
        tree = ast.parse(source, filename=rel)
    except SyntaxError as exc:
        return [f"{rel}: does not parse ({exc.msg})"]
    out: list[str] = []
    banned_mods = set(policy.banned_modules)
    banned_calls = set(policy.banned_calls)
    banned_attrs = set(policy.banned_attrs)
    dynamic_names = {"getattr", "setattr", "delattr", "globals", "locals", "vars", "dir", "__builtins__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in banned_mods or (top.startswith("_") and top != "__future__"):
                    out.append(f"{rel}: imports banned module {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level == 0 and (root in banned_mods or (root.startswith("_") and root != "__future__")):
                out.append(f"{rel}: imports banned module {node.module}")
            if node.level > 0:
                out.append(f"{rel}: relative import reaches outside the tool sandbox")
        elif isinstance(node, ast.Name):
            if node.id in banned_calls:
                out.append(f"{rel}: references banned builtin {node.id}")
            if node.id in dynamic_names:
                out.append(f"{rel}: uses dynamic attribute access ({node.id})")
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in banned_calls:
                out.append(f"{rel}: calls {fn.id}()")
            if isinstance(fn, ast.Attribute) and fn.attr in banned_attrs:
                out.append(f"{rel}: calls .{fn.attr}()")
        elif isinstance(node, ast.Attribute):
            if node.attr in banned_attrs:
                out.append(f"{rel}: touches banned attribute {node.attr}")
            elif node.attr.startswith("__") and node.attr.endswith("__"):
                if node.attr not in ("__init__", "__name__", "__doc__"):
                    out.append(f"{rel}: touches dunder attribute {node.attr}")
    return sorted(set(out))


def summarise(reasons: list[str]) -> dict[str, Any]:
    return {"violations": reasons, "count": len(reasons)}
