"""The catalogue of hidden laws, each decided exactly by brute force.

Every law is a first-order statement over a finite carrier, so each can later be
stated in Lean 4 and proved with `decide`. The statements are given in plain
English and in a Lean-style sketch for the Formalizer role.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .algebra import Map, Table, identity_element


@dataclass(frozen=True)
class Law:
    name: str
    english: str
    lean_sketch: str
    holds: Callable[[Table, Map], bool]


def _commutative(op: Table, f: Map) -> bool:
    n = len(op)
    return all(op[a][b] == op[b][a] for a in range(n) for b in range(n))


def _associative(op: Table, f: Map) -> bool:
    n = len(op)
    return all(
        op[op[a][b]][c] == op[a][op[b][c]] for a in range(n) for b in range(n) for c in range(n)
    )


def _has_identity(op: Table, f: Map) -> bool:
    return identity_element(op) is not None


def _has_inverses(op: Table, f: Map) -> bool:
    n, e = len(op), identity_element(op)
    if e is None:
        return False
    return all(any(op[a][b] == e and op[b][a] == e for b in range(n)) for a in range(n))


def _idempotent(op: Table, f: Map) -> bool:
    return all(op[a][a] == a for a in range(len(op)))


def _f_involution(op: Table, f: Map) -> bool:
    return all(f[f[a]] == a for a in range(len(f)))


def _f_homomorphism(op: Table, f: Map) -> bool:
    n = len(op)
    return all(f[op[a][b]] == op[f[a]][f[b]] for a in range(n) for b in range(n))


def _f_bijective(op: Table, f: Map) -> bool:
    return sorted(f) == list(range(len(f)))


LAWS: tuple[Law, ...] = (
    Law("commutative", "combining a with b gives the same as b with a",
        "∀ a b : S, a ⋆ b = b ⋆ a", _commutative),
    Law("associative", "grouping does not matter when combining three things",
        "∀ a b c : S, (a ⋆ b) ⋆ c = a ⋆ (b ⋆ c)", _associative),
    Law("has_identity", "some element leaves everything unchanged when combined",
        "∃ e : S, ∀ a : S, e ⋆ a = a ∧ a ⋆ e = a", _has_identity),
    Law("has_inverses", "there is an identity and every element can be undone",
        "∃ e : S, (∀ a, e ⋆ a = a ∧ a ⋆ e = a) ∧ ∀ a, ∃ b, a ⋆ b = e ∧ b ⋆ a = e", _has_inverses),
    Law("idempotent", "combining anything with itself gives itself",
        "∀ a : S, a ⋆ a = a", _idempotent),
    Law("f_involution", "applying the transform twice returns the start",
        "∀ a : S, f (f a) = a", _f_involution),
    Law("f_homomorphism", "the transform respects combination",
        "∀ a b : S, f (a ⋆ b) = f a ⋆ f b", _f_homomorphism),
    Law("f_bijective", "the transform is a one-to-one reshuffling",
        "Function.Bijective f", _f_bijective),
)

LAW_NAMES: tuple[str, ...] = tuple(law.name for law in LAWS)


def evaluate_laws(op: Table, f: Map) -> dict[str, bool]:
    return {law.name: law.holds(op, f) for law in LAWS}
