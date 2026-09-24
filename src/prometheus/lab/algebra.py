"""Planted finite structures for the Hidden-Law Lab.

A world is a finite carrier {0..n-1} with a binary operation `op` (an n x n table)
and a unary map `f` (a length-n list). Structures are drawn from known families so
that laws hold or fail for a reason (group, semilattice, rock-paper-scissors...),
not only by chance. Symbols are relabelled and renamed before the agent sees them.
"""

from __future__ import annotations

import itertools
import random
from typing import Callable

Table = list[list[int]]
Map = list[int]


def cyclic(n: int, rng: random.Random) -> Table:
    return [[(i + j) % n for j in range(n)] for i in range(n)]


def klein4(n: int, rng: random.Random) -> Table:
    return [[i ^ j for j in range(4)] for i in range(4)]


def symmetric3(n: int, rng: random.Random) -> Table:
    perms = list(itertools.permutations(range(3)))
    index = {p: k for k, p in enumerate(perms)}

    def compose(p, q):  # (p o q)(x) = p(q(x))
        return tuple(p[q[x]] for x in range(3))

    return [[index[compose(p, q)] for q in perms] for p in perms]


def max_semilattice(n: int, rng: random.Random) -> Table:
    return [[max(i, j) for j in range(n)] for i in range(n)]


def left_zero(n: int, rng: random.Random) -> Table:
    return [[i for _ in range(n)] for i in range(n)]


def rock_paper_scissors(n: int, rng: random.Random) -> Table:
    def wins(i, j):
        return (i - j) % 3 == 1

    return [[i if (i == j or wins(i, j)) else j for j in range(3)] for i in range(3)]


def mult_mod(n: int, rng: random.Random) -> Table:
    return [[(i * j) % n for j in range(n)] for i in range(n)]


def random_magma(n: int, rng: random.Random) -> Table:
    return [[rng.randrange(n) for _ in range(n)] for _ in range(n)]


# name -> (constructor, allowed sizes)
FAMILIES: dict[str, tuple[Callable[[int, random.Random], Table], tuple[int, ...]]] = {
    "cyclic": (cyclic, (3, 4, 5, 6, 7)),
    "klein4": (klein4, (4,)),
    "symmetric3": (symmetric3, (6,)),
    "max_semilattice": (max_semilattice, (3, 4, 5, 6)),
    "left_zero": (left_zero, (3, 4, 5)),
    "rock_paper_scissors": (rock_paper_scissors, (3,)),
    "mult_mod": (mult_mod, (4, 5, 6, 7)),
    "random_magma": (random_magma, (3, 4, 5)),
}


def identity_element(op: Table) -> int | None:
    n = len(op)
    for e in range(n):
        if all(op[e][x] == x and op[x][e] == x for x in range(n)):
            return e
    return None


def f_identity(op: Table, rng: random.Random) -> Map:
    return list(range(len(op)))


def f_inverse(op: Table, rng: random.Random) -> Map | None:
    n, e = len(op), identity_element(op)
    if e is None:
        return None
    inv = []
    for x in range(n):
        ys = [y for y in range(n) if op[x][y] == e and op[y][x] == e]
        if not ys:
            return None
        inv.append(ys[0])
    return inv


def f_shift(op: Table, rng: random.Random) -> Map:
    n = len(op)
    return [(x + 1) % n for x in range(n)]


def f_random_perm(op: Table, rng: random.Random) -> Map:
    perm = list(range(len(op)))
    rng.shuffle(perm)
    return perm


def f_random_function(op: Table, rng: random.Random) -> Map:
    n = len(op)
    return [rng.randrange(n) for _ in range(n)]


MAPS: dict[str, Callable[[Table, random.Random], Map | None]] = {
    "identity": f_identity,
    "inverse": f_inverse,
    "shift": f_shift,
    "random_perm": f_random_perm,
    "random_function": f_random_function,
}


def relabel(op: Table, f: Map, perm: list[int]) -> tuple[Table, Map]:
    """Apply the bijection perm: old symbol x becomes perm[x]."""
    n = len(op)
    inv = [0] * n
    for old, new in enumerate(perm):
        inv[new] = old
    new_op = [[perm[op[inv[a]][inv[b]]] for b in range(n)] for a in range(n)]
    new_f = [perm[f[inv[a]]] for a in range(n)]
    return new_op, new_f
