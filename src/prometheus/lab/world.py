"""A Hidden-Law Lab world: the only environment Prometheus v1 acts in.

The agent sees opaque symbol names and can run two kinds of experiment, each with a
cost and adjustable observation noise:

    combine(a, b) -> symbol     observe a ⋆ b
    transform(a)  -> symbol     observe f(a)

Worlds are fully determined by their seed, so the governor can rebuild any audit
world on the workstation (CPU only) and score claims against ground truth without
trusting the Colab runtime. `shift_laws` implements the Phase 3 "law shift"
perturbation: the same symbols, a different hidden structure.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .algebra import FAMILIES, MAPS, relabel
from .laws import evaluate_laws

SYLLABLES = ("ka", "lu", "mo", "ne", "pi", "ro", "su", "ta", "vi", "zo", "qe", "xu")


class BudgetExhausted(RuntimeError):
    pass


@dataclass
class ExperimentRecord:
    kind: str
    args: tuple[str, ...]
    result: str
    cost: float


@dataclass
class HiddenLawWorld:
    seed: int
    noise: float = 0.05
    budget: float = 500.0
    combine_cost: float = 1.0
    transform_cost: float = 0.5
    family: str = field(init=False)
    map_kind: str = field(init=False)
    symbols: list[str] = field(init=False)
    spent: float = field(init=False, default=0.0)
    history: list[ExperimentRecord] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 <= self.noise < 1.0:
            raise ValueError("noise must be in [0, 1)")
        self._build(random.Random(f"world:{self.seed}"))
        self._noise_rng = random.Random(f"noise:{self.seed}")

    def _build(self, rng: random.Random, *, avoid_family: str | None = None) -> None:
        families = sorted(k for k in FAMILIES if k != avoid_family)
        for _attempt in range(10_000):
            family = rng.choice(families)
            ctor, sizes = FAMILIES[family]
            n = rng.choice(sizes)
            if avoid_family is not None and n != len(self.symbols):
                continue
            op = ctor(n, rng)
            map_kind = rng.choice(sorted(MAPS))
            f = MAPS[map_kind](op, rng)
            if f is not None:
                break
        else:
            raise RuntimeError("could not build a world with these constraints")
        perm = list(range(n))
        rng.shuffle(perm)
        self._op, self._f = relabel(op, f, perm)
        self.family, self.map_kind = family, map_kind
        if avoid_family is None:
            names = rng.sample(SYLLABLES, n)
            self.symbols = names
        self._index = {s: i for i, s in enumerate(self.symbols)}

    # ----- experiments (the agent's only access) -----------------------------

    def _charge(self, cost: float) -> None:
        if self.spent + cost > self.budget + 1e-9:
            raise BudgetExhausted(f"budget {self.budget} exhausted")
        self.spent += cost

    def _observe(self, true_idx: int) -> str:
        if self._noise_rng.random() < self.noise:
            return self._noise_rng.choice(self.symbols)
        return self.symbols[true_idx]

    def combine(self, a: str, b: str) -> str:
        self._charge(self.combine_cost)
        out = self._observe(self._op[self._index[a]][self._index[b]])
        self.history.append(ExperimentRecord("combine", (a, b), out, self.combine_cost))
        return out

    def transform(self, a: str) -> str:
        self._charge(self.transform_cost)
        out = self._observe(self._f[self._index[a]])
        self.history.append(ExperimentRecord("transform", (a,), out, self.transform_cost))
        return out

    @property
    def remaining(self) -> float:
        return self.budget - self.spent

    # ----- ground truth (governor, evaluation and tests only) ----------------

    def ground_truth(self) -> dict[str, bool]:
        return evaluate_laws(self._op, self._f)

    def true_tables(self) -> tuple[dict[tuple[str, str], str], dict[str, str]]:
        s = self.symbols
        op = {(s[i], s[j]): s[self._op[i][j]] for i in range(len(s)) for j in range(len(s))}
        f = {s[i]: s[self._f[i]] for i in range(len(s))}
        return op, f

    def shift_laws(self, shift_seed: int) -> None:
        """Perturbation: replace the hidden structure, keeping the same symbols."""
        self._build(random.Random(f"shift:{self.seed}:{shift_seed}"), avoid_family=self.family)
