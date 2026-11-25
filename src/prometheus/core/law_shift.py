"""Phase 3 Law-Shift Stress Benchmarks & Non-Stationary Ashby Recovery Loop.

Evaluates how Prometheus v1 handles mid-run algebraic law shifts
(:meth:`HiddenLawWorld.shift_laws`) strictly across the Markov blanket:

1. **Pre-shift phase:** The agent discovers the initial world's laws.
2. **External perturbation:** The benchmark harness invokes
   ``world.shift_laws(shift_seed)`` outside the blanket (same opaque symbols,
   different algebraic family).
3. **Post-shift continuation:** Without resetting backend beliefs or informing
   the agent of the shift, :meth:`CognitiveAgent.continue_episode` monitors
   the world and attempts recovery.
4. **Static vs. Ashby-adapted comparison:**
   - A **static** agent (``forgetting_factor=1.0``) suffers high Bayesian
     surprise and stale-count lock-in, tripping :class:`HomeostaticMonitor`
     with ``HIGH_SURPRISE``.
   - :class:`AshbyProposer` responds to that ``HIGH_SURPRISE`` diagnosis by
     synthesising a validated ``forgetting_adapt`` harness
     (``forgetting_factor <= 0.65``, softened prior blend, tighter surprise
     veto), which detects the surprise spike, globally decays stale cell
     beliefs, retracts stale analogy priors, and recovers post-shift accuracy.
"""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from ..common.eventlog import EventLog
from ..lab.laws import LAW_NAMES
from ..lab.world import HiddenLawWorld
from .agent import CognitiveAgent
from .ashby import (
    AshbyConfig,
    AshbyProposer,
    HomeostaticMonitor,
    StressDiagnosis,
    _validate_candidate_files,
)
from .harness_loader import LoadedHarness
from .memory import AssemblyMemory
from .transfer import StructuralAnalogyTransfer


@dataclass(frozen=True)
class LawShiftEpisodeResult:
    """Outcome of a single pre-shift -> shift_laws() -> post-shift recovery episode."""

    seed: int
    shift_seed: int
    n: int
    pre_family: str
    post_family: str
    pre_accuracy: float
    post_accuracy: float
    pre_observations: int
    post_observations: int
    pre_budget_spent: float
    post_budget_spent: float
    mean_post_surprise: float
    max_post_surprise: float
    surprise_detected: bool
    transfer_vetoes: int
    ashby_triggered: bool
    post_stop_reason: str
    latency_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LawShiftBenchmarkSummary:
    """Aggregate comparison of static vs. Ashby-adapted agents under mid-run law shifts."""

    n_worlds: int
    noise: float
    budget: float
    static_pre_accuracy: float
    static_post_accuracy: float
    static_mean_max_surprise: float
    static_ashby_triggers: int
    adapted_label: str
    adapted_forgetting_factor: float
    adapted_pre_accuracy: float
    adapted_post_accuracy: float
    adapted_recovery_gain: float
    adapted_mean_post_budget: float
    adapted_surprise_detection_rate: float
    adapted_transfer_vetoes: int
    diagnosis_reasons: tuple[str, ...]
    static_results: tuple[LawShiftEpisodeResult, ...]
    adapted_results: tuple[LawShiftEpisodeResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_worlds": self.n_worlds,
            "noise": self.noise,
            "budget": self.budget,
            "static": {
                "pre_accuracy": self.static_pre_accuracy,
                "post_accuracy": self.static_post_accuracy,
                "mean_max_surprise": self.static_mean_max_surprise,
                "ashby_triggers": self.static_ashby_triggers,
            },
            "adapted": {
                "label": self.adapted_label,
                "forgetting_factor": self.adapted_forgetting_factor,
                "pre_accuracy": self.adapted_pre_accuracy,
                "post_accuracy": self.adapted_post_accuracy,
                "recovery_gain": self.adapted_recovery_gain,
                "mean_post_budget": self.adapted_mean_post_budget,
                "surprise_detection_rate": self.adapted_surprise_detection_rate,
                "transfer_vetoes": self.adapted_transfer_vetoes,
            },
            "diagnosis_reasons": list(self.diagnosis_reasons),
            "static_results": [r.to_dict() for r in self.static_results],
            "adapted_results": [r.to_dict() for r in self.adapted_results],
        }


def _law_acc(claims: dict[str, bool], truth: dict[str, bool]) -> float:
    return sum(1 for k in LAW_NAMES if claims.get(k) == truth.get(k)) / len(LAW_NAMES)


def run_law_shift_episode(
    seed: int,
    *,
    shift_seed: int = 777,
    harness: LoadedHarness | None = None,
    memory: AssemblyMemory | None = None,
    monitor: HomeostaticMonitor | None = None,
    event_log: EventLog | None = None,
    noise: float = 0.05,
    budget: float = 500.0,
    sentinel_probes: int = 8,
    use_transfer: bool = True,
    rng_seed: int = 0,
    episode_idx: int = 0,
    mode_label: str = "agent",
) -> LawShiftEpisodeResult:
    """Run one pre-shift -> ``world.shift_laws()`` -> post-shift recovery episode."""
    t0 = time.monotonic()
    har = harness or LoadedHarness()
    mem = memory if memory is not None else AssemblyMemory(har.memory_params)
    transfer = StructuralAnalogyTransfer(har.transfer_params) if use_transfer else None

    world = HiddenLawWorld(seed=seed, noise=noise, budget=budget)
    pre_family = world.family
    pre_truth = world.ground_truth()

    agent = CognitiveAgent(
        harness=har,
        memory=mem,
        transfer=transfer,
        monitor=None,  # monitor records the post-shift stress continuation
        rng_seed=rng_seed + episode_idx,
    )
    pre_res = agent.run_episode(world, learn=False, episode_idx=episode_idx)
    pre_acc = _law_acc(pre_res.claims, pre_truth)

    # External perturbation outside the agent's Markov blanket
    world.shift_laws(shift_seed=shift_seed)
    post_family = world.family
    post_truth = world.ground_truth()

    vetoes_before = transfer.surprise_vetoes if transfer is not None else 0
    decays_before = getattr(agent.backend, "shift_decay_count", 0)
    agent.monitor = monitor
    post_res = agent.continue_episode(
        world,
        sentinel_probes=sentinel_probes,
        learn=False,
        episode_idx=episode_idx,
    )
    post_acc = _law_acc(post_res.claims, post_truth)
    vetoes_delta = (transfer.surprise_vetoes - vetoes_before) if transfer is not None else 0
    decays_delta = getattr(agent.backend, "shift_decay_count", 0) - decays_before

    detected = (
        decays_delta > 0
        or post_res.metrics.max_surprise > 3.5
        or post_res.metrics.mean_surprise > 2.0
    )

    result = LawShiftEpisodeResult(
        seed=seed,
        shift_seed=shift_seed,
        n=len(world.symbols),
        pre_family=pre_family,
        post_family=post_family,
        pre_accuracy=round(pre_acc, 4),
        post_accuracy=round(post_acc, 4),
        pre_observations=pre_res.observations_used,
        post_observations=post_res.observations_used,
        pre_budget_spent=round(pre_res.budget_spent, 2),
        post_budget_spent=round(post_res.budget_spent, 2),
        mean_post_surprise=round(post_res.metrics.mean_surprise, 4),
        max_post_surprise=round(post_res.metrics.max_surprise, 4),
        surprise_detected=detected,
        transfer_vetoes=vetoes_delta,
        ashby_triggered=post_res.ashby_triggered,
        post_stop_reason=post_res.stop_reason,
        latency_s=round(time.monotonic() - t0, 3),
    )

    if event_log is not None:
        event_log.append(
            "law_shift_episode",
            {
                "mode": mode_label,
                "seed": seed,
                "shift_seed": shift_seed,
                "pre_family": pre_family,
                "post_family": post_family,
                "pre_accuracy": result.pre_accuracy,
                "post_accuracy": result.post_accuracy,
                "max_post_surprise": result.max_post_surprise,
                "mean_post_surprise": result.mean_post_surprise,
                "surprise_detected": result.surprise_detected,
                "transfer_vetoes": result.transfer_vetoes,
                "ashby_triggered": result.ashby_triggered,
                "post_budget_spent": result.post_budget_spent,
            },
        )
    return result


def select_ashby_forgetting_harness(
    base_harness: LoadedHarness,
    diagnosis: StressDiagnosis | None = None,
) -> tuple[str, LoadedHarness]:
    """Use :class:`AshbyProposer` to synthesise a validated ``forgetting_adapt`` harness."""
    diag = diagnosis
    if diag is None or "HIGH_SURPRISE" not in diag.reasons:
        diag = StressDiagnosis(
            triggered=True,
            reasons=("HIGH_SURPRISE",),
            violations=("law-shift surprise spike",),
        )
    proposer = AshbyProposer()
    cands = dict(proposer.generate_candidates(base_harness, diag))
    if "forgetting_adapt" not in cands:
        raise RuntimeError("AshbyProposer did not generate forgetting_adapt for HIGH_SURPRISE")
    loaded = _validate_candidate_files(cands["forgetting_adapt"])
    if loaded is None:
        raise RuntimeError("AshbyProposer forgetting_adapt candidate failed validation")
    return ("forgetting_adapt", loaded)


def run_law_shift_benchmark(
    seeds: Sequence[int],
    *,
    shift_seed: int = 777,
    base_harness: LoadedHarness | None = None,
    memory: AssemblyMemory | None = None,
    noise: float = 0.05,
    budget: float = 500.0,
    sentinel_probes: int = 8,
    use_transfer: bool = True,
    event_log: EventLog | None = None,
    checkpoint_path: Path | str | None = None,
    rng_seed: int = 0,
    on_episode: Any = None,
) -> LawShiftBenchmarkSummary:
    """Run paired static vs. Ashby-adapted law-shift recovery benchmarks across *seeds*."""
    seed_list = list(seeds)
    if not seed_list:
        raise ValueError("seeds must be non-empty")

    har_static = base_harness or LoadedHarness()
    mem_base = copy.deepcopy(memory) if memory is not None else AssemblyMemory(har_static.memory_params)

    # 1. Run static baseline under HomeostaticMonitor so real HIGH_SURPRISE stress is diagnosed
    mon_static = HomeostaticMonitor(
        AshbyConfig(
            window_size=min(8, max(2, len(seed_list))),
            min_episodes_before_trigger=min(2, len(seed_list)),
            min_breach_episodes=min(2, len(seed_list)),
            cooldown_episodes=2,
            surprise_spike_threshold=har_static.ashby_config.surprise_spike_threshold,
        ),
        event_log=event_log,
    )
    static_results: list[LawShiftEpisodeResult] = []
    for idx, s in enumerate(seed_list):
        r = run_law_shift_episode(
            s,
            shift_seed=shift_seed,
            harness=har_static,
            memory=copy.deepcopy(mem_base),
            monitor=mon_static,
            event_log=event_log,
            noise=noise,
            budget=budget,
            sentinel_probes=sentinel_probes,
            use_transfer=use_transfer,
            rng_seed=rng_seed,
            episode_idx=idx,
            mode_label="static",
        )
        static_results.append(r)
        if on_episode is not None:
            on_episode("static", idx, r)

    # 2. Let AshbyProposer synthesise the validated forgetting_adapt harness from the stress diagnosis
    diagnosis = mon_static.last_diagnosis
    adapted_label, har_adapted = select_ashby_forgetting_harness(har_static, diagnosis)

    # 3. Evaluate the Ashby-adapted harness on the exact same law-shift worlds
    mon_adapted = HomeostaticMonitor(har_adapted.ashby_config)
    adapted_results: list[LawShiftEpisodeResult] = []
    for idx, s in enumerate(seed_list):
        r = run_law_shift_episode(
            s,
            shift_seed=shift_seed,
            harness=har_adapted,
            memory=copy.deepcopy(mem_base),
            monitor=mon_adapted,
            event_log=event_log,
            noise=noise,
            budget=budget,
            sentinel_probes=sentinel_probes,
            use_transfer=use_transfer,
            rng_seed=rng_seed,
            episode_idx=idx,
            mode_label=adapted_label,
        )
        adapted_results.append(r)
        if on_episode is not None:
            on_episode(adapted_label, idx, r)

    n = len(seed_list)
    st_pre = sum(r.pre_accuracy for r in static_results) / n
    st_post = sum(r.post_accuracy for r in static_results) / n
    st_max_surp = sum(r.max_post_surprise for r in static_results) / n

    ad_pre = sum(r.pre_accuracy for r in adapted_results) / n
    ad_post = sum(r.post_accuracy for r in adapted_results) / n
    ad_post_budget = sum(r.post_budget_spent for r in adapted_results) / n
    ad_det = sum(1 for r in adapted_results if r.surprise_detected) / n
    ad_vetoes = sum(r.transfer_vetoes for r in adapted_results)

    summary = LawShiftBenchmarkSummary(
        n_worlds=n,
        noise=noise,
        budget=budget,
        static_pre_accuracy=round(st_pre, 4),
        static_post_accuracy=round(st_post, 4),
        static_mean_max_surprise=round(st_max_surp, 4),
        static_ashby_triggers=mon_static.trigger_count,
        adapted_label=adapted_label,
        adapted_forgetting_factor=har_adapted.efe_params.forgetting_factor,
        adapted_pre_accuracy=round(ad_pre, 4),
        adapted_post_accuracy=round(ad_post, 4),
        adapted_recovery_gain=round(ad_post - st_post, 4),
        adapted_mean_post_budget=round(ad_post_budget, 2),
        adapted_surprise_detection_rate=round(ad_det, 4),
        adapted_transfer_vetoes=ad_vetoes,
        diagnosis_reasons=diagnosis.reasons or ("HIGH_SURPRISE",),
        static_results=tuple(static_results),
        adapted_results=tuple(adapted_results),
    )

    if checkpoint_path is not None:
        cp = Path(checkpoint_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        tmp = cp.with_name(f".{cp.name}.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(summary.to_dict(), fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, cp)

    return summary
