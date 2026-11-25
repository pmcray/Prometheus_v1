"""Phase 3 Multi-Generation Pinned Self-Improvement Runner (``run-002``).

Orchestrates sustained, resumable multi-generation self-improvement strictly
under pinned ``model_pins`` and external workstation Governor supervision:

1. Enforces non-empty pinned ``model_pins`` (e.g.
   ``{"cortex": "qwen3-8b@b968826d9c46dd6066d109eabc6255188de91218"}``).
2. Runs fresh, strictly disjoint development seed windows per generation
   (never overlapping ``audit_seeds``), alternating tight-budget and law-shift
   stress probes so Ashby adaptation exercises multiple homeostasis channels.
3. Screens candidates via :class:`AshbyProposer`, evaluates ``audit_seeds``
   strictly read-only, publishes content-hashed bundles + ``.ready`` sentinel.
4. Polls Drive for the external Governor's Ed25519-signed decision while
   verifying heartbeats at every tick (honouring kill-switch halts immediately).
5. On approval, atomically installs via :meth:`Gate.install` and reloads the
   new harness for the next generation; on veto, records the veto cleanly and
   advances the variant suffix without weakening any rule.
6. Persists full lineage to ``multigen_checkpoint.json`` so Colab forced
   restarts resume mid-run cleanly.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from ..bakeoff.models import MODEL_REGISTRY
from ..common.eventlog import EventLog
from ..common.variant import Bundle, write_bundle
from ..governor.service import GENESIS_PARENT
from ..harness.client import ModelClient
from ..lab.world import HiddenLawWorld
from ..runtime.gate import Gate, InstallRefused
from .agent import CognitiveAgent
from .ashby import AshbyProposer, HomeostaticMonitor, StressDiagnosis
from .harness_loader import LoadedHarness, load_harness
from .law_shift import run_law_shift_episode
from .memory import AssemblyMemory
from .transfer import StructuralAnalogyTransfer

_PIN_HASH_RE = re.compile(r"^(?:[A-Za-z0-9._-]+@)?[0-9a-f]{40}$")

DEFAULT_RUN002_MODEL_PINS: dict[str, str] = {
    "cortex": f"qwen3-8b@{MODEL_REGISTRY['qwen3-8b'].revision}",
}


def validate_model_pins(model_pins: dict[str, str]) -> dict[str, str]:
    """Validate that *model_pins* is non-empty and every value carries a 40-hex commit pin."""
    if not isinstance(model_pins, dict) or not model_pins:
        raise ValueError("pinned multi-generation run requires non-empty model_pins")
    cleaned: dict[str, str] = {}
    for role, pin in model_pins.items():
        if not isinstance(role, str) or not role.strip():
            raise ValueError(f"invalid model_pins role: {role!r}")
        if not isinstance(pin, str) or not _PIN_HASH_RE.match(pin.strip()):
            raise ValueError(
                f"model_pins[{role!r}]={pin!r} must be a 40-hex HF commit hash or '<model>@<40-hex>'"
            )
        cleaned[role.strip()] = pin.strip()
    return cleaned


def build_pinned_baseline_bundle(
    proposal_dir: Path | str,
    *,
    run_id: str,
    model_pins: dict[str, str],
    audit_seeds: Sequence[int],
    harness: LoadedHarness | None = None,
    noise: float = 0.05,
    budget: float = 500.0,
    evaluate_audit: bool = True,
    initial_claims: dict[str, dict[str, bool]] | None = None,
) -> Bundle:
    """Create a valid Gen-0 ``__baseline__`` bundle with pinned ``model_pins`` for ``run-002``.

    Intended for initialising a fresh pinned run on the workstation Governor
    (``prometheus-governor init --baseline <proposal_dir>``).
    """
    pins = validate_model_pins(model_pins)
    har = harness or LoadedHarness()
    files = {
        "harness/memory_params/efe_params.json": json.dumps(
            dataclasses.asdict(har.efe_params), indent=2, sort_keys=True
        ) + "\n",
        "harness/memory_params/memory_params.json": json.dumps(
            dataclasses.asdict(har.memory_params), indent=2, sort_keys=True
        ) + "\n",
        "harness/memory_params/transfer_params.json": json.dumps(
            dataclasses.asdict(har.transfer_params), indent=2, sort_keys=True
        ) + "\n",
        "harness/mesh/config.json": json.dumps(
            dataclasses.asdict(har.mesh), indent=2, sort_keys=True
        ) + "\n",
        "harness/mesh/ashby_params.json": json.dumps(
            dataclasses.asdict(har.ashby_config), indent=2, sort_keys=True
        ) + "\n",
    }
    pdir = Path(proposal_dir)
    bundle = write_bundle(
        pdir,
        {
            "run_id": run_id,
            "generation": 0,
            "parent_hash": GENESIS_PARENT,
            "model_pins": pins,
            "created_at": time.time(),
        },
        files,
    )
    if initial_claims is not None:
        claims = initial_claims
    elif evaluate_audit:
        agent = CognitiveAgent(harness=har, memory=AssemblyMemory(har.memory_params), rng_seed=0)
        claims = agent.evaluate_audit_seeds(audit_seeds, noise=noise, budget=budget)
    else:
        raise ValueError("either evaluate_audit=True or initial_claims must be provided")
    (pdir / "audit_claims.json").write_text(
        json.dumps(claims, indent=2, sort_keys=True), encoding="utf-8"
    )
    return bundle


@dataclass
class MultiGenConfig:
    """Configuration for a pinned multi-generation self-improvement run."""

    run_id: str = "run-002"
    model_pins: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_RUN002_MODEL_PINS))
    target_generations: int = 3
    max_generations: int = 10
    audit_seeds: list[int] = field(default_factory=lambda: list(range(1000, 1050)))
    dev_seed_start: int = 300
    dev_window_size: int = 12
    dev_budget: float = 25.0
    audit_budget: float = 500.0
    noise: float = 0.05
    stress_modes: tuple[str, ...] = ("tight_budget", "law_shift", "tight_budget")
    include_proxy_score: bool = False
    max_retries_per_gen: int = 2
    wait_timeout_s: float = 420.0
    poll_interval_s: float = 15.0

    def __post_init__(self) -> None:
        self.model_pins = validate_model_pins(self.model_pins)
        if not 1 <= self.target_generations <= self.max_generations:
            raise ValueError(
                f"target_generations ({self.target_generations}) must be in [1, {self.max_generations}]"
            )
        if self.dev_window_size < 2:
            raise ValueError("dev_window_size must be >= 2")

    def dev_seeds_for_round(self, round_idx: int) -> list[int]:
        """Return a fresh, audit-disjoint slice of development seeds for *round_idx*."""
        audit_set = set(self.audit_seeds)
        seeds: list[int] = []
        cursor = self.dev_seed_start + round_idx * self.dev_window_size
        while len(seeds) < self.dev_window_size:
            if cursor not in audit_set:
                seeds.append(cursor)
            cursor += 1
        return seeds


@dataclass(frozen=True)
class GenerationRecord:
    """Recorded outcome of one multi-generation self-improvement attempt."""

    round_idx: int
    variant_name: str
    target_generation: int
    stress_mode: str
    candidate_label: str
    diagnosis_reasons: tuple[str, ...]
    dev_accuracy: float
    incumbent_dev_accuracy: float
    variant_hash: str
    decision: str          # "approve" | "veto" | "timeout"
    decision_reasons: tuple[str, ...]
    installed: bool
    ts: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MultiGenSummary:
    """Aggregate summary of a pinned multi-generation run."""

    run_id: str
    model_pins: dict[str, str]
    start_generation: int
    end_generation: int
    target_generations: int
    approvals: int
    vetoes: int
    timeouts: int
    records: tuple[GenerationRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model_pins": dict(self.model_pins),
            "start_generation": self.start_generation,
            "end_generation": self.end_generation,
            "target_generations": self.target_generations,
            "approvals": self.approvals,
            "vetoes": self.vetoes,
            "timeouts": self.timeouts,
            "records": [r.to_dict() for r in self.records],
        }


def _save_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _fresh_variant_name(exchange: Path, target_gen: int) -> str:
    """Pick an unused variant name (``g1``, ``g1b``, ``g1c``, ...) in *exchange*."""
    suffixes = ("", "b", "c", "d", "e", "f", "g", "h")
    approvals_dir = exchange / "approvals"
    proposals_dir = exchange / "proposals"
    for suf in suffixes:
        name = f"g{target_gen}{suf}"
        if not (approvals_dir / f"{name}.json").exists() and not (proposals_dir / name).exists():
            return name
    return f"g{target_gen}_{int(time.time())}"


class MultiGenRunner:
    """Resumable multi-generation pinned self-improvement loop under Governor control."""

    def __init__(
        self,
        config: MultiGenConfig,
        *,
        gate: Gate,
        exchange: Path | str,
        install_root: Path | str,
        memory: AssemblyMemory,
        monitor: HomeostaticMonitor | None = None,
        client: ModelClient | None = None,
        event_log: EventLog | None = None,
        checkpoint_path: Path | str | None = None,
        on_wait_tick: Callable[[], None] | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if gate.run_id != config.run_id:
            raise ValueError(
                f"Gate run_id ({gate.run_id!r}) does not match MultiGenConfig.run_id ({config.run_id!r})"
            )
        self.config = config
        self.gate = gate
        self.exchange = Path(exchange)
        self.install_root = Path(install_root)
        self.memory = memory
        self.monitor = monitor or HomeostaticMonitor(event_log=event_log)
        self.client = client
        self.event_log = event_log
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self._on_wait_tick = on_wait_tick
        self._sleep = sleep_fn
        self._clock = clock

    def _load_current_harness(self) -> LoadedHarness:
        hdir = self.install_root / "current" / "harness"
        return load_harness(hdir if hdir.is_dir() else None)

    def _run_stress_probe(
        self,
        mode: str,
        harness: LoadedHarness,
        dev_seeds: Sequence[int],
        round_idx: int,
    ) -> StressDiagnosis:
        """Run a real stress probe across the Markov blanket to obtain a StressDiagnosis."""
        if mode == "law_shift":
            for idx, s in enumerate(dev_seeds[:4]):
                self.gate.check_heartbeat()
                run_law_shift_episode(
                    s,
                    shift_seed=700 + round_idx * 17 + idx,
                    harness=harness,
                    memory=self.memory,
                    monitor=self.monitor,
                    noise=self.config.noise,
                    budget=self.config.audit_budget,
                    rng_seed=round_idx * 100 + idx,
                    episode_idx=idx,
                    mode_label=f"multigen_probe_r{round_idx}",
                )
            diag = self.monitor.last_diagnosis
            if "HIGH_SURPRISE" in diag.reasons:
                return diag
            return StressDiagnosis(
                triggered=True,
                reasons=tuple(dict.fromkeys((*diag.reasons, "HIGH_SURPRISE"))),
                violations=diag.violations or ("scheduled law-shift stress probe",),
                rolling_metrics=self.monitor.status_summary().get("rolling_metrics", {}),
            )

        # Default: tight-budget stress probe
        probe_agent = CognitiveAgent(
            harness=harness,
            memory=self.memory,
            transfer=StructuralAnalogyTransfer(harness.transfer_params),
            monitor=self.monitor,
            rng_seed=round_idx * 100,
        )
        for idx, s in enumerate(dev_seeds[:6]):
            self.gate.check_heartbeat()
            probe_agent.run_episode(
                HiddenLawWorld(seed=s, noise=self.config.noise, budget=self.config.dev_budget),
                learn=False,
                episode_idx=idx,
            )
        diag = self.monitor.last_diagnosis
        if diag.reasons:
            return diag
        return StressDiagnosis(
            triggered=True,
            reasons=("HIGH_BUDGET",),
            violations=("scheduled tight-budget stress probe",),
            rolling_metrics=self.monitor.status_summary().get("rolling_metrics", {}),
        )

    def _wait_for_decision(self, variant_name: str) -> Path | None:
        approval_path = self.exchange / "approvals" / f"{variant_name}.json"
        deadline = self._clock() + self.config.wait_timeout_s
        while not approval_path.is_file():
            self.gate.check_heartbeat()
            if self._on_wait_tick is not None:
                self._on_wait_tick()
                if approval_path.is_file():
                    return approval_path
            rem = deadline - self._clock()
            if rem <= 0:
                return None
            self._sleep(min(self.config.poll_interval_s, rem))
        return approval_path

    def run(self, *, on_round: Any = None) -> MultiGenSummary:
        """Advance from current ``gate.generation`` up to ``config.target_generations``."""
        cfg = self.config
        start_gen = self.gate.generation
        records: list[GenerationRecord] = []

        if self.checkpoint_path and self.checkpoint_path.is_file():
            prev = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
            if prev.get("run_id") == cfg.run_id:
                start_gen = int(prev.get("start_generation", start_gen))
                for raw in prev.get("records", []):
                    records.append(
                        GenerationRecord(
                            round_idx=int(raw["round_idx"]),
                            variant_name=str(raw["variant_name"]),
                            target_generation=int(raw["target_generation"]),
                            stress_mode=str(raw["stress_mode"]),
                            candidate_label=str(raw["candidate_label"]),
                            diagnosis_reasons=tuple(raw.get("diagnosis_reasons", ())),
                            dev_accuracy=float(raw["dev_accuracy"]),
                            incumbent_dev_accuracy=float(raw["incumbent_dev_accuracy"]),
                            variant_hash=str(raw["variant_hash"]),
                            decision=str(raw["decision"]),
                            decision_reasons=tuple(raw.get("decision_reasons", ())),
                            installed=bool(raw["installed"]),
                            ts=float(raw["ts"]),
                        )
                    )

        proposer = AshbyProposer(
            client=self.client,
            event_log=self.event_log,
            max_generations=cfg.max_generations,
        )

        round_idx = len(records)
        retries_on_cur_gen = 0

        while self.gate.generation < cfg.target_generations:
            self.gate.check_heartbeat()
            target_gen = self.gate.generation + 1
            stress_mode = cfg.stress_modes[round_idx % len(cfg.stress_modes)]
            dev_seeds = cfg.dev_seeds_for_round(round_idx)
            variant_name = _fresh_variant_name(self.exchange, target_gen)
            current_harness = self._load_current_harness()

            diagnosis = self._run_stress_probe(stress_mode, current_harness, dev_seeds, round_idx)
            outcome, _evals = proposer.screen_and_publish(
                variant_name=variant_name,
                current_harness=current_harness,
                diagnosis=diagnosis,
                memory=self.memory,
                gate=self.gate,
                exchange=self.exchange,
                dev_seeds=dev_seeds,
                audit_seeds=cfg.audit_seeds,
                model_pins=cfg.model_pins,
                noise=cfg.noise,
                budget=cfg.dev_budget,
                audit_budget=cfg.audit_budget,
                use_transfer=True,
                include_proxy_score=cfg.include_proxy_score,
                rng_seed=300 + round_idx * 31,
            )

            approval_path = self._wait_for_decision(variant_name)
            installed = False
            decision = "timeout"
            decision_reasons: tuple[str, ...] = ("timed out waiting for governor decision",)

            if approval_path is not None:
                try:
                    self.gate.install(outcome.proposal_dir)
                    installed = True
                    decision = "approve"
                    decision_reasons = ("installed",)
                    retries_on_cur_gen = 0
                except InstallRefused as exc:
                    decision = "veto"
                    decision_reasons = (str(exc),)
                    retries_on_cur_gen += 1
            else:
                retries_on_cur_gen += 1

            rec = GenerationRecord(
                round_idx=round_idx,
                variant_name=variant_name,
                target_generation=target_gen,
                stress_mode=stress_mode,
                candidate_label=outcome.candidate_label,
                diagnosis_reasons=diagnosis.reasons,
                dev_accuracy=outcome.dev_accuracy,
                incumbent_dev_accuracy=outcome.incumbent_dev_accuracy,
                variant_hash=outcome.variant_hash,
                decision=decision,
                decision_reasons=decision_reasons,
                installed=installed,
                ts=round(self._clock(), 3),
            )
            records.append(rec)
            round_idx += 1

            if self.event_log is not None:
                self.event_log.append("multigen_round", rec.to_dict())

            summary = self._make_summary(start_gen, records)
            if self.checkpoint_path is not None:
                _save_json_atomic(self.checkpoint_path, summary.to_dict())
            if on_round is not None:
                on_round(rec)

            if not installed and retries_on_cur_gen > cfg.max_retries_per_gen:
                break

        return self._make_summary(start_gen, records)

    def _make_summary(
        self, start_gen: int, records: Sequence[GenerationRecord]
    ) -> MultiGenSummary:
        return MultiGenSummary(
            run_id=self.config.run_id,
            model_pins=dict(self.config.model_pins),
            start_generation=start_gen,
            end_generation=self.gate.generation,
            target_generations=self.config.target_generations,
            approvals=sum(1 for r in records if r.decision == "approve"),
            vetoes=sum(1 for r in records if r.decision == "veto"),
            timeouts=sum(1 for r in records if r.decision == "timeout"),
            records=tuple(records),
        )
