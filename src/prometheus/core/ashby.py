"""Ashby ultrastability monitor and bounded self-improvement proposer (Phase 2).

Implements Ashby's homeostatic step-function control (*Design for a Brain*)
paired with Good's bounded self-improvement under the external Governor:

1. :class:`HomeostaticMonitor` (implements :class:`UltrastabilityMonitor`) tracks
   a rolling window of :class:`EssentialMetrics` against the **immutable**
   :class:`EssentialBounds` (protected core code — mutable harness variants can
   never weaken these bounds).  When essential variables persistently leave
   bounds, it emits a :class:`StressDiagnosis` and triggers adaptation.
2. :class:`AshbyProposer` responds to stress by generating candidate mutable
   harness mutations strictly within the governor-allowed ``harness/`` subdirs
   (``memory_params/``, ``mesh/``, ``prompts/``), validating every candidate
   through :func:`load_harness`, screening on held-out development seeds
   (strictly disjoint from ``audit_seeds``!), evaluating ``audit_seeds`` in
   read-only memory mode (``learn=False``), and publishing a content-hashed
   bundle + ``audit_claims.json`` + ``.ready`` sentinel for the external
   workstation Governor to approve or veto.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import shutil
import tempfile
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..common.eventlog import EventLog
from ..common.variant import write_bundle
from ..harness.client import ModelClient, ModelClientError
from ..lab.laws import LAW_NAMES
from ..lab.world import HiddenLawWorld
from ..runtime.gate import Gate
from .agent import CognitiveAgent
from .blanket import DEFAULT_ESSENTIAL_BOUNDS, EssentialBounds, EssentialMetrics
from .harness_loader import (
    HarnessValidationError,
    LoadedHarness,
    MeshConfig,
    load_harness,
    run_harness_evaluator,
)
from .inference import EFEParams
from .interfaces import AshbyConfig
from .memory import AssemblyMemory, MemoryParams
from .transfer import StructuralAnalogyTransfer, TransferParams


@dataclass(frozen=True)
class StressDiagnosis:
    """Structured diagnosis of which immutable essential variables are stressed."""

    triggered: bool
    reasons: tuple[str, ...] = ()
    violations: tuple[str, ...] = ()
    rolling_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StressDiagnosis":
        return cls(
            triggered=bool(data.get("triggered", False)),
            reasons=tuple(str(r) for r in data.get("reasons", ())),
            violations=tuple(str(v) for v in data.get("violations", ())),
            rolling_metrics={str(k): float(v) for k, v in data.get("rolling_metrics", {}).items()},
        )


class HomeostaticMonitor:
    """Ashby ultrastability monitor guarding immutable :class:`EssentialBounds`."""

    def __init__(
        self,
        config: AshbyConfig | None = None,
        *,
        event_log: EventLog | None = None,
    ) -> None:
        self.config = config or AshbyConfig()
        # Immutable protected bounds — never loaded from mutable harness artifacts.
        self._bounds: EssentialBounds = DEFAULT_ESSENTIAL_BOUNDS
        self._event_log = event_log
        self._window: deque[dict[str, float | int]] = deque(maxlen=max(2, self.config.window_size))
        self.total_episodes: int = 0
        self.trigger_count: int = 0
        self._episodes_since_trigger: int = self.config.cooldown_episodes
        self._last_violations: list[str] = []
        self.last_diagnosis: StressDiagnosis = StressDiagnosis(triggered=False)

    @property
    def bounds(self) -> EssentialBounds:
        """Read-only access to the immutable essential-variable bounds."""
        return self._bounds

    def _compute_rolling(self) -> dict[str, float]:
        if not self._window:
            return {}
        n = len(self._window)
        mean_conf = sum(float(m["mean_confidence"]) for m in self._window) / n
        mean_bfrac = sum(float(m["budget_fraction"]) for m in self._window) / n
        mean_surp = sum(float(m["mean_surprise"]) for m in self._window) / n
        max_surp = max(float(m["max_surprise"]) for m in self._window)
        tot_err = sum(int(m["sandbox_errors"]) for m in self._window)
        tot_obs = sum(max(1, int(m["observations_used"])) for m in self._window)
        err_rate = tot_err / max(1, tot_obs) if tot_err else 0.0
        return {
            "window_episodes": float(n),
            "mean_confidence": round(mean_conf, 4),
            "mean_budget_fraction": round(mean_bfrac, 4),
            "mean_surprise": round(mean_surp, 4),
            "max_surprise": round(max_surp, 4),
            "sandbox_error_rate": round(err_rate, 4),
        }

    def record(self, metrics: EssentialMetrics) -> bool:
        """Record one episode's essential metrics; return True if adaptation should trigger."""
        ok, ep_viols = metrics.check_bounds(self._bounds)
        spike = metrics.max_surprise > self.config.surprise_spike_threshold
        if spike:
            ep_viols = list(ep_viols) + [
                f"max_surprise {metrics.max_surprise:.3f} > {self.config.surprise_spike_threshold:.3f}"
            ]
            ok = False

        entry = {
            "mean_confidence": float(metrics.mean_confidence),
            "budget_fraction": float(metrics.budget_fraction),
            "mean_surprise": float(metrics.mean_surprise),
            "max_surprise": float(metrics.max_surprise),
            "observations_used": int(metrics.observations_used),
            "sandbox_errors": int(metrics.sandbox_errors),
            "breached": 0 if ok else 1,
        }
        self._window.append(entry)
        self.total_episodes += 1
        self._episodes_since_trigger += 1

        rolling = self._compute_rolling()
        reasons: list[str] = []
        active_viols: list[str] = list(ep_viols)

        if rolling["mean_confidence"] < self._bounds.min_confidence:
            reasons.append("LOW_CONFIDENCE")
            active_viols.append(
                f"rolling mean_confidence {rolling['mean_confidence']:.3f} < {self._bounds.min_confidence:.3f}"
            )
        if rolling["mean_budget_fraction"] > self._bounds.max_budget_fraction:
            reasons.append("HIGH_BUDGET")
            active_viols.append(
                f"rolling budget_fraction {rolling['mean_budget_fraction']:.3f} > {self._bounds.max_budget_fraction:.3f}"
            )
        if (
            rolling["mean_surprise"] > self._bounds.max_rolling_surprise
            or rolling["max_surprise"] > self.config.surprise_spike_threshold
        ):
            reasons.append("HIGH_SURPRISE")
            if rolling["mean_surprise"] > self._bounds.max_rolling_surprise:
                active_viols.append(
                    f"rolling mean_surprise {rolling['mean_surprise']:.3f} > {self._bounds.max_rolling_surprise:.3f}"
                )
        if rolling["sandbox_error_rate"] > self._bounds.max_sandbox_error_rate:
            reasons.append("SANDBOX_ERRORS")
            active_viols.append(
                f"rolling sandbox_error_rate {rolling['sandbox_error_rate']:.3f} > {self._bounds.max_sandbox_error_rate:.3f}"
            )

        # Also catch per-episode breach channel if breaches accumulated without moving full mean yet
        breach_count = sum(int(m["breached"]) for m in self._window)
        if breach_count >= self.config.min_breach_episodes and not reasons:
            if metrics.mean_confidence < self._bounds.min_confidence:
                reasons.append("LOW_CONFIDENCE")
            if metrics.budget_fraction > self._bounds.max_budget_fraction:
                reasons.append("HIGH_BUDGET")
            if metrics.mean_surprise > self._bounds.max_rolling_surprise or spike:
                reasons.append("HIGH_SURPRISE")
            if metrics.sandbox_errors > 0:
                reasons.append("SANDBOX_ERRORS")

        dedup_viols = list(dict.fromkeys(active_viols))
        dedup_reasons = tuple(dict.fromkeys(reasons))
        self._last_violations = dedup_viols

        ready = (
            self.total_episodes >= self.config.min_episodes_before_trigger
            and self._episodes_since_trigger >= self.config.cooldown_episodes
        )
        triggered = bool(ready and dedup_reasons and breach_count >= self.config.min_breach_episodes)
        if triggered:
            self.trigger_count += 1
            self._episodes_since_trigger = 0

        self.last_diagnosis = StressDiagnosis(
            triggered=triggered,
            reasons=dedup_reasons,
            violations=tuple(dedup_viols),
            rolling_metrics=rolling,
        )

        if triggered and self._event_log is not None:
            self._event_log.append(
                "ashby_trigger",
                {
                    "episode": self.total_episodes,
                    "trigger_count": self.trigger_count,
                    "reasons": list(dedup_reasons),
                    "violations": dedup_viols,
                    "rolling_metrics": rolling,
                },
            )
        return triggered

    def violations(self) -> list[str]:
        return list(self._last_violations)

    def status_summary(self) -> dict[str, Any]:
        rolling = self._compute_rolling()
        stressed = bool(self.last_diagnosis.reasons)
        return {
            "status": "STRESSED" if stressed else "HOMEOSTATIC",
            "total_episodes": self.total_episodes,
            "trigger_count": self.trigger_count,
            "active_reasons": list(self.last_diagnosis.reasons),
            "violations": list(self._last_violations),
            "rolling_metrics": rolling,
        }

    # ----- persistence (survives Colab forced restarts) ---------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "config": asdict(self.config),
            "total_episodes": self.total_episodes,
            "trigger_count": self.trigger_count,
            "episodes_since_trigger": self._episodes_since_trigger,
            "window": list(self._window),
            "last_violations": list(self._last_violations),
            "last_diagnosis": self.last_diagnosis.to_dict(),
            "summary": self.status_summary(),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        config_override: AshbyConfig | None = None,
        event_log: EventLog | None = None,
    ) -> "HomeostaticMonitor":
        raw_cfg = data.get("config") or {}
        if config_override is not None:
            cfg = config_override
        else:
            cfg = AshbyConfig(**{k: raw_cfg[k] for k in raw_cfg if hasattr(AshbyConfig, k)})
        mon = cls(config=cfg, event_log=event_log)
        mon.total_episodes = int(data.get("total_episodes", 0))
        mon.trigger_count = int(data.get("trigger_count", 0))
        mon._episodes_since_trigger = int(data.get("episodes_since_trigger", cfg.cooldown_episodes))
        for item in data.get("window", []):
            if isinstance(item, dict):
                mon._window.append(item)
        mon._last_violations = [str(v) for v in data.get("last_violations", [])]
        if isinstance(data.get("last_diagnosis"), dict):
            mon.last_diagnosis = StressDiagnosis.from_dict(data["last_diagnosis"])
        return mon

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f".{p.name}.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, sort_keys=True, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        config_override: AshbyConfig | None = None,
        event_log: EventLog | None = None,
    ) -> "HomeostaticMonitor":
        p = Path(path)
        if not p.is_file():
            return cls(config=config_override, event_log=event_log)
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_dict(data, config_override=config_override, event_log=event_log)


# ---------------------------------------------------------------------------
# Bounded Ashby step-function harness proposer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CandidateEvaluation:
    """Development-set screening outcome for one candidate harness configuration."""

    label: str
    dev_accuracy: float
    mean_budget_spent: float
    mean_confidence: float
    mean_surprise: float
    bound_violations: int
    proxy_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProposalOutcome:
    """Result of publishing one bounded Ashby harness proposal to the Governor."""

    variant_name: str
    proposal_dir: Path
    variant_hash: str
    generation: int
    candidate_label: str
    proxy_score: float
    dev_accuracy: float
    incumbent_dev_accuracy: float
    diagnosis: StressDiagnosis
    files: tuple[str, ...]


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(val)))


def _serialize_harness_files(
    *,
    efe: EFEParams,
    mem: MemoryParams,
    transfer: TransferParams,
    mesh: MeshConfig,
    ashby: AshbyConfig,
    prompts: dict[str, str] | None = None,
    tool_source: str | None = None,
    evaluator_source: str | None = None,
) -> dict[str, str]:
    files: dict[str, str] = {
        "harness/memory_params/efe_params.json": json.dumps(asdict(efe), indent=2, sort_keys=True) + "\n",
        "harness/memory_params/memory_params.json": json.dumps(asdict(mem), indent=2, sort_keys=True) + "\n",
        "harness/memory_params/transfer_params.json": json.dumps(asdict(transfer), indent=2, sort_keys=True) + "\n",
        "harness/mesh/config.json": json.dumps(asdict(mesh), indent=2, sort_keys=True) + "\n",
        "harness/mesh/ashby_params.json": json.dumps(asdict(ashby), indent=2, sort_keys=True) + "\n",
    }
    for name, body in (prompts or {}).items():
        safe_name = "".join(ch for ch in name if ch.isalnum() or ch in ("_", "-"))
        if safe_name:
            files[f"harness/prompts/{safe_name}.md"] = body
    if tool_source is not None:
        files["harness/tools/strategy.py"] = tool_source
    if evaluator_source is not None:
        files["harness/evaluators/score.py"] = evaluator_source
    return files


def _validate_candidate_files(files: dict[str, str]) -> LoadedHarness | None:
    """Validate candidate ``harness/...`` files through :func:`load_harness` in a temp dir."""
    for rel in files:
        if not rel.startswith("harness/"):
            return None
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for rel, content in files.items():
            dest = root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
        try:
            return load_harness(root / "harness")
        except (HarnessValidationError, ValueError, OSError):
            return None


class AshbyProposer:
    """Generates, screens, and publishes governor-bound harness mutations."""

    def __init__(
        self,
        *,
        client: ModelClient | None = None,
        event_log: EventLog | None = None,
        max_generations: int = 10,
    ) -> None:
        self.client = client
        self.event_log = event_log
        self.max_generations = int(max_generations)

    def _propose_llm_candidate(
        self,
        current: LoadedHarness,
        diagnosis: StressDiagnosis,
    ) -> tuple[str, dict[str, str]] | None:
        """Optionally ask the Cortex LLM for a JSON hyperparameter refinement."""
        if self.client is None:
            return None
        messages = [
            {
                "role": "system",
                "content": (
                    "You are proposing a bounded hyperparameter mutation for an active-inference "
                    "cognitive harness experiencing Ashby essential-variable stress. "
                    "Reply ONLY with a JSON object containing optional keys "
                    "'efe_params', 'memory_params', 'transfer_params', 'mesh' with numeric/bool overrides."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "stress_reasons": list(diagnosis.reasons),
                        "violations": list(diagnosis.violations),
                        "rolling_metrics": diagnosis.rolling_metrics,
                        "current": {
                            "efe_params": asdict(current.efe_params),
                            "memory_params": asdict(current.memory_params),
                            "transfer_params": asdict(current.transfer_params),
                            "mesh": asdict(current.mesh),
                        },
                    },
                    indent=2,
                    sort_keys=True,
                ),
            },
        ]
        try:
            raw = self.client.chat_json(messages)
        except ModelClientError:
            return None
        if not isinstance(raw, dict):
            return None

        base_files = _serialize_harness_files(
            efe=current.efe_params,
            mem=current.memory_params,
            transfer=current.transfer_params,
            mesh=current.mesh,
            ashby=current.ashby_config,
            prompts=current.prompts,
            tool_source=current.tool_source,
            evaluator_source=current.evaluator_source,
        )
        mapping = {
            "efe_params": ("harness/memory_params/efe_params.json", asdict(current.efe_params)),
            "memory_params": ("harness/memory_params/memory_params.json", asdict(current.memory_params)),
            "transfer_params": ("harness/memory_params/transfer_params.json", asdict(current.transfer_params)),
            "mesh": ("harness/mesh/config.json", asdict(current.mesh)),
        }
        changed = False
        for key, (rel, cur_dict) in mapping.items():
            patch = raw.get(key)
            if isinstance(patch, dict) and patch:
                merged = dict(cur_dict)
                for k, v in patch.items():
                    if k in merged:
                        merged[k] = v
                        changed = True
                base_files[rel] = json.dumps(merged, indent=2, sort_keys=True) + "\n"
        if not changed:
            return None
        if _validate_candidate_files(base_files) is None:
            return None
        return ("llm_refinement", base_files)

    def generate_candidates(
        self,
        current: LoadedHarness,
        diagnosis: StressDiagnosis,
    ) -> list[tuple[str, dict[str, str]]]:
        """Generate validated candidate harness file maps tailored to ``diagnosis``."""
        candidates: list[tuple[str, dict[str, str]]] = []
        efe = current.efe_params
        mem = current.memory_params
        tr = current.transfer_params
        mesh = current.mesh
        ashby = current.ashby_config
        reasons = set(diagnosis.reasons)

        def _add(label: str, *, efe_n=efe, mem_n=mem, tr_n=tr, mesh_n=mesh, tool_n=current.tool_source):
            files = _serialize_harness_files(
                efe=efe_n,
                mem=mem_n,
                transfer=tr_n,
                mesh=mesh_n,
                ashby=ashby,
                prompts=current.prompts,
                tool_source=tool_n,
                evaluator_source=current.evaluator_source,
            )
            if _validate_candidate_files(files) is not None:
                candidates.append((label, files))

        if "LOW_CONFIDENCE" in reasons or not reasons:
            _add(
                "precision_boost",
                efe_n=dataclasses.replace(
                    efe,
                    beta_law=_clamp(max(efe.beta_law * 1.4, 4.5), 0.5, 50.0),
                    lambda_cost=_clamp(min(efe.lambda_cost * 0.65, 0.08), 0.01, 5.0),
                    stop_entropy_bits=_clamp(min(efe.stop_entropy_bits * 0.6, 0.12), 0.04, 2.0),
                    stop_eig_per_cost=_clamp(min(efe.stop_eig_per_cost * 0.35, 0.012), 0.002, 1.0),
                    min_observations_before_stop=min(12, max(4, efe.min_observations_before_stop + 2)),
                ),
            )

        if "HIGH_BUDGET" in reasons or not reasons:
            _add(
                "budget_efficiency",
                efe_n=dataclasses.replace(
                    efe,
                    lambda_cost=_clamp(efe.lambda_cost * 1.6, 0.02, 5.0),
                    stop_entropy_bits=_clamp(efe.stop_entropy_bits * 1.35, 0.08, 0.6),
                    stop_eig_per_cost=_clamp(efe.stop_eig_per_cost * 1.5, 0.005, 0.25),
                ),
                mem_n=dataclasses.replace(
                    mem,
                    prior_blend=_clamp(mem.prior_blend + 0.12, 0.1, 0.70),
                ),
                tr_n=dataclasses.replace(
                    tr,
                    enabled=True,
                    max_prior_blend=_clamp(tr.max_prior_blend + 0.10, 0.2, 0.75),
                ),
            )

        if "HIGH_SURPRISE" in reasons:
            _add(
                "forgetting_adapt",
                efe_n=dataclasses.replace(
                    efe,
                    forgetting_factor=_clamp(min(efe.forgetting_factor, 0.65), 0.25, 0.90),
                    beta_law=_clamp(max(efe.beta_law, 5.0), 1.0, 50.0),
                ),
                mem_n=dataclasses.replace(
                    mem,
                    prior_blend=_clamp(mem.prior_blend * 0.5, 0.05, 0.5),
                ),
                tr_n=dataclasses.replace(
                    tr,
                    surprise_veto_threshold=_clamp(min(tr.surprise_veto_threshold, 2.5), 1.5, 5.0),
                    max_prior_blend=_clamp(tr.max_prior_blend * 0.6, 0.1, 0.5),
                ),
            )

        if "SANDBOX_ERRORS" in reasons:
            _add(
                "disable_faulty_tool",
                mesh_n=dataclasses.replace(mesh, use_tool_hook=False),
                tool_n=None,
            )

        # Always include a balanced transfer-enhanced candidate for comparison
        _add(
            "analogy_balanced",
            efe_n=dataclasses.replace(
                efe,
                beta_law=_clamp(max(efe.beta_law, 4.5), 1.0, 50.0),
                lambda_cost=_clamp(min(efe.lambda_cost, 0.10), 0.01, 5.0),
                stop_entropy_bits=_clamp(min(efe.stop_entropy_bits, 0.14), 0.05, 1.0),
                stop_eig_per_cost=_clamp(min(efe.stop_eig_per_cost, 0.012), 0.002, 1.0),
            ),
            tr_n=dataclasses.replace(
                tr,
                enabled=True,
                enable_probe_suggestions=True,
                min_confidence=_clamp(min(tr.min_confidence, 0.42), 0.2, 0.8),
            ),
        )

        llm_cand = self._propose_llm_candidate(current, diagnosis)
        if llm_cand is not None:
            candidates.append(llm_cand)

        return candidates

    def evaluate_harness_on_dev(
        self,
        label: str,
        harness: LoadedHarness,
        *,
        dev_seeds: Sequence[int],
        memory: AssemblyMemory,
        noise: float = 0.05,
        budget: float = 500.0,
        use_transfer: bool = True,
        rng_seed: int = 0,
    ) -> CandidateEvaluation:
        """Screen *harness* on development seeds using a read-only copy of *memory*."""
        mem_copy = copy.deepcopy(memory)
        transfer = StructuralAnalogyTransfer(harness.transfer_params) if use_transfer else None
        agent = CognitiveAgent(
            harness=harness,
            memory=mem_copy,
            transfer=transfer,
            rng_seed=rng_seed,
        )
        hits = 0
        total = 0
        spent_sum = 0.0
        conf_sum = 0.0
        surp_sum = 0.0
        bound_viols = 0
        for idx, s in enumerate(dev_seeds):
            w = HiddenLawWorld(seed=s, noise=noise, budget=budget)
            res = agent.run_episode(w, learn=False, episode_idx=idx)
            truth = w.ground_truth()
            hits += sum(res.claims[k] == truth[k] for k in LAW_NAMES)
            total += len(LAW_NAMES)
            spent_sum += res.budget_spent
            conf_sum += res.metrics.mean_confidence
            surp_sum += res.metrics.mean_surprise
            ok, _ = res.metrics.check_bounds(DEFAULT_ESSENTIAL_BOUNDS)
            if not ok:
                bound_viols += 1

        n = max(1, len(dev_seeds))
        dev_acc = hits / max(1, total)
        mean_spent = spent_sum / n
        mean_conf = conf_sum / n
        mean_surp = surp_sum / n
        # Composite proxy score (prefers accuracy, penalises budget waste and bound breaches)
        budget_eff = max(0.0, 1.0 - (mean_spent / max(budget, 1.0)))
        proxy = round(max(0.0, min(1.0, 0.85 * dev_acc + 0.15 * budget_eff - 0.05 * (bound_viols / n))), 4)

        if harness.evaluator_source:
            eval_score, err = run_harness_evaluator(
                harness.evaluator_source,
                {
                    "dev_accuracy": dev_acc,
                    "mean_budget_spent": mean_spent,
                    "mean_confidence": mean_conf,
                    "mean_surprise": mean_surp,
                    "bound_violations": bound_viols,
                },
            )
            if err is None and eval_score is not None:
                proxy = round(eval_score, 4)

        return CandidateEvaluation(
            label=label,
            dev_accuracy=round(dev_acc, 4),
            mean_budget_spent=round(mean_spent, 2),
            mean_confidence=round(mean_conf, 4),
            mean_surprise=round(mean_surp, 4),
            bound_violations=bound_viols,
            proxy_score=proxy,
        )

    def screen_and_publish(
        self,
        *,
        variant_name: str,
        current_harness: LoadedHarness,
        diagnosis: StressDiagnosis,
        memory: AssemblyMemory,
        gate: Gate,
        exchange: Path | str,
        dev_seeds: Sequence[int],
        audit_seeds: Sequence[int],
        model_pins: dict[str, str],
        noise: float = 0.05,
        budget: float = 500.0,
        audit_budget: float = 500.0,
        use_transfer: bool = True,
        include_proxy_score: bool = True,
        rng_seed: int = 0,
    ) -> tuple[ProposalOutcome, list[CandidateEvaluation]]:
        """Screen candidate mutations on dev seeds, audit winner read-only, and publish bundle."""
        overlap = set(dev_seeds) & set(audit_seeds)
        if overlap:
            raise ValueError(f"dev_seeds must be strictly disjoint from audit_seeds (overlap: {sorted(overlap)[:5]})")
        next_gen = gate.generation + 1
        if next_gen > self.max_generations:
            raise RuntimeError(
                f"generation cap of {self.max_generations} reached (current={gate.generation})"
            )

        incumbent_eval = self.evaluate_harness_on_dev(
            "incumbent",
            current_harness,
            dev_seeds=dev_seeds,
            memory=memory,
            noise=noise,
            budget=budget,
            use_transfer=use_transfer,
            rng_seed=rng_seed,
        )
        evaluations: list[CandidateEvaluation] = [incumbent_eval]

        candidates = self.generate_candidates(current_harness, diagnosis)
        if not candidates:
            raise RuntimeError("AshbyProposer generated zero valid candidate harnesses")

        best_label: str = ""
        best_files: dict[str, str] = {}
        best_loaded: LoadedHarness | None = None
        best_eval: CandidateEvaluation | None = None

        for label, files in candidates:
            loaded = _validate_candidate_files(files)
            if loaded is None:
                continue
            ev = self.evaluate_harness_on_dev(
                label,
                loaded,
                dev_seeds=dev_seeds,
                memory=memory,
                noise=noise,
                budget=budget,
                use_transfer=use_transfer,
                rng_seed=rng_seed,
            )
            evaluations.append(ev)
            if best_eval is None or (
                ev.dev_accuracy,
                -ev.bound_violations,
                -ev.mean_budget_spent,
                ev.proxy_score,
            ) > (
                best_eval.dev_accuracy,
                -best_eval.bound_violations,
                -best_eval.mean_budget_spent,
                best_eval.proxy_score,
            ):
                best_label = label
                best_files = files
                best_loaded = loaded
                best_eval = ev

        assert best_loaded is not None and best_eval is not None

        # Evaluate frozen AUDIT_SEEDS strictly read-only (learn=False, no ground truth!)
        mem_ro = copy.deepcopy(memory)
        transfer = StructuralAnalogyTransfer(best_loaded.transfer_params) if use_transfer else None
        audit_agent = CognitiveAgent(
            harness=best_loaded,
            memory=mem_ro,
            transfer=transfer,
            rng_seed=rng_seed + 1000,
        )
        audit_claims = audit_agent.evaluate_audit_seeds(
            audit_seeds, noise=noise, budget=audit_budget
        )

        # Stage and publish content-hashed bundle + audit_claims.json + .ready sentinel
        ex_path = Path(exchange)
        proposals_root = ex_path / "proposals"
        proposals_root.mkdir(parents=True, exist_ok=True)
        staging_dir = proposals_root / f".staging-{variant_name}"
        proposal_dir = proposals_root / variant_name
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        if proposal_dir.exists():
            shutil.rmtree(proposal_dir)

        manifest_fields: dict[str, Any] = {
            "run_id": gate.run_id,
            "generation": next_gen,
            "parent_hash": gate.installed_hash,
            "model_pins": dict(model_pins),
            "created_at": time.time(),
        }
        if include_proxy_score:
            manifest_fields["proxy_score"] = best_eval.proxy_score

        bundle = write_bundle(staging_dir, manifest_fields, best_files)
        (staging_dir / "audit_claims.json").write_text(
            json.dumps(audit_claims, indent=2, sort_keys=True), encoding="utf-8"
        )
        staging_dir.rename(proposal_dir)
        (proposal_dir / ".ready").write_text(bundle.variant_hash + "\n", encoding="utf-8")

        outcome = ProposalOutcome(
            variant_name=variant_name,
            proposal_dir=proposal_dir,
            variant_hash=bundle.variant_hash,
            generation=next_gen,
            candidate_label=best_label,
            proxy_score=best_eval.proxy_score,
            dev_accuracy=best_eval.dev_accuracy,
            incumbent_dev_accuracy=incumbent_eval.dev_accuracy,
            diagnosis=diagnosis,
            files=tuple(sorted(best_files)),
        )

        if self.event_log is not None:
            self.event_log.append(
                "ashby_proposal",
                {
                    "variant_name": variant_name,
                    "variant_hash": bundle.variant_hash,
                    "generation": next_gen,
                    "candidate_label": best_label,
                    "proxy_score": best_eval.proxy_score,
                    "dev_accuracy": best_eval.dev_accuracy,
                    "incumbent_dev_accuracy": incumbent_eval.dev_accuracy,
                    "reasons": list(diagnosis.reasons),
                    "n_candidates_screened": len(evaluations) - 1,
                },
            )
        return outcome, evaluations
