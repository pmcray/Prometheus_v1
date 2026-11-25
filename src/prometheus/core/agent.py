"""Hybrid multi-turn cognitive episode runner for Prometheus v1.

Orchestrates the Phase 1 perception-inference-memory-action loop:

1. Enforces the immutable Markov blanket via :class:`RestrictedWorld` (ground
   truth, family name, and raw tables are never accessible inside the agent).
2. Runs Friston active inference / EFE through an :class:`InferenceBackend`
   (default: pure-Python :class:`ExactBayesianBackend`).
3. Integrates Good's MDL :class:`AssemblyMemory` for probabilistic prior
   regeneration and post-episode consolidation (read-only during audits).
4. Optionally consults:
   - a sandboxed mutable tool hook from ``harness/tools/strategy.py``,
   - a Cortex LLM advisor via :class:`ModelClient` (when ``client`` is supplied
     and ``mesh.use_llm_advisor`` is enabled),
   - a Phase 2 :class:`AnalogyTransferAdapter`, and
   - a Phase 2/3 :class:`UltrastabilityMonitor`.
5. Optionally logs structured hash-chained ``cognitive_episode`` events to an
   :class:`EventLog` for Observatory inspection.
"""

from __future__ import annotations

import dataclasses
import random
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..common.eventlog import EventLog
from ..harness.client import ModelClient, ModelClientError
from ..harness.prompt import build_advisor_messages, parse_advisor_reply
from ..lab.laws import LAW_NAMES
from ..lab.world import BudgetExhausted, HiddenLawWorld
from .blanket import (
    ActionScore,
    EssentialMetrics,
    ExperimentAction,
    Observation,
    RestrictedWorld,
    WorldView,
)
from .harness_loader import LoadedHarness, run_harness_tool
from .inference import ExactBayesianBackend
from .interfaces import AnalogyTransferAdapter, InferenceBackend, UltrastabilityMonitor
from .memory import AssemblyMemory


@dataclass
class CognitiveEpisodeResult:
    """Blind episode outcome (intentionally excludes ground truth / family)."""

    seed: int | None
    n: int
    claims: dict[str, bool]
    law_probs: dict[str, float]
    observations_used: int
    budget_spent: float
    final_entropy_bits: float
    metrics: EssentialMetrics
    activated_assemblies: dict[str, float] = field(default_factory=dict)
    stopped_early: bool = False
    stop_reason: str = ""
    latency_s: float = 0.0
    llm_notes: str | None = None
    error: str | None = None
    transfer_hints: int = 0
    transfer_source: str | None = None
    transfer_confidence: float = 0.0
    ashby_triggered: bool = False
    ashby_violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _action_key(action: ExperimentAction) -> tuple[str, tuple[str, ...]]:
    return (action.kind, action.args)


def _pick_from_tool_return(
    raw: Any,
    legal_by_key: dict[tuple[str, tuple[str, ...]], ExperimentAction],
) -> ExperimentAction | None:
    if not isinstance(raw, dict):
        return None
    kind = raw.get("kind")
    args = raw.get("args")
    if isinstance(kind, str) and isinstance(args, (list, tuple)) and all(isinstance(x, str) for x in args):
        return legal_by_key.get((kind, tuple(args)))
    return None


class CognitiveAgent:
    """Phase 1 cognitive agent running offline on CPU or LLM-assisted on Colab."""

    def __init__(
        self,
        *,
        harness: LoadedHarness | None = None,
        backend: InferenceBackend | None = None,
        memory: AssemblyMemory | None = None,
        client: ModelClient | None = None,
        transfer: AnalogyTransferAdapter | None = None,
        monitor: UltrastabilityMonitor | None = None,
        event_log: EventLog | None = None,
        rng_seed: int = 0,
    ) -> None:
        self.harness = harness or LoadedHarness()
        self.backend: InferenceBackend = backend or ExactBayesianBackend(self.harness.efe_params)
        self.memory = memory if memory is not None else AssemblyMemory(self.harness.memory_params)
        self.client = client
        self.transfer = transfer
        self.monitor = monitor
        self.event_log = event_log
        self._rng = random.Random(rng_seed)

    def _consult_tool(
        self,
        view: WorldView,
        law_probs: dict[str, float],
        scores: list[ActionScore],
    ) -> tuple[ExperimentAction | None, bool]:
        """Run optional sandboxed ``harness/tools/strategy.py``; return ``(action, had_error)``."""
        if not self.harness.tool_source or not self.harness.mesh.use_tool_hook:
            return (None, False)
        top = scores[:8]
        legal = {_action_key(s.action): s.action for s in scores}
        tool_inputs = {
            "symbols": view.symbols,
            "remaining_budget": view.remaining,
            "noise": view.noise,
            "law_probs": dict(law_probs),
            "top_candidates": [
                {
                    "kind": s.action.kind,
                    "args": list(s.action.args),
                    "cost": s.expected_cost,
                    "efe": round(s.efe, 6),
                    "law_eig_bits": round(s.law_eig_bits, 6),
                }
                for s in top
            ],
        }
        res = run_harness_tool(self.harness.tool_source, tool_inputs)
        if res.error:
            return (None, True)
        picked = _pick_from_tool_return(res.return_value, legal)
        return (picked, False)

    def _consult_llm_advisor(
        self,
        view: WorldView,
        observations: list[Observation],
        law_probs: dict[str, float],
        scores: list[ActionScore],
        activated_assemblies: dict[str, float],
    ) -> tuple[ExperimentAction | None, dict[str, bool] | None, str | None]:
        """Query optional Cortex LLM advisor; never raises."""
        if self.client is None or not self.harness.mesh.use_llm_advisor:
            return (None, None, None)
        obs_dicts = [
            {
                "kind": o.action.kind,
                "args": o.action.args,
                "result": o.result,
                "cost": o.action.cost,
            }
            for o in observations[-40:]
        ]
        top_cands = [
            {
                "kind": s.action.kind,
                "args": list(s.action.args),
                "cost": s.expected_cost,
                "efe": round(s.efe, 4),
                "law_eig_bits": round(s.law_eig_bits, 4),
            }
            for s in scores[:6]
        ]
        sys_override = self.harness.prompts.get("advisor")
        messages = build_advisor_messages(
            symbols=view.symbols,
            law_probs=law_probs,
            observations=obs_dicts,
            top_candidates=top_cands,
            memory_activations=activated_assemblies,
            budget_remaining=view.remaining,
            noise=view.noise,
            system_prompt=sys_override,
        )
        try:
            raw = self.client.chat_json(messages)
            suggested_spec, suggested_claims, note = parse_advisor_reply(raw)
        except (ModelClientError, ValueError):
            return (None, None, None)

        legal = {_action_key(s.action): s.action for s in scores}
        picked = _pick_from_tool_return(suggested_spec, legal) if suggested_spec else None
        return (picked, suggested_claims, note)

    def run_episode(
        self,
        world: HiddenLawWorld | WorldView,
        *,
        learn: bool = True,
        episode_idx: int = 0,
    ) -> CognitiveEpisodeResult:
        """Run one active-inference episode across the restricted Markov blanket."""
        t0 = time.monotonic()
        seed: int | None = world.seed if isinstance(world, HiddenLawWorld) else None
        view: WorldView = RestrictedWorld(world) if isinstance(world, HiddenLawWorld) else world

        symbols = view.symbols
        n = len(symbols)
        initial_budget = view.budget
        self.backend.reset(
            symbols,
            noise=view.noise,
            combine_cost=view.combine_cost,
            transform_cost=view.transform_cost,
        )

        # Initial memory recall (soft prior over valid profiles)
        recall_res = self.memory.recall(
            self.backend.belief(), self.backend.feature_summary(), n
        )
        if recall_res.activations:
            self.backend.set_prior(recall_res.prior_joint)

        observations: list[Observation] = []
        llm_calls = 0
        sandbox_errors = 0
        transfer_hints = 0
        transfer_source: str | None = None
        transfer_confidence = 0.0
        llm_claims_hint: dict[str, bool] | None = None
        llm_notes: str | None = None
        stopped_early = False
        stop_reason = "max_steps"

        mesh = self.harness.mesh
        for step in range(mesh.max_steps_per_episode):
            scores = self.backend.score_actions(view.remaining)
            should_stop, reason = self.backend.should_stop(scores)
            if should_stop:
                stopped_early = reason != "budget_exhausted"
                stop_reason = reason
                break

            cur_belief = self.backend.belief()
            chosen: ExperimentAction | None = None

            # Optional Phase 2 Hofstadter transfer seam
            if self.transfer is not None:
                hint = self.transfer.propose_transfer(
                    cur_belief, observations, self.memory.assemblies
                )
                if hint is not None:
                    if hint.prior_joint is not None:
                        self.backend.set_prior(hint.prior_joint)
                    if hint.source_assembly_id is not None and hint.confidence > 0.0:
                        transfer_hints += 1
                        if hint.confidence >= transfer_confidence:
                            transfer_source = hint.source_assembly_id
                            transfer_confidence = float(hint.confidence)
                    legal = {_action_key(s.action): s.action for s in scores}
                    for cand in hint.suggested_actions:
                        if _action_key(cand) in legal:
                            chosen = legal[_action_key(cand)]
                            break

            # Optional sandboxed harness/tools/strategy.py hook
            if chosen is None and self.harness.tool_source and mesh.use_tool_hook:
                tool_pick, had_err = self._consult_tool(view, cur_belief.law_probs, scores)
                if had_err:
                    sandbox_errors += 1
                elif tool_pick is not None:
                    chosen = tool_pick

            # Optional Cortex LLM advisor (only when uncertainty exceeds mesh threshold)
            if (
                chosen is None
                and self.client is not None
                and mesh.use_llm_advisor
                and llm_calls < mesh.max_llm_calls_per_episode
                and cur_belief.entropy_bits >= mesh.llm_entropy_trigger_bits
            ):
                llm_calls += 1
                llm_pick, hint_claims, note = self._consult_llm_advisor(
                    view, observations, cur_belief.law_probs, scores, recall_res.activations
                )
                if hint_claims is not None:
                    llm_claims_hint = hint_claims
                if note:
                    llm_notes = note
                if llm_pick is not None:
                    chosen = llm_pick

            # Default: Friston EFE action selection
            if chosen is None:
                if isinstance(self.backend, ExactBayesianBackend):
                    chosen = self.backend.select_action(scores, self._rng)
                else:
                    chosen = scores[0].action

            try:
                result_sym = chosen.execute(view)
            except BudgetExhausted:
                stop_reason = "budget_exhausted"
                break

            surprise = self.backend.observe(chosen, result_sym)
            observations.append(Observation(step=step, action=chosen, result=result_sym, surprise=surprise))

            # Periodically refresh probabilistic memory recall as evidence accumulates
            if self.memory.assemblies and (step + 1) % mesh.memory_refresh_interval == 0:
                recall_res = self.memory.recall(
                    self.backend.belief(), self.backend.feature_summary(), n
                )
                self.backend.set_prior(recall_res.prior_joint)

        final_belief = self.backend.belief()
        final_features = self.backend.feature_summary()
        recall_final = self.memory.recall(final_belief, final_features, n)

        # Final claims from Bayesian posterior, optionally blended with LLM hint
        claims = dict(final_belief.map_claims)
        if llm_claims_hint is not None and mesh.llm_blend_weight > 0.0:
            w = mesh.llm_blend_weight
            for name in LAW_NAMES:
                p_core = final_belief.law_probs[name]
                p_llm = 1.0 if llm_claims_hint[name] else 0.0
                p_blend = (1.0 - w) * p_core + w * p_llm
                claims[name] = p_blend > 0.5
            # Preserve algebraic tautologies even after LLM blend
            if claims["has_inverses"] and not claims["has_identity"]:
                claims["has_inverses"] = False
            if claims["f_involution"] and not claims["f_bijective"]:
                claims["f_involution"] = False
            if claims["idempotent"] and claims["has_inverses"]:
                if final_belief.law_probs["idempotent"] >= final_belief.law_probs["has_inverses"]:
                    claims["has_inverses"] = False
                else:
                    claims["idempotent"] = False

        mdl_gain = recall_final.compression_gain_bits
        if learn:
            report = self.memory.consolidate(
                final_belief, final_features, n, episode_idx=episode_idx
            )
            mdl_gain = max(mdl_gain, report.compression_gain_bits)

        mean_surp, max_surp = self.backend.surprise_stats()
        budget_spent = view.spent
        budget_frac = budget_spent / initial_budget if initial_budget > 0 else 0.0
        metrics = EssentialMetrics(
            mean_confidence=final_belief.mean_confidence,
            final_entropy_bits=final_belief.entropy_bits,
            budget_spent=budget_spent,
            budget_fraction=budget_frac,
            observations_used=len(observations),
            mean_surprise=mean_surp,
            max_surprise=max_surp,
            mdl_compression_bits=mdl_gain,
            llm_calls=llm_calls,
            sandbox_errors=sandbox_errors,
        )

        ashby_triggered = False
        ashby_violations: list[str] = []
        if self.monitor is not None:
            ashby_triggered = bool(self.monitor.record(metrics))
            ashby_violations = list(self.monitor.violations())

        latency_s = time.monotonic() - t0
        if self.event_log is not None:
            self.event_log.append(
                "cognitive_episode",
                {
                    "seed": seed,
                    "episode_idx": episode_idx,
                    "n": n,
                    "observations_used": len(observations),
                    "budget_spent": round(budget_spent, 3),
                    "budget_fraction": round(budget_frac, 4),
                    "mean_confidence": round(metrics.mean_confidence, 4),
                    "final_entropy_bits": round(metrics.final_entropy_bits, 4),
                    "mean_surprise": round(mean_surp, 4),
                    "max_surprise": round(max_surp, 4),
                    "mdl_compression_bits": round(mdl_gain, 4),
                    "n_assemblies": len(self.memory.assemblies),
                    "stopped_early": stopped_early,
                    "stop_reason": stop_reason,
                    "llm_calls": llm_calls,
                    "sandbox_errors": sandbox_errors,
                    "transfer_hints": transfer_hints,
                    "transfer_source": transfer_source,
                    "transfer_confidence": round(transfer_confidence, 4),
                    "ashby_triggered": ashby_triggered,
                    "learn": learn,
                },
            )

        return CognitiveEpisodeResult(
            seed=seed,
            n=n,
            claims=claims,
            law_probs=dict(final_belief.law_probs),
            observations_used=len(observations),
            budget_spent=budget_spent,
            final_entropy_bits=final_belief.entropy_bits,
            metrics=metrics,
            activated_assemblies=dict(recall_final.activations),
            stopped_early=stopped_early,
            stop_reason=stop_reason,
            latency_s=latency_s,
            llm_notes=llm_notes,
            transfer_hints=transfer_hints,
            transfer_source=transfer_source,
            transfer_confidence=round(transfer_confidence, 4),
            ashby_triggered=ashby_triggered,
            ashby_violations=ashby_violations,
        )

    def evaluate_audit_seeds(
        self,
        seeds: Iterable[int],
        *,
        noise: float = 0.05,
        budget: float = 500.0,
    ) -> dict[str, dict[str, bool]]:
        """Evaluate frozen audit seeds in read-only memory mode (``learn=False``).

        Returns the exact ``{str(seed): {law_name: bool}}`` dict expected by
        ``audit_claims.json`` and the workstation Governor.
        """
        claims_out: dict[str, dict[str, bool]] = {}
        for idx, seed in enumerate(seeds):
            world = HiddenLawWorld(seed=seed, noise=noise, budget=budget)
            res = self.run_episode(world, learn=False, episode_idx=idx)
            claims_out[str(seed)] = {k: bool(res.claims[k]) for k in LAW_NAMES}
        return claims_out
