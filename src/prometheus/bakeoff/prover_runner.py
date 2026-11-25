"""Phase 3 Lean 4 Prover Bake-Off Runner (Goedel-Prover-V2-8B & cortex comparison).

Runs resumable, Drive-checkpointed Lean 4 theorem-proving benchmarks with:
- Separate Lean 4.9.0 toolchain support (``lean_compat="4.9"`` +
  ``lean/PrometheusProver49``) kept completely isolated from the ``v4.34.1``
  ``decide`` ground-truth verification project (``lean/PrometheusLaws``).
- ``pass@k`` sampling (``attempts_per_world >= 1``).
- Strict anti-cheat validation on every attempt (never weakens
  :func:`validate_prover_source`).
- Optional :class:`Gate` heartbeat checks so the workstation kill switch halts
  long prover runs cleanly.
- Structured per-family pass rates, error breakdown, and Observatory-compatible
  JSON checkpoints (``prover_<model>.json``).
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..harness.client import ModelClient, ModelClientError
from ..harness.prover import ProverResult, run_prover_episode
from ..lab.lean_layer import PROVER_LEAN_TOOLCHAIN, prover_lean_project_dir
from ..lab.world import HiddenLawWorld
from ..runtime.gate import Gate
from .models import MODEL_REGISTRY, ModelSpec

PROVER_BAKEOFF_MODELS: tuple[str, ...] = ("goedel-prover-v2-8b", "qwen3-8b")


@dataclass
class ProverBakeoffConfig:
    """Configuration for one Phase 3 Lean 4 prover bake-off run."""

    model_name: str = "goedel-prover-v2-8b"
    n_worlds: int = 50
    seeds: list[int] | None = None
    attempts_per_world: int = 1
    temperature: float = 0.0
    sample_temperature: float = 0.6
    max_tokens: int = 4096
    lean_compat: str = "4.9"
    lean_toolchain: str = PROVER_LEAN_TOOLCHAIN
    lean_project_dir: str | None = None
    lean_timeout: int = 60
    vllm_base_url: str = "http://localhost:8000/v1"
    checkpoint_path: str | None = None
    hf_commit: str | None = None
    check_health: bool = True

    def __post_init__(self) -> None:
        if self.seeds is None:
            self.seeds = list(range(self.n_worlds))
        elif len(self.seeds) < self.n_worlds:
            raise ValueError(
                f"seeds length ({len(self.seeds)}) is smaller than n_worlds ({self.n_worlds})"
            )
        if self.lean_compat not in ("4.9", "4.34"):
            raise ValueError(f"unsupported lean_compat {self.lean_compat!r}")
        if self.attempts_per_world < 1:
            raise ValueError("attempts_per_world must be >= 1")

    def model_spec(self) -> ModelSpec:
        if self.model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model {self.model_name!r}. Known: {sorted(MODEL_REGISTRY)}"
            )
        return MODEL_REGISTRY[self.model_name]


@dataclass
class ProverBakeoffSummary:
    model_name: str
    hf_id: str
    hf_commit: str | None
    lean_toolchain: str
    lean_compat: str
    attempts_per_world: int
    n_worlds: int
    n_proved: int
    pass_rate: float
    mean_latency_s: float
    per_family_pass_rate: dict[str, float]
    error_breakdown: dict[str, int]
    results: list[ProverResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "hf_id": self.hf_id,
            "hf_commit": self.hf_commit,
            "lean_toolchain": self.lean_toolchain,
            "lean_compat": self.lean_compat,
            "attempts_per_world": self.attempts_per_world,
            "n_worlds": self.n_worlds,
            "n_proved": self.n_proved,
            "pass_rate": round(self.pass_rate, 4),
            "mean_latency_s": round(self.mean_latency_s, 3),
            "per_family_pass_rate": self.per_family_pass_rate,
            "error_breakdown": self.error_breakdown,
        }


def _save_atomic(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def summarise_prover_results(
    *,
    model_name: str,
    hf_id: str,
    hf_commit: str | None,
    lean_toolchain: str,
    lean_compat: str,
    attempts_per_world: int,
    results: list[ProverResult],
) -> ProverBakeoffSummary:
    n = len(results)
    if n == 0:
        return ProverBakeoffSummary(
            model_name=model_name,
            hf_id=hf_id,
            hf_commit=hf_commit,
            lean_toolchain=lean_toolchain,
            lean_compat=lean_compat,
            attempts_per_world=attempts_per_world,
            n_worlds=0,
            n_proved=0,
            pass_rate=0.0,
            mean_latency_s=0.0,
            per_family_pass_rate={},
            error_breakdown={},
            results=[],
        )

    n_proved = sum(1 for r in results if r.proved)
    mean_lat = sum(r.latency_s for r in results) / n
    fam_totals: dict[str, list[bool]] = {}
    err_counts: Counter[str] = Counter()
    for r in results:
        fam_totals.setdefault(r.family, []).append(r.proved)
        if not r.proved:
            err_counts[r.error_kind or "unknown"] += 1

    per_family = {
        fam: round(sum(vals) / len(vals), 4)
        for fam, vals in sorted(fam_totals.items())
    }
    return ProverBakeoffSummary(
        model_name=model_name,
        hf_id=hf_id,
        hf_commit=hf_commit,
        lean_toolchain=lean_toolchain,
        lean_compat=lean_compat,
        attempts_per_world=attempts_per_world,
        n_worlds=n,
        n_proved=n_proved,
        pass_rate=n_proved / n,
        mean_latency_s=mean_lat,
        per_family_pass_rate=per_family,
        error_breakdown=dict(sorted(err_counts.items())),
        results=results,
    )


class ProverBakeoffRunner:
    """Resumable Lean 4 prover bake-off runner (supports Lean 4.9 & pass@k)."""

    def __init__(
        self,
        config: ProverBakeoffConfig,
        *,
        client: ModelClient | None = None,
        gate: Gate | None = None,
    ) -> None:
        self.config = config
        self.spec = config.model_spec()
        self._client = client
        self.gate = gate

    def _make_client(self) -> ModelClient:
        if self._client is not None:
            return self._client
        return ModelClient(
            base_url=self.config.vllm_base_url,
            model_id=self.spec.hf_id,
            temperature=self.config.temperature,
            max_tokens=max(self.config.max_tokens, self.spec.default_max_tokens),
            extra_body=dict(self.spec.extra_body),
        )

    def _resolve_project_dir(self) -> Path | None:
        if self.config.lean_project_dir:
            return Path(self.config.lean_project_dir)
        if self.config.lean_compat == "4.9":
            return prover_lean_project_dir()
        return None

    def run(self, *, on_world: Any = None) -> ProverBakeoffSummary:
        cfg = self.config
        client = self._make_client()
        proj_dir = self._resolve_project_dir()

        ckpt_path = Path(cfg.checkpoint_path) if cfg.checkpoint_path else None
        if ckpt_path and ckpt_path.is_file():
            state = json.loads(ckpt_path.read_text(encoding="utf-8"))
            if state.get("model_name") != cfg.model_name:
                state = None
        else:
            state = None

        if state is None:
            state = {
                "model_name": cfg.model_name,
                "hf_id": self.spec.hf_id,
                "hf_commit": cfg.hf_commit or self.spec.revision,
                "lean_toolchain": cfg.lean_toolchain,
                "lean_compat": cfg.lean_compat,
                "attempts_per_world": cfg.attempts_per_world,
                "results": {},
            }

        assert cfg.seeds is not None
        target_seeds = cfg.seeds[: cfg.n_worlds]
        remaining_seeds = [s for s in target_seeds if str(s) not in state["results"]]

        if remaining_seeds and cfg.check_health:
            if not client.health_check():
                raise ModelClientError(
                    f"vLLM server at {cfg.vllm_base_url} failed health check for "
                    f"prover model {self.spec.hf_id!r}."
                )

        for idx, seed in enumerate(target_seeds, start=1):
            key = str(seed)
            if key in state["results"]:
                continue
            if self.gate is not None:
                self.gate.check_heartbeat()
            world = HiddenLawWorld(seed=seed)
            res = run_prover_episode(
                world,
                client,
                lean_timeout=cfg.lean_timeout,
                lean_compat=cfg.lean_compat,
                project_dir=proj_dir,
                attempts=cfg.attempts_per_world,
                sample_temperature=cfg.sample_temperature,
            )
            state["results"][key] = res.to_dict()
            if ckpt_path is not None:
                _save_atomic(ckpt_path, state)
            if on_world is not None:
                on_world(idx, res)

        ordered_results: list[ProverResult] = []
        for s in target_seeds:
            raw = state["results"].get(str(s))
            if isinstance(raw, dict):
                ordered_results.append(
                    ProverResult(
                        seed=int(raw["seed"]),
                        family=str(raw["family"]),
                        proved=bool(raw["proved"]),
                        latency_s=float(raw.get("latency_s", 0.0)),
                        error=raw.get("error"),
                        attempts=int(raw.get("attempts", 1)),
                        winning_attempt=raw.get("winning_attempt"),
                        error_kind=raw.get("error_kind"),
                        lean_compat=str(raw.get("lean_compat", cfg.lean_compat)),
                    )
                )

        summary = summarise_prover_results(
            model_name=cfg.model_name,
            hf_id=self.spec.hf_id,
            hf_commit=state.get("hf_commit"),
            lean_toolchain=str(state.get("lean_toolchain", cfg.lean_toolchain)),
            lean_compat=str(state.get("lean_compat", cfg.lean_compat)),
            attempts_per_world=cfg.attempts_per_world,
            results=ordered_results,
        )
        if ckpt_path is not None:
            state["summary"] = summary.to_dict()
            _save_atomic(ckpt_path, state)
        return summary
