"""Bake-off runner: 200-episode loop with MLflow logging and Drive checkpoint.

Each run:
  1. Creates (or resumes from) a JSON checkpoint on Drive.
  2. Runs up to ``config.n_episodes`` Hidden-Law Lab episodes.
  3. Logs every episode result to an MLflow file-store experiment.
  4. At the end, logs aggregate metrics and a per-family accuracy table.

MLflow is a runtime dependency (installed on Colab); it is imported lazily
so the module can be imported and tested without it.

Checkpoint format::

    {
      "model_name": "qwen3-8b",
      "config": { ... },
      "next_episode": 42,
      "results": { "0": {...}, "1": {...}, ... }
    }
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..harness.client import ModelClient
from ..harness.episode import EpisodeResult, run_episode
from ..lab.laws import LAW_NAMES
from ..lab.world import HiddenLawWorld
from .models import MODEL_REGISTRY, ModelSpec

# mlflow is imported lazily
_mlflow: Any = None


def _get_mlflow():
    global _mlflow
    if _mlflow is None:
        try:
            import mlflow as _mod
            _mlflow = _mod
        except ImportError as exc:
            raise ImportError(
                "The 'mlflow' package is required to use BakeoffRunner. "
                "Install it with:  pip install mlflow"
            ) from exc
    return _mlflow


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class BakeoffConfig:
    """Configuration for one bake-off run."""

    model_name: str                    # key in MODEL_REGISTRY
    n_episodes: int = 200
    seeds: list[int] = field(default_factory=lambda: list(range(200)))
    noise: float = 0.05
    budget: float = 500.0
    max_obs_per_episode: int = 200
    # vLLM server settings
    vllm_base_url: str = "http://localhost:8000/v1"
    temperature: float = 0.0
    max_tokens: int = 2048
    # MLflow
    mlflow_tracking_uri: str = "mlruns"   # file-store path
    mlflow_experiment: str = "phase0-bakeoff"
    # Checkpoint path (Drive path on Colab)
    checkpoint_path: str | None = None

    def model_spec(self) -> ModelSpec:
        if self.model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model {self.model_name!r}. "
                f"Known: {sorted(MODEL_REGISTRY)}"
            )
        return MODEL_REGISTRY[self.model_name]


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

@dataclass
class BakeoffSummary:
    model_name: str
    n_episodes: int
    n_errors: int
    accuracy: float                       # fraction of (episode × law) correct
    per_law_accuracy: dict[str, float]
    per_family_accuracy: dict[str, float]
    mean_latency_s: float
    results: list[EpisodeResult]

    def to_dict(self) -> dict[str, Any]:
        d = {
            "model_name": self.model_name,
            "n_episodes": self.n_episodes,
            "n_errors": self.n_errors,
            "accuracy": self.accuracy,
            "per_law_accuracy": self.per_law_accuracy,
            "per_family_accuracy": self.per_family_accuracy,
            "mean_latency_s": self.mean_latency_s,
        }
        return d


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_checkpoint(path: Path, model_name: str) -> dict:
    if path.exists():
        state = json.loads(path.read_text())
        if state.get("model_name") == model_name:
            print(
                f"Checkpoint resumed: model={model_name}, "
                f"episode {state['next_episode']}, "
                f"{len(state['results'])} results so far"
            )
            return state
        print(
            f"Checkpoint is for model={state.get('model_name')!r}, "
            f"not {model_name!r} — starting fresh."
        )
    return {"model_name": model_name, "next_episode": 0, "results": {}}


def _save_checkpoint(state: dict, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class BakeoffRunner:
    """Orchestrates the 200-episode bake-off for one model.

    Usage::

        config = BakeoffConfig(model_name="qwen3-8b",
                               checkpoint_path="/content/drive/.../bakeoff_qwen3-8b.json")
        runner = BakeoffRunner(config)
        summary = runner.run()
    """

    def __init__(self, config: BakeoffConfig) -> None:
        self.config = config
        self.spec = config.model_spec()

    def _make_client(self) -> ModelClient:
        return ModelClient(
            base_url=self.config.vllm_base_url,
            model_id=self.spec.hf_id,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            extra_body=dict(self.spec.extra_body),
        )

    def run(self, *, on_episode: Any = None) -> BakeoffSummary:
        """Run the bake-off.

        Args:
            on_episode: Optional callable ``(episode_idx, result)`` called
                        after each episode (useful for Colab progress display).

        Returns:
            A :class:`BakeoffSummary` with all episode results.
        """
        cfg = self.config
        client = self._make_client()

        # Checkpoint
        ckpt_path = Path(cfg.checkpoint_path) if cfg.checkpoint_path else None
        state = (
            _load_checkpoint(ckpt_path, cfg.model_name)
            if ckpt_path else
            {"model_name": cfg.model_name, "next_episode": 0, "results": {}}
        )

        # MLflow setup
        mlflow = _get_mlflow()
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
        mlflow.set_experiment(cfg.mlflow_experiment)

        all_results: list[EpisodeResult] = [
            # Replay already-completed results (for summary only)
            EpisodeResult(**r) for r in state["results"].values()
        ]

        run_tags = {
            "model_name": cfg.model_name,
            "hf_id": self.spec.hf_id,
            "revision": str(self.spec.revision),
            "t4_feasible": str(self.spec.t4_feasible),
        }

        with mlflow.start_run(run_name=cfg.model_name, tags=run_tags):
            mlflow.log_params({
                "model_name": cfg.model_name,
                "hf_id": self.spec.hf_id,
                "n_episodes": cfg.n_episodes,
                "noise": cfg.noise,
                "budget": cfg.budget,
                "temperature": cfg.temperature,
            })

            for episode_idx in range(state["next_episode"], cfg.n_episodes):
                seed = cfg.seeds[episode_idx]
                world = HiddenLawWorld(seed=seed, noise=cfg.noise, budget=cfg.budget)

                result = run_episode(
                    world, client, max_obs=cfg.max_obs_per_episode
                )
                all_results.append(result)
                state["results"][str(seed)] = result.to_dict()
                state["next_episode"] = episode_idx + 1

                if ckpt_path:
                    _save_checkpoint(state, ckpt_path)

                # Per-episode MLflow metrics
                mlflow.log_metrics(
                    {
                        "episode_accuracy": result.accuracy,
                        "episode_latency_s": result.latency_s,
                        "episode_correct": result.correct,
                        "episode_error": int(result.error is not None),
                    },
                    step=episode_idx,
                )

                if on_episode is not None:
                    on_episode(episode_idx, result)
                elif (episode_idx + 1) % 20 == 0 or episode_idx == cfg.n_episodes - 1:
                    done = episode_idx + 1
                    running_acc = sum(r.accuracy for r in all_results) / len(all_results)
                    print(
                        f"  Episode {done:4d}/{cfg.n_episodes}  "
                        f"running accuracy={running_acc:.3f}"
                    )

            # Aggregate metrics
            summary = _summarise(cfg.model_name, all_results)

            mlflow.log_metrics({
                "accuracy": summary.accuracy,
                "mean_latency_s": summary.mean_latency_s,
                "n_errors": summary.n_errors,
                **{f"accuracy_law_{k}": v for k, v in summary.per_law_accuracy.items()},
                **{f"accuracy_family_{k}": v
                   for k, v in summary.per_family_accuracy.items()},
            })

            # Persist full results as an artifact
            artifact_path = "results.json"
            results_json = json.dumps(
                [r.to_dict() for r in all_results], indent=2, sort_keys=True
            )
            # Write to a temp file so mlflow can log it
            import tempfile, os
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as tmp:
                tmp.write(results_json)
                tmp_name = tmp.name
            try:
                mlflow.log_artifact(tmp_name, artifact_path="")
            finally:
                os.unlink(tmp_name)

        return summary


# ---------------------------------------------------------------------------
# Summarise helper
# ---------------------------------------------------------------------------

def _summarise(model_name: str, results: list[EpisodeResult]) -> BakeoffSummary:
    if not results:
        return BakeoffSummary(
            model_name=model_name,
            n_episodes=0,
            n_errors=0,
            accuracy=0.0,
            per_law_accuracy={},
            per_family_accuracy={},
            mean_latency_s=0.0,
            results=[],
        )

    n = len(results)
    n_errors = sum(1 for r in results if r.error is not None)
    accuracy = sum(r.accuracy for r in results) / n

    per_law: dict[str, list[bool]] = {name: [] for name in LAW_NAMES}
    per_family: dict[str, list[float]] = {}
    latencies = []

    for r in results:
        for name in LAW_NAMES:
            claimed = r.claims.get(name)
            truth = r.truth.get(name)
            per_law[name].append(claimed == truth)
        per_family.setdefault(r.family, []).append(r.accuracy)
        latencies.append(r.latency_s)

    per_law_accuracy = {
        name: sum(vals) / len(vals) if vals else 0.0
        for name, vals in per_law.items()
    }
    per_family_accuracy = {
        fam: sum(vals) / len(vals)
        for fam, vals in per_family.items()
    }
    mean_latency_s = sum(latencies) / len(latencies) if latencies else 0.0

    return BakeoffSummary(
        model_name=model_name,
        n_episodes=n,
        n_errors=n_errors,
        accuracy=accuracy,
        per_law_accuracy=per_law_accuracy,
        per_family_accuracy=per_family_accuracy,
        mean_latency_s=mean_latency_s,
        results=results,
    )
