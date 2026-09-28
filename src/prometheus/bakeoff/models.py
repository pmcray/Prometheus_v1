"""Candidate model registry for the Phase 0 bake-off.

Each entry records:
  - The Hugging Face model ID (used as the vLLM ``--model`` argument).
  - The commit hash to pin (``--revision``), ensuring exact reproducibility.
    Set to ``None`` during the bake-off scan; pin after the winner is chosen.
  - Suggested vLLM launch flags.
  - Whether the model fits on a single free Colab T4 (15 GB VRAM).
  - A brief rationale note.

The CLAUDE.md design doc lists the lead candidates:
  cortex  – Qwen3-8B (primary) / gpt-oss-20b (T4 fallback)
  prover  – Goedel-Prover-V2-8B

The bake-off evaluates only the *cortex* role (law classification from noisy
experiments).  The prover role (Lean 4 proof generation) is a separate
evaluation task not covered in Phase 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    """Specification for one candidate model."""

    # Human-friendly short name used as the MLflow run name and in log files.
    name: str
    # Hugging Face model id (passed to ``vllm serve --model``).
    hf_id: str
    # HF commit hash to pin for reproducibility.  ``None`` = float to HEAD
    # during scanning; must be pinned before a production run.
    revision: str | None
    # Extra flags for ``vllm serve`` (list of strings, appended verbatim).
    vllm_flags: tuple[str, ...] = field(default_factory=tuple)
    # Extra body parameters sent with every chat-completions request.
    extra_body: dict[str, Any] = field(default_factory=dict)
    # Does this model fit on a single free Colab T4 (≤15 GB VRAM)?
    t4_feasible: bool = True
    # Short rationale / notes.
    notes: str = ""


MODEL_REGISTRY: dict[str, ModelSpec] = {
    # ── Primary cortex candidate ─────────────────────────────────────────
    "qwen3-8b": ModelSpec(
        name="qwen3-8b",
        hf_id="Qwen/Qwen3-8B",
        revision=None,   # pin to HEAD during scan; pin hash after winner chosen
        vllm_flags=(
            "--max-model-len", "4096",
            "--gpu-memory-utilization", "0.90",
            "--dtype", "auto",
        ),
        # Disable thinking mode for classification tasks (faster, cheaper).
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        t4_feasible=True,
        notes=(
            "Primary cortex candidate. 8B dense, ~16 GB FP16 / ~5 GB Q4. "
            "Fits T4 at BF16 with --max-model-len 4096. "
            "Strong reasoning, Apache 2.0."
        ),
    ),
    # ── T4 fallback cortex ───────────────────────────────────────────────
    "gpt-oss-20b": ModelSpec(
        name="gpt-oss-20b",
        hf_id="openai/gpt-oss-20b",
        revision=None,
        vllm_flags=(
            "--max-model-len", "4096",
            "--gpu-memory-utilization", "0.92",
            "--dtype", "auto",
        ),
        extra_body={},
        t4_feasible=True,
        notes=(
            "T4 fallback. MoE, ~16 GB natively quantised MXFP4. "
            "Fits T4. o3-mini level on reasoning benchmarks. "
            "Apache 2.0. Requires vLLM ≥ 0.9."
        ),
    ),
    # ── Alternative cortex ───────────────────────────────────────────────
    "gemma4-4b": ModelSpec(
        name="gemma4-4b",
        hf_id="google/gemma-4-4b-it",
        revision=None,
        vllm_flags=(
            "--max-model-len", "4096",
            "--gpu-memory-utilization", "0.90",
            "--dtype", "auto",
        ),
        extra_body={},
        t4_feasible=True,
        notes=(
            "Lightweight alternative. 4B dense, easily fits T4. "
            "Apache 2.0 (Gemma Terms). Lower capacity but fast."
        ),
    ),
    # ── Prover candidate (reference only; not evaluated in law classification) ──
    "goedel-prover-v2-8b": ModelSpec(
        name="goedel-prover-v2-8b",
        hf_id="Goedel-LM/Goedel-Prover-V2-8B",
        revision=None,
        vllm_flags=(
            "--max-model-len", "4096",
            "--gpu-memory-utilization", "0.90",
            "--dtype", "auto",
        ),
        extra_body={},
        t4_feasible=True,
        notes=(
            "Lean 4 prover role (Phase 0 step 5b, separate evaluation). "
            "84.6% pass@32 on MiniF2F. "
            "Listed here for registry completeness; not part of the law-"
            "classification bake-off."
        ),
    ),
}

# Ordered list for the bake-off: cortex candidates only.
BAKEOFF_MODELS: tuple[str, ...] = ("qwen3-8b", "gpt-oss-20b", "gemma4-4b")
