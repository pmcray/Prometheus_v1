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
    # Extra flags for ``vllm serve`` on T4 (15 GB, SM75) — backwards-compatible default.
    vllm_flags: tuple[str, ...] = field(default_factory=tuple)
    # Extra flags for ``vllm serve`` on L4 / A100 / G4 (>=20 GB, SM>=8.0, native BF16).
    bf16_vllm_flags: tuple[str, ...] = field(default_factory=tuple)
    # Extra body parameters sent with every chat-completions request.
    extra_body: dict[str, Any] = field(default_factory=dict)
    # Does this model fit on a single free Colab T4 (≤15 GB VRAM)?
    t4_feasible: bool = True
    # Recommended max_tokens for bake-off completions (higher for reasoning models).
    default_max_tokens: int = 2048
    # Short rationale / notes.
    notes: str = ""


MODEL_REGISTRY: dict[str, ModelSpec] = {
    # ── Primary cortex candidate ─────────────────────────────────────────
    "qwen3-8b": ModelSpec(
        name="qwen3-8b",
        hf_id="Qwen/Qwen3-8B",
        revision="b968826d9c46dd6066d109eabc6255188de91218",  # pinned from Phase 0 Colab bake-off (2026-11-23)
        vllm_flags=(
            "--max-model-len", "4096",
            "--gpu-memory-utilization", "0.90",
            # T4 (Turing SM75, 15 GB) lacks native BF16 and cannot hold ~16.4 GB
            # FP16 weights + KV cache; use FP16 compute with 4-bit bitsandbytes.
            "--dtype", "half",
            "--quantization", "bitsandbytes",
            "--load-format", "bitsandbytes",
        ),
        bf16_vllm_flags=(
            "--max-model-len", "4096",
            "--gpu-memory-utilization", "0.90",
            "--dtype", "bfloat16",
            "--enforce-eager",
        ),
        # Disable thinking mode for classification tasks (faster, cheaper).
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        t4_feasible=True,
        default_max_tokens=2048,
        notes=(
            "Primary cortex candidate. 8B dense, ~16.4 GB BF16 / ~5.5 GB 4-bit. "
            "Runs unquantized BF16 on L4/A100/G4; 4-bit bitsandbytes on T4. "
            "Strong reasoning, Apache 2.0."
        ),
    ),
    # ── MoE reasoning cortex ─────────────────────────────────────────────
    "gpt-oss-20b": ModelSpec(
        name="gpt-oss-20b",
        hf_id="openai/gpt-oss-20b",
        revision=None,
        vllm_flags=(
            "--max-model-len", "4096",
            "--gpu-memory-utilization", "0.92",
            "--dtype", "half",
            "--enforce-eager",
            "--reasoning-parser", "openai_gptoss",
        ),
        bf16_vllm_flags=(
            "--max-model-len", "8192",
            "--gpu-memory-utilization", "0.90",
            "--enforce-eager",
            "--reasoning-parser", "openai_gptoss",
        ),
        extra_body={},
        t4_feasible=True,
        default_max_tokens=4096,
        notes=(
            "20B MoE (~13.8 GB native MXFP4 weights). Recommended on L4/A100/G4 "
            "(tight on T4 15 GB). Uses Harmony reasoning parser. Apache 2.0."
        ),
    ),
    # ── Compact Gemma 4 alternative cortex ───────────────────────────────
    "gemma4-4b": ModelSpec(
        name="gemma4-4b",
        hf_id="google/gemma-4-E4B-it",
        revision=None,
        vllm_flags=(
            "--max-model-len", "4096",
            "--gpu-memory-utilization", "0.90",
            # T4 (SM75) does not support BF16; --dtype half avoids vLLM startup failure.
            "--dtype", "half",
            "--enforce-eager",
            "--reasoning-parser", "gemma4",
        ),
        bf16_vllm_flags=(
            "--max-model-len", "4096",
            "--gpu-memory-utilization", "0.90",
            "--dtype", "bfloat16",
            "--enforce-eager",
            "--reasoning-parser", "gemma4",
        ),
        extra_body={},
        t4_feasible=True,
        default_max_tokens=4096,
        notes=(
            "Gemma 4 E4B instruction-tuned (~8B total with per-layer embeddings). "
            "Fits T4/L4/A100/G4. Note: gated on HF (set HF_TOKEN in Colab Secrets)."
        ),
    ),
    # ── Pro-GPU larger Gemma 4 candidates (L4/A100/G4 only) ──────────────
    "gemma4-26b-a4b": ModelSpec(
        name="gemma4-26b-a4b",
        hf_id="google/gemma-4-26B-A4B-it",
        revision=None,
        vllm_flags=(
            "--max-model-len", "4096",
            "--gpu-memory-utilization", "0.90",
            "--dtype", "bfloat16",
            "--enforce-eager",
            "--reasoning-parser", "gemma4",
        ),
        bf16_vllm_flags=(
            "--max-model-len", "8192",
            "--gpu-memory-utilization", "0.90",
            "--dtype", "bfloat16",
            "--enforce-eager",
            "--reasoning-parser", "gemma4",
        ),
        extra_body={},
        t4_feasible=False,
        default_max_tokens=4096,
        notes=(
            "Gemma 4 26B-A4B MoE (4B active params). Needs A100 (80 GB) or G4 (96 GB) "
            "in BF16. Gated on HF (set HF_TOKEN)."
        ),
    ),
    "gemma4-31b": ModelSpec(
        name="gemma4-31b",
        hf_id="google/gemma-4-31B-it",
        revision=None,
        vllm_flags=(
            "--max-model-len", "4096",
            "--gpu-memory-utilization", "0.90",
            "--dtype", "bfloat16",
            "--enforce-eager",
            "--reasoning-parser", "gemma4",
        ),
        bf16_vllm_flags=(
            "--max-model-len", "8192",
            "--gpu-memory-utilization", "0.90",
            "--dtype", "bfloat16",
            "--enforce-eager",
            "--reasoning-parser", "gemma4",
        ),
        extra_body={},
        t4_feasible=False,
        default_max_tokens=4096,
        notes=(
            "Gemma 4 31B dense (design-doc alternative). Needs A100 (80 GB) or "
            "G4 (96 GB) in BF16. Gated on HF (set HF_TOKEN)."
        ),
    ),
    # ── Prover candidate (Phase 0 step 5b) ───────────────────────────────
    "goedel-prover-v2-8b": ModelSpec(
        name="goedel-prover-v2-8b",
        hf_id="Goedel-LM/Goedel-Prover-V2-8B",
        revision=None,
        vllm_flags=(
            "--max-model-len", "4096",
            "--gpu-memory-utilization", "0.90",
            "--dtype", "half",
            "--quantization", "bitsandbytes",
            "--load-format", "bitsandbytes",
        ),
        bf16_vllm_flags=(
            "--max-model-len", "8192",
            "--gpu-memory-utilization", "0.90",
            "--dtype", "bfloat16",
            "--enforce-eager",
        ),
        extra_body={},
        t4_feasible=True,
        default_max_tokens=4096,
        notes=(
            "Lean 4 prover role (Phase 0 step 5b). 84.6% pass@32 on MiniF2F "
            "(trained on Lean 4.9). Unquantized BF16 on L4/A100/G4; 4-bit on T4."
        ),
    ),
}

# Ordered list for the baseline T4-feasible cortex bake-off.
BAKEOFF_MODELS: tuple[str, ...] = ("qwen3-8b", "gpt-oss-20b", "gemma4-4b")


def vllm_flags_for_gpu(
    spec: ModelSpec,
    *,
    compute_cap: tuple[int, int] | None = None,
    vram_gb: float | None = None,
) -> tuple[str, ...]:
    """Choose GPU-appropriate vLLM flags (BF16 on L4/A100/G4, T4 fallback otherwise).

    Automatically appends ``--revision <hash>`` when ``spec.revision`` is pinned.
    """
    use_bf16 = (
        bool(spec.bf16_vllm_flags)
        and compute_cap is not None
        and compute_cap >= (8, 0)
        and (vram_gb is None or vram_gb >= 20.0)
    )
    flags = list(spec.bf16_vllm_flags if use_bf16 else spec.vllm_flags)
    if spec.revision:
        flags.extend(["--revision", spec.revision])
    return tuple(flags)
