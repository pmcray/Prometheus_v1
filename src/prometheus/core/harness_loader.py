"""Strict immutable->mutable loader for governor-approved ``harness/`` artifacts.

Protected core code.  Reads ONLY from the five governor-allowed subdirectories
under an installed harness root (e.g. ``INSTALL_ROOT/current/harness``):

    prompts/         Markdown / text prompt overrides (e.g. ``advisor.md``)
    mesh/            Cognitive mesh routing config (``config.json``)
    memory_params/   EFE & memory parameters (``efe_params.json``, ``memory_params.json``)
    tools/           Optional agent-authored strategy scripts (sandboxed ONLY)
    evaluators/      Optional proxy-score evaluator scripts (sandboxed ONLY)

Security invariants:
  - Symlinks and path traversal outside the harness directory are rejected.
  - All numeric parameters reject booleans, NaN, Inf, and out-of-range values.
  - Scripts from ``tools/`` and ``evaluators/`` are NEVER imported or exec'd in
    the host/runtime process — they are executed solely via ``sandbox.run()``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..common.variant import BundleError, safe_relpath
from ..runtime.sandbox import SandboxResult, SandboxViolation, run as sandbox_run
from .inference import EFEParams
from .interfaces import AshbyConfig
from .memory import MemoryParams
from .transfer import TransferParams

ALLOWED_SUBDIRS: tuple[str, ...] = (
    "prompts",
    "mesh",
    "memory_params",
    "tools",
    "evaluators",
)
ALLOWED_SUFFIXES: tuple[str, ...] = (".py", ".json", ".yaml", ".yml", ".md", ".txt")
MAX_ARTIFACT_BYTES: int = 256_000


class HarnessValidationError(ValueError):
    """Raised when a mutable harness artifact fails validation."""


@dataclass(frozen=True)
class MeshConfig:
    """Routing parameters controlling how core, LLM advisor, and tools interact."""

    use_llm_advisor: bool = False
    llm_entropy_trigger_bits: float = 1.5
    llm_blend_weight: float = 0.25
    max_llm_calls_per_episode: int = 2
    use_tool_hook: bool = True
    max_steps_per_episode: int = 120
    memory_refresh_interval: int = 5


@dataclass(frozen=True)
class LoadedHarness:
    """Validated, immutable snapshot of the active mutable harness generation."""

    efe_params: EFEParams = EFEParams()
    memory_params: MemoryParams = MemoryParams()
    transfer_params: TransferParams = TransferParams()
    mesh: MeshConfig = MeshConfig()
    ashby_config: AshbyConfig = AshbyConfig()
    prompts: dict[str, str] = None  # type: ignore[assignment]
    tool_source: str | None = None
    evaluator_source: str | None = None
    source_dir: Path | None = None
    loaded_files: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.prompts is None:
            object.__setattr__(self, "prompts", {})


# ----- strict scalar validators ---------------------------------------------

def _check_float(
    data: dict[str, Any], key: str, default: float, *, lo: float, hi: float
) -> float:
    if key not in data:
        return default
    val = data[key]
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise HarnessValidationError(f"{key} must be a finite number, got {val!r}")
    f = float(val)
    if math.isnan(f) or math.isinf(f):
        raise HarnessValidationError(f"{key} must be finite, got {val!r}")
    if not (lo <= f <= hi):
        raise HarnessValidationError(f"{key}={f} outside allowed range [{lo}, {hi}]")
    return f


def _check_int(
    data: dict[str, Any], key: str, default: int, *, lo: int, hi: int
) -> int:
    if key not in data:
        return default
    val = data[key]
    if isinstance(val, bool) or not isinstance(val, int):
        raise HarnessValidationError(f"{key} must be an integer, got {val!r}")
    if not (lo <= val <= hi):
        raise HarnessValidationError(f"{key}={val} outside allowed range [{lo}, {hi}]")
    return val


def _check_bool(data: dict[str, Any], key: str, default: bool) -> bool:
    if key not in data:
        return default
    val = data[key]
    if not isinstance(val, bool):
        raise HarnessValidationError(f"{key} must be a boolean, got {val!r}")
    return val


def _check_violation_grid(
    data: dict[str, Any], default: tuple[float, ...]
) -> tuple[float, ...]:
    if "violation_grid" not in data:
        return default
    raw = data["violation_grid"]
    if not isinstance(raw, list) or not (1 <= len(raw) <= 16):
        raise HarnessValidationError("violation_grid must be a list of 1..16 floats in (0, 1]")
    out: list[float] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise HarnessValidationError(f"invalid violation_grid entry: {item!r}")
        f = float(item)
        if math.isnan(f) or math.isinf(f) or not (0.0 < f <= 1.0):
            raise HarnessValidationError(f"violation_grid entry {f} not in (0, 1]")
        out.append(f)
    return tuple(out)


def _validate_efe_params(data: dict[str, Any]) -> EFEParams:
    d = EFEParams()
    return EFEParams(
        beta_law=_check_float(data, "beta_law", d.beta_law, lo=0.0, hi=100.0),
        beta_cell=_check_float(data, "beta_cell", d.beta_cell, lo=0.0, hi=50.0),
        lambda_cost=_check_float(data, "lambda_cost", d.lambda_cost, lo=0.0, hi=50.0),
        action_precision=_check_float(data, "action_precision", d.action_precision, lo=0.0, hi=100.0),
        stop_entropy_bits=_check_float(data, "stop_entropy_bits", d.stop_entropy_bits, lo=0.0, hi=6.0),
        stop_eig_per_cost=_check_float(data, "stop_eig_per_cost", d.stop_eig_per_cost, lo=0.0, hi=5.0),
        min_observations_before_stop=_check_int(
            data, "min_observations_before_stop", d.min_observations_before_stop, lo=0, hi=100
        ),
        violation_grid=_check_violation_grid(data, d.violation_grid),
        noise_floor=_check_float(data, "noise_floor", d.noise_floor, lo=1e-9, hi=0.25),
        forgetting_factor=_check_float(data, "forgetting_factor", d.forgetting_factor, lo=0.1, hi=1.0),
        max_assoc_triples=_check_int(data, "max_assoc_triples", d.max_assoc_triples, lo=1, hi=1000),
        max_homo_pairs=_check_int(data, "max_homo_pairs", d.max_homo_pairs, lo=1, hi=200),
    )


def _validate_memory_params(data: dict[str, Any]) -> MemoryParams:
    d = MemoryParams()
    return MemoryParams(
        prior_blend=_check_float(data, "prior_blend", d.prior_blend, lo=0.0, hi=0.95),
        recall_temperature=_check_float(data, "recall_temperature", d.recall_temperature, lo=0.05, hi=50.0),
        spawn_mdl_threshold_bits=_check_float(
            data, "spawn_mdl_threshold_bits", d.spawn_mdl_threshold_bits, lo=0.0, hi=20.0
        ),
        merge_kl_threshold=_check_float(data, "merge_kl_threshold", d.merge_kl_threshold, lo=0.0, hi=5.0),
        hebbian_decay=_check_float(data, "hebbian_decay", d.hebbian_decay, lo=0.5, hi=1.0),
        max_assemblies=_check_int(data, "max_assemblies", d.max_assemblies, lo=1, hi=256),
        min_confidence_to_learn=_check_float(
            data, "min_confidence_to_learn", d.min_confidence_to_learn, lo=0.5, hi=0.999
        ),
        model_cost_bits=_check_float(data, "model_cost_bits", d.model_cost_bits, lo=0.0, hi=20.0),
    )


def _validate_mesh_config(data: dict[str, Any]) -> MeshConfig:
    d = MeshConfig()
    return MeshConfig(
        use_llm_advisor=_check_bool(data, "use_llm_advisor", d.use_llm_advisor),
        llm_entropy_trigger_bits=_check_float(
            data, "llm_entropy_trigger_bits", d.llm_entropy_trigger_bits, lo=0.0, hi=8.0
        ),
        llm_blend_weight=_check_float(data, "llm_blend_weight", d.llm_blend_weight, lo=0.0, hi=0.9),
        max_llm_calls_per_episode=_check_int(
            data, "max_llm_calls_per_episode", d.max_llm_calls_per_episode, lo=0, hi=20
        ),
        use_tool_hook=_check_bool(data, "use_tool_hook", d.use_tool_hook),
        max_steps_per_episode=_check_int(
            data, "max_steps_per_episode", d.max_steps_per_episode, lo=1, hi=500
        ),
        memory_refresh_interval=_check_int(
            data, "memory_refresh_interval", d.memory_refresh_interval, lo=1, hi=100
        ),
    )


def _validate_transfer_params(data: dict[str, Any]) -> TransferParams:
    d = TransferParams()
    return TransferParams(
        enabled=_check_bool(data, "enabled", d.enabled),
        min_observations=_check_int(data, "min_observations", d.min_observations, lo=1, hi=50),
        min_confidence=_check_float(data, "min_confidence", d.min_confidence, lo=0.05, hi=0.99),
        max_prior_blend=_check_float(data, "max_prior_blend", d.max_prior_blend, lo=0.0, hi=0.85),
        temperature=_check_float(data, "temperature", d.temperature, lo=0.1, hi=25.0),
        surprise_veto_threshold=_check_float(
            data, "surprise_veto_threshold", d.surprise_veto_threshold, lo=1.0, hi=20.0
        ),
        enable_probe_suggestions=_check_bool(
            data, "enable_probe_suggestions", d.enable_probe_suggestions
        ),
        max_probe_steps=_check_int(data, "max_probe_steps", d.max_probe_steps, lo=0, hi=40),
    )


_FORBIDDEN_ASHBY_KEYS = frozenset({
    "essential_bounds",
    "bounds",
    "min_confidence",
    "max_budget_fraction",
    "max_rolling_surprise",
    "max_sandbox_error_rate",
})


def _validate_ashby_config(data: dict[str, Any]) -> AshbyConfig:
    forbidden = sorted(k for k in data if k in _FORBIDDEN_ASHBY_KEYS)
    if forbidden:
        raise HarnessValidationError(
            f"mutable harness cannot modify protected essential-variable bounds: {forbidden}"
        )
    d = AshbyConfig()
    return AshbyConfig(
        window_size=_check_int(data, "window_size", d.window_size, lo=2, hi=50),
        min_episodes_before_trigger=_check_int(
            data, "min_episodes_before_trigger", d.min_episodes_before_trigger, lo=2, hi=50
        ),
        min_breach_episodes=_check_int(
            data, "min_breach_episodes", d.min_breach_episodes, lo=1, hi=20
        ),
        cooldown_episodes=_check_int(
            data, "cooldown_episodes", d.cooldown_episodes, lo=1, hi=100
        ),
        surprise_spike_threshold=_check_float(
            data, "surprise_spike_threshold", d.surprise_spike_threshold, lo=1.5, hi=20.0
        ),
    )


# ----- safe file reading & loader -------------------------------------------

def _read_safe_file(root: Path, rel: str) -> str:
    try:
        norm = safe_relpath(rel)
    except BundleError as exc:
        raise HarnessValidationError(str(exc)) from exc
    parts = norm.split("/")
    if not parts or parts[0] not in ALLOWED_SUBDIRS:
        raise HarnessValidationError(f"path outside allowed harness subdirs: {norm!r}")
    if not norm.endswith(ALLOWED_SUFFIXES):
        raise HarnessValidationError(f"disallowed file suffix: {norm!r}")

    cur = root
    for part in parts:
        cur = cur / part
        if cur.is_symlink():
            raise HarnessValidationError(f"symlink not allowed in harness: {norm!r}")
    resolved_root = root.resolve()
    resolved_target = cur.resolve()
    try:
        resolved_target.relative_to(resolved_root)
    except ValueError as exc:
        raise HarnessValidationError(f"path escapes harness root: {norm!r}") from exc
    if not cur.is_file():
        raise HarnessValidationError(f"not a regular file: {norm!r}")
    size = cur.stat().st_size
    if size > MAX_ARTIFACT_BYTES:
        raise HarnessValidationError(f"artifact exceeds {MAX_ARTIFACT_BYTES} bytes: {norm!r}")
    return cur.read_text(encoding="utf-8")


def _read_json_artifact(root: Path, rel: str) -> dict[str, Any]:
    raw = _read_safe_file(root, rel)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HarnessValidationError(f"{rel} is not valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise HarnessValidationError(f"{rel} must contain a JSON object")
    return obj


def load_harness(harness_dir: Path | str | None = None) -> LoadedHarness:
    """Load and validate mutable artifacts from *harness_dir*.

    Accepts either ``INSTALL_ROOT/current/harness`` directly or
    ``INSTALL_ROOT/current`` (automatically descending into ``harness/`` if
    present).  If *harness_dir* is ``None`` or does not exist, returns Gen-0
    built-in defaults.
    """
    if harness_dir is None:
        return LoadedHarness()
    root = Path(harness_dir)
    if not root.exists():
        return LoadedHarness()
    if (root / "harness").is_dir() and not any((root / sub).exists() for sub in ALLOWED_SUBDIRS):
        root = root / "harness"
    if root.is_symlink() or not root.is_dir():
        raise HarnessValidationError(f"invalid harness directory: {root}")

    loaded: list[str] = []
    # Validate no symlinks anywhere under the allowed subtrees
    for sub in ALLOWED_SUBDIRS:
        sub_dir = root / sub
        if sub_dir.is_symlink():
            raise HarnessValidationError(f"symlinked harness subdir not allowed: {sub}")

    efe_params = EFEParams()
    efe_path = root / "memory_params" / "efe_params.json"
    if efe_path.exists() or efe_path.is_symlink():
        efe_params = _validate_efe_params(_read_json_artifact(root, "memory_params/efe_params.json"))
        loaded.append("memory_params/efe_params.json")

    memory_params = MemoryParams()
    mem_path = root / "memory_params" / "memory_params.json"
    if mem_path.exists() or mem_path.is_symlink():
        memory_params = _validate_memory_params(
            _read_json_artifact(root, "memory_params/memory_params.json")
        )
        loaded.append("memory_params/memory_params.json")

    transfer_params = TransferParams()
    tr_path = root / "memory_params" / "transfer_params.json"
    if tr_path.exists() or tr_path.is_symlink():
        transfer_params = _validate_transfer_params(
            _read_json_artifact(root, "memory_params/transfer_params.json")
        )
        loaded.append("memory_params/transfer_params.json")

    mesh = MeshConfig()
    mesh_path = root / "mesh" / "config.json"
    if mesh_path.exists() or mesh_path.is_symlink():
        mesh = _validate_mesh_config(_read_json_artifact(root, "mesh/config.json"))
        loaded.append("mesh/config.json")

    ashby_config = AshbyConfig()
    ashby_path = root / "mesh" / "ashby_params.json"
    if ashby_path.exists() or ashby_path.is_symlink():
        ashby_config = _validate_ashby_config(_read_json_artifact(root, "mesh/ashby_params.json"))
        loaded.append("mesh/ashby_params.json")

    prompts: dict[str, str] = {}
    prompts_dir = root / "prompts"
    if prompts_dir.is_dir():
        for p in sorted(prompts_dir.iterdir()):
            if p.name.startswith("."):
                continue
            rel = f"prompts/{p.name}"
            prompts[p.stem] = _read_safe_file(root, rel)
            loaded.append(rel)

    tool_source: str | None = None
    tool_path = root / "tools" / "strategy.py"
    if tool_path.exists() or tool_path.is_symlink():
        tool_source = _read_safe_file(root, "tools/strategy.py")
        loaded.append("tools/strategy.py")

    evaluator_source: str | None = None
    eval_path = root / "evaluators" / "score.py"
    if eval_path.exists() or eval_path.is_symlink():
        evaluator_source = _read_safe_file(root, "evaluators/score.py")
        loaded.append("evaluators/score.py")

    return LoadedHarness(
        efe_params=efe_params,
        memory_params=memory_params,
        transfer_params=transfer_params,
        mesh=mesh,
        ashby_config=ashby_config,
        prompts=prompts,
        tool_source=tool_source,
        evaluator_source=evaluator_source,
        source_dir=root,
        loaded_files=tuple(sorted(loaded)),
    )


# ----- mandatory sandboxed execution for mutable Python artifacts -----------

def run_harness_tool(
    tool_source: str,
    inputs: dict[str, Any],
    *,
    cpu_seconds: int = 3,
    wall_seconds: float = 10.0,
    memory_mb: int = 128,
) -> SandboxResult:
    """Execute a mutable ``harness/tools/*.py`` script strictly inside the sandbox."""
    try:
        return sandbox_run(
            tool_source,
            inputs=inputs,
            cpu_seconds=cpu_seconds,
            wall_seconds=wall_seconds,
            memory_mb=memory_mb,
        )
    except SandboxViolation as exc:
        return SandboxResult(error=f"SandboxViolation: {exc}")


def run_harness_evaluator(
    evaluator_source: str,
    inputs: dict[str, Any],
    *,
    cpu_seconds: int = 5,
    wall_seconds: float = 15.0,
    memory_mb: int = 128,
) -> tuple[float | None, str | None]:
    """Execute a mutable ``harness/evaluators/*.py`` script inside the sandbox.

    Returns ``(proxy_score, error)``.  Non-finite or boolean return values are
    rejected cleanly.
    """
    try:
        res = sandbox_run(
            evaluator_source,
            inputs=inputs,
            cpu_seconds=cpu_seconds,
            wall_seconds=wall_seconds,
            memory_mb=memory_mb,
        )
    except SandboxViolation as exc:
        return (None, f"SandboxViolation: {exc}")
    if res.error:
        return (None, res.error)
    val = res.return_value
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return (None, f"evaluator must return a finite float, got {val!r}")
    f = float(val)
    if math.isnan(f) or math.isinf(f):
        return (None, f"evaluator returned non-finite score: {val!r}")
    return (f, None)
