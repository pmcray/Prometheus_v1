"""Read-only data collector for the Prometheus v1 Observatory.

Reads:
  - Governor home (optional): policy.json, state.json, decisions.jsonl,
    governor_public_key.txt.
  - Exchange directory: heartbeat.json, baseline_hash.txt, proposals/,
    approvals/, logs/*.jsonl, checkpoint.json, bakeoff_*.json.
  - Prometheus root (optional, parent of exchange): governor_public_key.txt,
    bakeoff_*.json, mlflow.db (SQLite, opened read-only).

Everything is read-only and tolerant of missing or partially populated paths.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from ..common.eventlog import read_log, verify_log
from ..governor.keys import load_public_key_hex
from ..governor.tokens import TokenError, verify
from ..lab.laws import LAW_NAMES


def _safe_read_json(path: Path) -> Any | None:
    try:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _load_public_key(governor_home: Path | None, prometheus_dir: Path | None):
    candidates: list[Path] = []
    if governor_home is not None:
        candidates.append(governor_home / "governor_public_key.txt")
    if prometheus_dir is not None:
        candidates.append(prometheus_dir / "governor_public_key.txt")
    for p in candidates:
        try:
            if p.is_file():
                hex_str = p.read_text(encoding="utf-8").strip()
                if len(hex_str) == 64:
                    return load_public_key_hex(hex_str)
        except (OSError, ValueError):
            continue
    return None


def _inspect_token(path: Path, pub: Any | None) -> dict[str, Any] | None:
    raw = _safe_read_json(path)
    if not isinstance(raw, dict):
        return None
    body = raw.get("body") if isinstance(raw.get("body"), dict) else None
    sig_valid: bool | None = None
    sig_error: str | None = None
    if pub is not None:
        try:
            body = verify(raw, pub)
            sig_valid = True
        except TokenError as exc:
            sig_valid = False
            sig_error = str(exc)
    return {
        "file": path.name,
        "key_id": raw.get("key_id"),
        "sig_valid": sig_valid,
        "sig_error": sig_error,
        "body": body,
    }


def _inspect_log(path: Path, label: str) -> dict[str, Any]:
    res = verify_log(path)
    return {
        "label": label,
        "path": str(path),
        "ok": res.ok,
        "records": res.records,
        "first_bad_seq": res.first_bad_seq,
        "reason": res.reason,
    }


def _summarise_checkpoint(data: dict[str, Any]) -> dict[str, Any]:
    results = data.get("results", {})
    if not isinstance(results, dict) or not results:
        return {
            "next_episode": data.get("next_episode", 0),
            "episodes": 0,
            "accuracy": None,
            "per_law": {},
            "per_family": {},
            "total_spent": 0.0,
        }
    entries = list(results.values())
    n = len(entries)
    per_law_hits: dict[str, list[bool]] = {law: [] for law in LAW_NAMES}
    per_fam: dict[str, list[float]] = {}
    total_correct = 0
    total_spent = 0.0
    n_errors = 0
    for r in entries:
        if not isinstance(r, dict):
            continue
        has_err = bool(r.get("error"))
        if has_err:
            n_errors += 1
        claims = r.get("claims") or {}
        truth = r.get("truth") or {}
        ep_correct = 0
        for law in LAW_NAMES:
            ok = (not has_err) and (claims.get(law) == truth.get(law))
            per_law_hits[law].append(ok)
            if ok:
                ep_correct += 1
        total_correct += ep_correct
        fam = str(r.get("family", "unknown"))
        per_fam.setdefault(fam, []).append(ep_correct / len(LAW_NAMES))
        total_spent += float(r.get("spent", r.get("budget_spent", 0.0)) or 0.0)
    return {
        "next_episode": data.get("next_episode", n),
        "episodes": n,
        "n_errors": n_errors,
        "accuracy": total_correct / (n * len(LAW_NAMES)),
        "per_law": {k: sum(v) / len(v) for k, v in per_law_hits.items() if v},
        "per_family": {k: sum(v) / len(v) for k, v in sorted(per_fam.items()) if v},
        "total_spent": round(total_spent, 2),
    }


def _collect_bakeoffs(dirs: list[Path]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for d in dirs:
        if not d.is_dir():
            continue
        for p in sorted(d.glob("bakeoff_*.json")):
            data = _safe_read_json(p)
            if not isinstance(data, dict) or "results" not in data:
                continue
            model = str(data.get("model_name") or p.stem.removeprefix("bakeoff_"))
            summary = _summarise_checkpoint(data)
            lats = [
                float(r.get("latency_s", 0.0))
                for r in data.get("results", {}).values()
                if isinstance(r, dict) and r.get("latency_s") is not None
            ]
            seen[model] = {
                "model_name": model,
                "file": str(p),
                "episodes": summary["episodes"],
                "n_errors": summary.get("n_errors", 0),
                "accuracy": summary["accuracy"],
                "mean_latency_s": (sum(lats) / len(lats)) if lats else None,
                "per_law": summary["per_law"],
                "per_family": summary["per_family"],
            }
    rows = list(seen.values())
    rows.sort(key=lambda r: -(r["accuracy"] if r["accuracy"] is not None else -1.0))
    return rows


def _collect_mlflow_runs(prometheus_dir: Path | None) -> list[dict[str, Any]]:
    if prometheus_dir is None:
        return []
    db_path = prometheus_dir / "mlflow.db"
    if not db_path.is_file():
        return []
    uri = f"file:{db_path}?mode=ro"
    runs: list[dict[str, Any]] = []
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=2.0)
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT run_uuid, name, status, start_time, end_time "
                "FROM runs WHERE lifecycle_stage = 'active' ORDER BY start_time DESC LIMIT 20"
            )
            for run_uuid, name, status, start_t, end_t in cur.fetchall():
                metrics: dict[str, float] = {}
                try:
                    mcur = conn.execute(
                        "SELECT key, value FROM latest_metrics WHERE run_uuid = ?",
                        (run_uuid,),
                    )
                    metrics = {k: float(v) for k, v in mcur.fetchall()}
                except sqlite3.Error:
                    pass
                runs.append({
                    "run_id": run_uuid,
                    "name": name,
                    "status": status,
                    "start_time": start_t,
                    "end_time": end_t,
                    "metrics": metrics,
                })
        finally:
            conn.close()
    except sqlite3.Error:
        return []
    return runs


def collect_snapshot(
    exchange: Path | str,
    *,
    governor_home: Path | str | None = None,
    prometheus_dir: Path | str | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Assemble a complete read-only observability snapshot."""
    now_ts = time.time() if now is None else float(now)
    ex = Path(exchange)
    gov = Path(governor_home) if governor_home is not None else None
    prom = Path(prometheus_dir) if prometheus_dir is not None else ex.parent

    pub = _load_public_key(gov, prom)

    # Governor policy & state (optional — governor home may be mode 0700)
    policy = _safe_read_json(gov / "policy.json") if gov else None
    state = _safe_read_json(gov / "state.json") if gov else None

    # Heartbeat
    hb_info = _inspect_token(ex / "heartbeat.json", pub)
    heartbeat: dict[str, Any] = {"present": hb_info is not None}
    if hb_info is not None and hb_info["body"] is not None:
        b = hb_info["body"]
        expires_at = b.get("expires_at")
        expires_in = (float(expires_at) - now_ts) if isinstance(expires_at, (int, float)) else None
        heartbeat.update({
            "run_id": b.get("run_id"),
            "counter": b.get("counter"),
            "halt": bool(b.get("halt", False)),
            "issued_at": b.get("issued_at"),
            "expires_at": expires_at,
            "expires_in_s": round(expires_in, 1) if expires_in is not None else None,
            "fresh": (expires_in is not None and expires_in >= -60.0),
            "sig_valid": hb_info["sig_valid"],
            "sig_error": hb_info["sig_error"],
        })
    elif hb_info is not None:
        heartbeat.update({"sig_valid": hb_info["sig_valid"], "sig_error": hb_info["sig_error"]})

    # Approvals / vetoes from exchange
    approvals_dir = ex / "approvals"
    approvals: list[dict[str, Any]] = []
    if approvals_dir.is_dir():
        for p in sorted(approvals_dir.glob("*.json")):
            tok = _inspect_token(p, pub)
            if tok and tok["body"]:
                b = tok["body"]
                approvals.append({
                    "proposal": p.stem,
                    "decision": b.get("decision"),
                    "generation": b.get("generation"),
                    "variant_hash": b.get("variant_hash"),
                    "reasons": b.get("reasons", []),
                    "issued_at": b.get("issued_at"),
                    "expires_at": b.get("expires_at"),
                    "sig_valid": tok["sig_valid"],
                })

    # Pending proposals
    proposals_dir = ex / "proposals"
    pending: list[dict[str, Any]] = []
    decided_names = {a["proposal"] for a in approvals}
    if proposals_dir.is_dir():
        for p in sorted(proposals_dir.iterdir()):
            if not p.is_dir() or p.name.startswith(".") or p.name in decided_names:
                continue
            m = _safe_read_json(p / "manifest.json") or {}
            pending.append({
                "name": p.name,
                "ready": (p / ".ready").is_file(),
                "generation": m.get("generation"),
                "has_claims": (p / "audit_claims.json").is_file(),
            })

    # Hash-chained audit logs
    logs: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    if gov and (gov / "decisions.jsonl").is_file():
        dlog = gov / "decisions.jsonl"
        logs.append(_inspect_log(dlog, "governor/decisions.jsonl"))
        try:
            for rec in read_log(dlog):
                kind = rec.get("kind")
                pl = rec.get("payload") or {}
                if kind == "init_run":
                    trajectory.append({
                        "generation": 0,
                        "decision": "baseline",
                        "audit_score": pl.get("audit_score"),
                        "proxy_score": None,
                        "ts": rec.get("ts"),
                    })
                elif kind == "decision":
                    decisions.append({
                        "seq": rec.get("seq"),
                        "ts": rec.get("ts"),
                        **pl,
                    })
                    trajectory.append({
                        "generation": pl.get("generation"),
                        "proposal": pl.get("proposal"),
                        "decision": pl.get("decision"),
                        "audit_score": pl.get("audit_score"),
                        "proxy_score": pl.get("proxy_score"),
                        "ts": rec.get("ts"),
                    })
                elif kind == "kill":
                    decisions.append({
                        "seq": rec.get("seq"),
                        "ts": rec.get("ts"),
                        "proposal": "—",
                        "decision": "KILL",
                        "generation": None,
                        "audit_score": None,
                        "proxy_score": None,
                        "reasons": [pl.get("reason", "")],
                    })
        except (OSError, json.JSONDecodeError):
            pass

    logs_dir = ex / "logs"
    cognitive_records: list[dict[str, Any]] = []
    ashby_triggers: list[dict[str, Any]] = []
    ashby_proposals: list[dict[str, Any]] = []
    if logs_dir.is_dir():
        for p in sorted(logs_dir.glob("*.jsonl")):
            logs.append(_inspect_log(p, f"exchange/logs/{p.name}"))
            try:
                for rec in read_log(p):
                    kind = rec.get("kind")
                    pl = rec.get("payload")
                    if not isinstance(pl, dict):
                        continue
                    if kind == "cognitive_episode":
                        cognitive_records.append(pl)
                    elif kind == "ashby_trigger":
                        ashby_triggers.append(pl)
                    elif kind == "ashby_proposal":
                        ashby_proposals.append(pl)
            except (OSError, json.JSONDecodeError):
                pass

    # Phase 1/2 optional cognitive telemetry (memory_state.json + cognitive_episode logs)
    mem_raw = _safe_read_json(ex / "memory_state.json")
    memory_summary: dict[str, Any] | None = None
    if isinstance(mem_raw, dict) and isinstance(mem_raw.get("assemblies"), list):
        asms = mem_raw["assemblies"]
        saliences = [float(a.get("salience", 0.0)) for a in asms if isinstance(a, dict)]
        memory_summary = {
            "n_assemblies": len(asms),
            "episodes_consolidated": int(mem_raw.get("episodes_consolidated", 0)),
            "mean_salience": round(sum(saliences) / len(saliences), 3) if saliences else 0.0,
        }

    cognitive_summary: dict[str, Any] | None = None
    transfer_summary: dict[str, Any] | None = None
    if cognitive_records:
        n_cog = len(cognitive_records)
        cognitive_summary = {
            "episodes": n_cog,
            "mean_budget_spent": round(
                sum(float(r.get("budget_spent", 0.0)) for r in cognitive_records) / n_cog, 2
            ),
            "mean_entropy_bits": round(
                sum(float(r.get("final_entropy_bits", 0.0)) for r in cognitive_records) / n_cog, 3
            ),
            "mean_mdl_gain_bits": round(
                sum(float(r.get("mdl_compression_bits", 0.0)) for r in cognitive_records) / n_cog, 3
            ),
            "early_stop_rate": round(
                sum(1 for r in cognitive_records if r.get("stopped_early")) / n_cog, 3
            ),
        }
        hint_eps = [r for r in cognitive_records if int(r.get("transfer_hints", 0)) > 0]
        if any("transfer_hints" in r for r in cognitive_records):
            mean_conf = (
                sum(float(r.get("transfer_confidence", 0.0)) for r in hint_eps) / len(hint_eps)
                if hint_eps else 0.0
            )
            transfer_summary = {
                "episodes_with_hints": len(hint_eps),
                "hint_rate": round(len(hint_eps) / n_cog, 3),
                "total_hints": sum(int(r.get("transfer_hints", 0)) for r in cognitive_records),
                "mean_confidence": round(mean_conf, 3),
            }

    # Phase 2 optional Ashby ultrastability telemetry (ashby_state.json + ashby_* logs)
    ashby_raw = _safe_read_json(ex / "ashby_state.json")
    ashby_summary: dict[str, Any] | None = None
    if isinstance(ashby_raw, dict) or ashby_triggers or ashby_proposals:
        st_sum = ashby_raw.get("summary") if isinstance(ashby_raw, dict) else None
        if not isinstance(st_sum, dict):
            st_sum = {}
        ashby_summary = {
            "status": str(st_sum.get("status") or ("STRESSED" if ashby_triggers else "HOMEOSTATIC")),
            "total_episodes": int(
                st_sum.get("total_episodes", ashby_raw.get("total_episodes", 0) if isinstance(ashby_raw, dict) else 0)
            ),
            "trigger_count": max(
                int(st_sum.get("trigger_count", 0)),
                len(ashby_triggers),
            ),
            "proposals_published": len(ashby_proposals),
            "active_reasons": list(st_sum.get("active_reasons") or []),
            "rolling_metrics": dict(st_sum.get("rolling_metrics") or {}),
            "last_proposal": ashby_proposals[-1] if ashby_proposals else None,
        }

    # Checkpoint progress (prefer Phase 2 / Phase 1 checkpoint if present)
    cp_raw = (
        _safe_read_json(ex / "phase2_checkpoint.json")
        or _safe_read_json(ex / "phase1_checkpoint.json")
        or _safe_read_json(ex / "checkpoint.json")
    )
    checkpoint = _summarise_checkpoint(cp_raw) if isinstance(cp_raw, dict) else None

    # Bake-off leaderboard
    bakeoff_dirs = [d for d in (prom, ex) if d is not None]
    bakeoff = _collect_bakeoffs(bakeoff_dirs)

    # Optional MLflow SQLite runs
    mlflow_runs = _collect_mlflow_runs(prom)

    # Overall status badge
    if heartbeat.get("halt") or (state and state.get("killed")):
        overall = "HALTED"
    elif not heartbeat.get("present"):
        overall = "NO_HEARTBEAT"
    elif heartbeat.get("sig_valid") is False:
        overall = "BAD_SIGNATURE"
    elif not heartbeat.get("fresh"):
        overall = "STALE_HEARTBEAT"
    elif any(not entry["ok"] for entry in logs):
        overall = "LOG_CORRUPTED"
    else:
        overall = "HEALTHY"

    return {
        "generated_at": round(now_ts, 3),
        "overall_status": overall,
        "exchange": str(ex),
        "governor_home": str(gov) if gov else None,
        "public_key_loaded": pub is not None,
        "policy": {
            "run_id": policy.get("run_id"),
            "max_generations": policy.get("max_generations"),
            "n_audit_seeds": len(policy.get("audit_seeds", [])),
            "model_pins": policy.get("model_pins", {}),
            "require_ready_marker": policy.get("require_ready_marker", False),
        } if isinstance(policy, dict) else None,
        "state": state,
        "heartbeat": heartbeat,
        "pending_proposals": pending,
        "approvals": approvals,
        "decisions": decisions,
        "trajectory": trajectory,
        "logs": logs,
        "checkpoint": checkpoint,
        "bakeoff": bakeoff,
        "mlflow_runs": mlflow_runs,
        "memory": memory_summary,
        "cognitive": cognitive_summary,
        "transfer": transfer_summary,
        "ashby": ashby_summary,
    }
