"""Minimal local HTTP server and HTML renderer for the Observatory dashboard."""

from __future__ import annotations

import html
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .collector import collect_snapshot


def _esc(val: Any) -> str:
    return html.escape("" if val is None else str(val), quote=True)


def _fmt_pct(val: float | None) -> str:
    if val is None:
        return "—"
    return f"{val * 100:.1f}%"


def _bar(val: float | None) -> str:
    if val is None:
        return ""
    pct = max(0.0, min(100.0, float(val) * 100.0))
    return f'<div class="bar"><div class="bar-fill" style="width:{pct:.1f}%"></div></div>'


def render_dashboard_html(snap: dict[str, Any]) -> str:
    """Render a self-contained HTML dashboard from a snapshot dict."""
    status = snap.get("overall_status", "UNKNOWN")
    badge_cls = {
        "HEALTHY": "ok",
        "HALTED": "danger",
        "LOG_CORRUPTED": "danger",
        "BAD_SIGNATURE": "danger",
        "STALE_HEARTBEAT": "warn",
        "NO_HEARTBEAT": "warn",
    }.get(status, "warn")

    hb = snap.get("heartbeat") or {}
    st = snap.get("state") or {}
    pol = snap.get("policy") or {}
    cog = snap.get("cognitive") or {}
    mem = snap.get("memory") or {}
    tr = snap.get("transfer") or {}
    ashby = snap.get("ashby") or {}
    law_shift = snap.get("law_shift") or {}
    multigen = snap.get("multigen") or {}
    ashby_status = ashby.get("status", "—")
    ashby_cls = "ok" if ashby_status == "HOMEOSTATIC" else ("warn" if ashby_status == "STRESSED" else "muted")
    ashby_reasons = ", ".join(ashby.get("active_reasons") or []) or "none"

    run_id = hb.get("run_id") or pol.get("run_id") or "—"
    gen_str = (
        f"{st.get('generation', '—')} / {pol.get('max_generations', '—')}"
        if st else "—"
    )

    # Decision rows (prefer governor decisions.jsonl, fall back to exchange/approvals)
    decisions = snap.get("decisions") or snap.get("approvals") or []
    decision_rows = []
    for d in reversed(decisions[-30:]):
        dec = str(d.get("decision", "—"))
        dec_cls = "ok" if dec == "approve" else ("danger" if dec in ("veto", "KILL") else "muted")
        reasons = "; ".join(str(r) for r in (d.get("reasons") or []))
        decision_rows.append(
            "<tr>"
            f"<td>{_esc(d.get('proposal', '—'))}</td>"
            f"<td>{_esc(d.get('generation', '—'))}</td>"
            f'<td><span class="pill {dec_cls}">{_esc(dec)}</span></td>'
            f"<td>{_fmt_pct(d.get('audit_score'))}</td>"
            f"<td>{_fmt_pct(d.get('proxy_score'))}</td>"
            f"<td>{_esc(reasons)}</td>"
            "</tr>"
        )
    if not decision_rows:
        decision_rows.append('<tr><td colspan="6" class="muted">No decisions recorded yet.</td></tr>')

    # Pending proposals
    pending = snap.get("pending_proposals") or []
    pending_rows = [
        f"<tr><td>{_esc(p['name'])}</td><td>{_esc(p.get('generation', '—'))}</td>"
        f"<td>{'yes' if p.get('ready') else 'no'}</td>"
        f"<td>{'yes' if p.get('has_claims') else 'no'}</td></tr>"
        for p in pending
    ] or ['<tr><td colspan="4" class="muted">No pending proposals.</td></tr>']

    # Log integrity
    logs = snap.get("logs") or []
    log_rows = []
    for entry in logs:
        cls = "ok" if entry["ok"] else "danger"
        state_txt = "VALID" if entry["ok"] else f"CORRUPT (seq {entry.get('first_bad_seq')}): {entry.get('reason')}"
        log_rows.append(
            f"<tr><td>{_esc(entry['label'])}</td><td>{_esc(entry['records'])}</td>"
            f'<td><span class="pill {cls}">{_esc(state_txt)}</span></td></tr>'
        )
    if not log_rows:
        log_rows.append('<tr><td colspan="3" class="muted">No event logs found.</td></tr>')

    # Bake-off leaderboard
    bakeoff = snap.get("bakeoff") or []
    bakeoff_rows = []
    for i, b in enumerate(bakeoff):
        lat = f"{b['mean_latency_s']:.2f}s" if b.get("mean_latency_s") is not None else "—"
        tag = " ★" if i == 0 and len(bakeoff) > 1 else ""
        bakeoff_rows.append(
            "<tr>"
            f"<td>{_esc(b['model_name'])}{tag}</td>"
            f"<td>{_esc(b['episodes'])}</td>"
            f"<td>{_fmt_pct(b.get('accuracy'))} {_bar(b.get('accuracy'))}</td>"
            f"<td>{_esc(b.get('n_errors', 0))}</td>"
            f"<td>{_esc(lat)}</td>"
            "</tr>"
        )
    if not bakeoff_rows:
        bakeoff_rows.append('<tr><td colspan="5" class="muted">No bake-off checkpoints found.</td></tr>')

    # Lean 4 prover bake-off leaderboard
    prover_rows = []
    for p_entry in (snap.get("prover_bakeoff") or []):
        plat = f"{p_entry['mean_latency_s']:.2f}s" if p_entry.get("mean_latency_s") is not None else "—"
        prover_rows.append(
            "<tr>"
            f"<td>{_esc(p_entry['model_name'])}</td>"
            f"<td><code>{_esc(p_entry.get('lean_toolchain', '—'))}</code></td>"
            f"<td>{_esc(p_entry.get('n_proved', 0))}/{_esc(p_entry.get('n_worlds', 0))} "
            f"(pass@{_esc(p_entry.get('attempts_per_world', 1))})</td>"
            f"<td>{_fmt_pct(p_entry.get('pass_rate'))} {_bar(p_entry.get('pass_rate'))}</td>"
            f"<td>{_esc(plat)}</td>"
            "</tr>"
        )
    if not prover_rows:
        prover_rows.append('<tr><td colspan="5" class="muted">No Lean 4 prover checkpoints found.</td></tr>')

    ls_gain = law_shift.get("recovery_gain")
    ls_gain_str = f"{ls_gain * 100:+.1f}%" if isinstance(ls_gain, (int, float)) else "—"
    mg_pins_str = ", ".join(f"{k}={v[:18]}…" if len(v) > 20 else f"{k}={v}" for k, v in (multigen.get("model_pins") or {}).items()) or "—"

    # Checkpoint per-law breakdown
    cp = snap.get("checkpoint")
    law_rows = []
    if cp and cp.get("per_law"):
        for law, acc in sorted(cp["per_law"].items()):
            law_rows.append(
                f"<tr><td>{_esc(law)}</td><td>{_fmt_pct(acc)}</td><td>{_bar(acc)}</td></tr>"
            )
    else:
        law_rows.append('<tr><td colspan="3" class="muted">No episode checkpoint yet.</td></tr>')

    hb_sig = (
        "verified" if hb.get("sig_valid") is True
        else ("INVALID" if hb.get("sig_valid") is False else "unchecked (no pubkey)")
    )
    hb_exp = f"{hb['expires_in_s']}s" if hb.get("expires_in_s") is not None else "—"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="30">
<title>Prometheus v1 Observatory — {_esc(run_id)}</title>
<style>
  :root {{ --bg:#0e1117; --card:#161b22; --fg:#e6edf3; --muted:#8b949e; --border:#30363d;
           --ok:#2ea043; --warn:#d29922; --danger:#f85149; --accent:#58a6ff; }}
  body {{ margin:0; padding:24px; background:var(--bg); color:var(--fg);
          font:14px/1.5 -apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif; }}
  h1,h2 {{ margin:0 0 12px; font-weight:600; }}
  h1 {{ font-size:20px; display:flex; align-items:center; gap:12px; }}
  h2 {{ font-size:15px; color:var(--accent); margin-top:4px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:16px; margin-bottom:20px; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:8px; padding:16px; }}
  .kv {{ display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid #21262d; }}
  .kv:last-child {{ border-bottom:none; }}
  .muted {{ color:var(--muted); }}
  .pill {{ display:inline-block; padding:2px 8px; border-radius:12px; font-size:12px; font-weight:600; }}
  .pill.ok {{ background:rgba(46,160,67,.2); color:#3fb950; }}
  .pill.warn {{ background:rgba(210,153,34,.2); color:#e3b341; }}
  .pill.danger {{ background:rgba(248,81,73,.2); color:#ff7b72; }}
  .pill.muted {{ background:#21262d; color:var(--muted); }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th,td {{ text-align:left; padding:7px 8px; border-bottom:1px solid #21262d; vertical-align:middle; }}
  th {{ color:var(--muted); font-weight:600; }}
  .bar {{ display:inline-block; width:90px; height:8px; background:#21262d; border-radius:4px; overflow:hidden; vertical-align:middle; margin-left:6px; }}
  .bar-fill {{ height:100%; background:var(--accent); }}
  .stack {{ display:grid; gap:16px; }}
</style>
</head>
<body>
  <h1>Prometheus v1 Observatory <span class="pill {badge_cls}">{_esc(status)}</span></h1>
  <p class="muted">Run: <strong>{_esc(run_id)}</strong> &middot; Exchange: <code>{_esc(snap.get('exchange'))}</code> &middot; Auto-refresh 30s &middot; <a href="/api/snapshot" style="color:var(--accent)">JSON</a></p>

  <div class="grid">
    <div class="card">
      <h2>Heartbeat &amp; Kill Switch</h2>
      <div class="kv"><span class="muted">Present</span><span>{'yes' if hb.get('present') else 'no'}</span></div>
      <div class="kv"><span class="muted">Counter</span><span>{_esc(hb.get('counter', '—'))}</span></div>
      <div class="kv"><span class="muted">Halt flag</span><span>{'TRUE' if hb.get('halt') else 'false'}</span></div>
      <div class="kv"><span class="muted">Expires in</span><span>{_esc(hb_exp)}</span></div>
      <div class="kv"><span class="muted">Signature</span><span>{_esc(hb_sig)}</span></div>
    </div>

    <div class="card">
      <h2>Governor State</h2>
      <div class="kv"><span class="muted">Generation</span><span>{_esc(gen_str)}</span></div>
      <div class="kv"><span class="muted">Audit score</span><span>{_fmt_pct(st.get('audit_score'))}</span></div>
      <div class="kv"><span class="muted">Proxy score</span><span>{_fmt_pct(st.get('proxy_score'))}</span></div>
      <div class="kv"><span class="muted">Approvals / Vetoes</span><span>{_esc(st.get('approvals', '—'))} / {_esc(st.get('vetoes', '—'))}</span></div>
      <div class="kv"><span class="muted">Audit worlds</span><span>{_esc(pol.get('n_audit_seeds', '—'))}</span></div>
    </div>

    <div class="card">
      <h2>Runtime Checkpoint</h2>
      <div class="kv"><span class="muted">Episodes</span><span>{_esc(cp['episodes'] if cp else '—')}</span></div>
      <div class="kv"><span class="muted">Accuracy</span><span>{_fmt_pct(cp['accuracy'] if cp else None)}</span></div>
      <div class="kv"><span class="muted">Budget spent</span><span>{_esc(cp['total_spent'] if cp else '—')}</span></div>
      <div class="kv"><span class="muted">Pending proposals</span><span>{len(pending)}</span></div>
    </div>

    <div class="card">
      <h2>Cognitive Core &amp; MDL Memory</h2>
      <div class="kv"><span class="muted">Cognitive episodes</span><span>{_esc(cog.get('episodes', '—'))}</span></div>
      <div class="kv"><span class="muted">Mean budget / ep</span><span>{_esc(cog.get('mean_budget_spent', '—'))}</span></div>
      <div class="kv"><span class="muted">Mean entropy (bits)</span><span>{_esc(cog.get('mean_entropy_bits', '—'))}</span></div>
      <div class="kv"><span class="muted">Early-stop rate</span><span>{_fmt_pct(cog.get('early_stop_rate'))}</span></div>
      <div class="kv"><span class="muted">Mean MDL gain (bits)</span><span>{_esc(cog.get('mean_mdl_gain_bits', '—'))}</span></div>
      <div class="kv"><span class="muted">Assemblies / consolidated</span><span>{_esc(mem.get('n_assemblies', '—'))} / {_esc(mem.get('episodes_consolidated', '—'))}</span></div>
    </div>

    <div class="card">
      <h2>Ashby Ultrastability &amp; Analogy Transfer</h2>
      <div class="kv"><span class="muted">Homeostasis</span><span class="pill {ashby_cls}">{_esc(ashby_status)}</span></div>
      <div class="kv"><span class="muted">Stress triggers / proposals</span><span>{_esc(ashby.get('trigger_count', '—'))} / {_esc(ashby.get('proposals_published', '—'))}</span></div>
      <div class="kv"><span class="muted">Active stress channels</span><span>{_esc(ashby_reasons)}</span></div>
      <div class="kv"><span class="muted">Analogy hint rate</span><span>{_fmt_pct(tr.get('hint_rate'))}</span></div>
      <div class="kv"><span class="muted">Total analogy hints</span><span>{_esc(tr.get('total_hints', '—'))}</span></div>
      <div class="kv"><span class="muted">Mean analogy confidence</span><span>{_fmt_pct(tr.get('mean_confidence'))}</span></div>
    </div>

    <div class="card">
      <h2>Phase 3: Law-Shift &amp; Pinned Multi-Gen</h2>
      <div class="kv"><span class="muted">Law-shift static post-acc</span><span>{_fmt_pct(law_shift.get('static_post_accuracy'))}</span></div>
      <div class="kv"><span class="muted">Law-shift adapted post-acc</span><span>{_fmt_pct(law_shift.get('adapted_post_accuracy'))} ({_esc(ls_gain_str)})</span></div>
      <div class="kv"><span class="muted">Surprise detection rate</span><span>{_fmt_pct(law_shift.get('surprise_detection_rate'))}</span></div>
      <div class="kv"><span class="muted">Multi-gen progress</span><span>{_esc(multigen.get('end_generation', '—'))} / {_esc(multigen.get('target_generations', '—'))}</span></div>
      <div class="kv"><span class="muted">Multi-gen approvals / vetoes</span><span>{_esc(multigen.get('approvals', '—'))} / {_esc(multigen.get('vetoes', '—'))}</span></div>
      <div class="kv"><span class="muted">Pinned models</span><span><code>{_esc(mg_pins_str)}</code></span></div>
    </div>
  </div>

  <div class="stack">
    <div class="card">
      <h2>Foundation-Model Bake-off Leaderboard</h2>
      <table>
        <thead><tr><th>Model</th><th>Episodes</th><th>Accuracy</th><th>Errors</th><th>Mean Latency</th></tr></thead>
        <tbody>{''.join(bakeoff_rows)}</tbody>
      </table>
    </div>

    <div class="card">
      <h2>Lean 4 Prover Bake-off Leaderboard</h2>
      <table>
        <thead><tr><th>Prover</th><th>Toolchain</th><th>Proved</th><th>Pass Rate</th><th>Mean Latency</th></tr></thead>
        <tbody>{''.join(prover_rows)}</tbody>
      </table>
    </div>

    <div class="card">
      <h2>Governor Decisions &amp; Lineage</h2>
      <table>
        <thead><tr><th>Proposal</th><th>Gen</th><th>Decision</th><th>Audit</th><th>Proxy</th><th>Reasons</th></tr></thead>
        <tbody>{''.join(decision_rows)}</tbody>
      </table>
    </div>

    <div class="grid">
      <div class="card">
        <h2>Audit Log Integrity</h2>
        <table>
          <thead><tr><th>Log</th><th>Records</th><th>Status</th></tr></thead>
          <tbody>{''.join(log_rows)}</tbody>
        </table>
      </div>

      <div class="card">
        <h2>Pending Proposals</h2>
        <table>
          <thead><tr><th>Name</th><th>Gen</th><th>Ready</th><th>Claims</th></tr></thead>
          <tbody>{''.join(pending_rows)}</tbody>
        </table>
      </div>
    </div>

    <div class="card">
      <h2>Checkpoint Per-Law Accuracy</h2>
      <table>
        <thead><tr><th>Law</th><th>Accuracy</th><th>Bar</th></tr></thead>
        <tbody>{''.join(law_rows)}</tbody>
      </table>
    </div>
  </div>
</body>
</html>
"""


def make_handler(
    exchange: Path,
    *,
    governor_home: Path | None = None,
    prometheus_dir: Path | None = None,
) -> type[BaseHTTPRequestHandler]:
    class ObservatoryHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/healthz":
                self._send(HTTPStatus.OK, b'{"ok":true}\n', "application/json")
                return
            snap = collect_snapshot(
                exchange,
                governor_home=governor_home,
                prometheus_dir=prometheus_dir,
            )
            if path == "/api/snapshot":
                body = json.dumps(snap, indent=2, sort_keys=True).encode("utf-8")
                self._send(HTTPStatus.OK, body, "application/json; charset=utf-8")
                return
            if path in ("/", "/index.html"):
                body = render_dashboard_html(snap).encode("utf-8")
                self._send(HTTPStatus.OK, body, "text/html; charset=utf-8")
                return
            self._send(HTTPStatus.NOT_FOUND, b"not found\n", "text/plain; charset=utf-8")

        def _send(self, code: HTTPStatus, payload: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    return ObservatoryHandler


def serve(
    exchange: Path | str,
    *,
    governor_home: Path | str | None = None,
    prometheus_dir: Path | str | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    """Start the local Observatory HTTP server (blocking until Ctrl-C)."""
    handler = make_handler(
        Path(exchange),
        governor_home=Path(governor_home) if governor_home else None,
        prometheus_dir=Path(prometheus_dir) if prometheus_dir else None,
    )
    httpd = ThreadingHTTPServer((host, port), handler)
    print(f"Prometheus Observatory listening on http://{host}:{port}/ (read-only)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
