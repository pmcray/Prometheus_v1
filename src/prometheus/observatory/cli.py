"""Command-line entry point for the Prometheus v1 Observatory.

    prometheus-observatory serve    --exchange ~/drive/prometheus/run-001 [--governor-home ...] [--port 8765]
    prometheus-observatory snapshot --exchange ~/drive/prometheus/run-001 [--governor-home ...] [--json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .collector import collect_snapshot
from .server import serve


def _print_summary(snap: dict) -> None:
    hb = snap.get("heartbeat") or {}
    st = snap.get("state") or {}
    print(f"Status     : {snap['overall_status']}")
    print(f"Exchange   : {snap['exchange']}")
    if hb.get("present"):
        print(
            f"Heartbeat  : counter={hb.get('counter')} halt={hb.get('halt')} "
            f"expires_in={hb.get('expires_in_s')}s sig_valid={hb.get('sig_valid')}"
        )
    else:
        print("Heartbeat  : missing")
    if st:
        print(
            f"Governor   : gen={st.get('generation')} audit={st.get('audit_score')} "
            f"proxy={st.get('proxy_score')} approvals={st.get('approvals')} vetoes={st.get('vetoes')}"
        )
    logs = snap.get("logs") or []
    for entry in logs:
        tag = "OK" if entry["ok"] else f"CORRUPT ({entry['reason']})"
        print(f"Log        : {entry['label']} ({entry['records']} records) — {tag}")
    bakeoff = snap.get("bakeoff") or []
    if bakeoff:
        print("Bake-off   :")
        for b in bakeoff:
            acc = f"{b['accuracy']:.3f}" if b["accuracy"] is not None else "n/a"
            print(f"  {b['model_name']:20s} n={b['episodes']:4d} acc={acc} errors={b['n_errors']}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="prometheus-observatory")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name in ("serve", "snapshot"):
        s = sub.add_parser(name)
        s.add_argument("--exchange", required=True, help="Synced run exchange directory")
        s.add_argument("--governor-home", default=None, help="Optional local governor home directory")
        s.add_argument("--prometheus-dir", default=None, help="Optional parent prometheus Drive dir")
        if name == "serve":
            s.add_argument("--host", default="127.0.0.1")
            s.add_argument("--port", type=int, default=8765)
        else:
            s.add_argument("--json", action="store_true", dest="as_json")

    args = p.parse_args(argv)
    if args.cmd == "serve":
        serve(
            Path(args.exchange),
            governor_home=Path(args.governor_home) if args.governor_home else None,
            prometheus_dir=Path(args.prometheus_dir) if args.prometheus_dir else None,
            host=args.host,
            port=args.port,
        )
        return 0

    snap = collect_snapshot(
        Path(args.exchange),
        governor_home=Path(args.governor_home) if args.governor_home else None,
        prometheus_dir=Path(args.prometheus_dir) if args.prometheus_dir else None,
    )
    if args.as_json:
        print(json.dumps(snap, indent=2, sort_keys=True))
    else:
        _print_summary(snap)
    return 0 if snap["overall_status"] != "LOG_CORRUPTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
