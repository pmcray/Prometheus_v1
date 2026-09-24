"""Command-line entry point for the governor workstation.

    prometheus-governor keygen  --home ~/.prometheus-governor
    prometheus-governor policy  --home ... --run-id run-001 --audit-seeds 1000-1049 --model-pin qwen=<hash>
    prometheus-governor init    --home ... --exchange ~/drive/prometheus/run-001 --baseline <bundle>
    prometheus-governor run     --home ... --exchange ...        (loop: decide proposals, send heartbeats)
    prometheus-governor kill    --home ... --exchange ... --reason "..."
    prometheus-governor status  --home ...
    prometheus-governor verify-log <path.jsonl>

The key passphrase is read from the PROMETHEUS_GOVERNOR_PASSPHRASE environment
variable if set, otherwise prompted for.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import time
from pathlib import Path

from ..common.eventlog import verify_log
from .keys import generate_private_key, load_private_key, public_key_hex, save_private_key
from .policy import Policy
from .service import Governor

KEY_FILE = "governor_key.pem"


def _passphrase(confirm: bool = False) -> bytes | None:
    env = os.environ.get("PROMETHEUS_GOVERNOR_PASSPHRASE")
    if env is not None:
        return env.encode() or None
    pw = getpass.getpass("Governor key passphrase (empty for none): ")
    if confirm and pw != getpass.getpass("Repeat passphrase: "):
        sys.exit("passphrases do not match")
    return pw.encode() or None


def _seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


def _governor(args) -> Governor:
    key = load_private_key(Path(args.home) / KEY_FILE, _passphrase())
    return Governor(Path(args.home), Path(args.exchange), key)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="prometheus-governor")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("keygen"); s.add_argument("--home", required=True)
    s = sub.add_parser("policy")
    s.add_argument("--home", required=True); s.add_argument("--run-id", required=True)
    s.add_argument("--audit-seeds", required=True, help="e.g. 1000-1049")
    s.add_argument("--model-pin", action="append", default=[], help="name=commit_hash")
    s.add_argument("--max-generations", type=int, default=10)
    for name in ("init", "run", "kill"):
        s = sub.add_parser(name)
        s.add_argument("--home", required=True); s.add_argument("--exchange", required=True)
        if name == "init":
            s.add_argument("--baseline", required=True)
        if name == "run":
            s.add_argument("--once", action="store_true")
        if name == "kill":
            s.add_argument("--reason", required=True)
    s = sub.add_parser("status"); s.add_argument("--home", required=True)
    s = sub.add_parser("verify-log"); s.add_argument("path")
    args = p.parse_args(argv)

    if args.cmd == "keygen":
        home = Path(args.home); home.mkdir(parents=True, exist_ok=True, mode=0o700)
        key = generate_private_key()
        save_private_key(key, home / KEY_FILE, _passphrase(confirm=True))
        pub = public_key_hex(key)
        (home / "governor_public_key.txt").write_text(pub + "\n")
        print(f"public key (copy to the Colab runtime config): {pub}")
    elif args.cmd == "policy":
        pins = dict(pin.split("=", 1) for pin in args.model_pin)
        policy = Policy(run_id=args.run_id, audit_seeds=_seeds(args.audit_seeds),
                        model_pins=pins, max_generations=args.max_generations)
        policy.save(Path(args.home) / "policy.json")
        print(f"policy written: {len(policy.audit_seeds)} audit worlds, {len(pins)} model pins")
    elif args.cmd == "init":
        st = _governor(args).init_run(Path(args.baseline))
        print(f"run initialised: baseline audit score {st.audit_score:.3f}")
    elif args.cmd == "run":
        gov = _governor(args)
        while True:
            for d in gov.step():
                b = d["body"]
                print(f"gen {b['generation']}: {b['decision']} - {'; '.join(b['reasons'])}")
            if args.once:
                break
            time.sleep(gov.policy.heartbeat_interval_s)
    elif args.cmd == "kill":
        _governor(args).kill(args.reason)
        print("kill switch pulled; halt heartbeat written")
    elif args.cmd == "status":
        print((Path(args.home) / "state.json").read_text())
    elif args.cmd == "verify-log":
        r = verify_log(args.path)
        print(json.dumps(r.__dict__))
        return 0 if r.ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
