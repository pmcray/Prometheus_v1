# Prometheus v1

A small, fully inspectable demonstrator of I.J. Good's ultraintelligence principles:
meaning as compression, recall as probabilistic regeneration, assembly-style memory,
and a self-improvement loop kept under control by a governor the agent cannot modify.

Design document: *Prometheus v1 Demonstrator: Design and Roadmap* (Claude Docs).

## Status: Phase 0, first increment

| Component | State | Where |
| --- | --- | --- |
| Hash-chained audit log | Done, tested | `src/prometheus/common/eventlog.py` |
| Variant bundles (content-hashed harness changes) | Done, tested | `src/prometheus/common/variant.py` |
| Governor: Ed25519 keys, signed approvals and heartbeats | Done, tested | `src/prometheus/governor/` |
| Governor: fixed rules, audit scoring, proxy-gain check, kill switch | Done, tested | `governor/rules.py`, `governor/service.py` |
| Governor CLI and systemd unit | Done; CLI tested, unit not yet installed | `governor/cli.py`, `deploy/` |
| Colab-side gate (heartbeat check, verified install) | Done, tested | `src/prometheus/runtime/gate.py` |
| Hidden-Law Lab world generator, 8 laws, law shift | Done, tested | `src/prometheus/lab/` |
| Sandbox for agent-written code | Not started (needs testing on Colab) | Phase 0 |
| Lean 4.9 statements and proofs for each law | Not started | Phase 0 |
| Foundation-model bake-off | Not started (needs Colab GPU) | Phase 0 |
| Resumable Colab runtime template | Not started | Phase 0 |
| Observatory dashboard | Not started | Phase 0 |

Tests: 53 passing, including 34 red-team cases the governor must block (`pytest -k redteam`).

## Layout

```
src/prometheus/
  common/     canonical JSON + hashing, event log, variant bundles   (shared)
  governor/   runs on the workstation only: keys, tokens, policy, rules, service, CLI
  runtime/    runs on Colab: the gate that enforces the governor's decisions
  lab/        Hidden-Law Lab: planted algebraic worlds, law catalogue, audit scoring
tests/        pytest suite (red-team cases named test_redteam_*)
deploy/       systemd unit and rclone sync script for the governor workstation
```

## How the pieces fit

```
Colab runtime                       Google Drive (rclone)         Linux workstation
-------------                       ---------------------         -----------------
Ashby loop proposes variant  --->   proposals/<name>/      --->   governor: rules,
  (bundle + audit_claims.json)                                    audit re-scored from
                                                                  ground truth, signs
gate.install() verifies      <---   approvals/<name>.json  <---   approve / veto
gate.check_heartbeat()       <---   heartbeat.json         <---   every 5 min (halt flag)
```

- The signing key never leaves the workstation. The runtime holds only the public key.
- An approval binds run, generation, the variant's content hash and an expiry, so it
  cannot be replayed onto different files, a different generation or a later time.
- The runtime cannot report its own audit score: it submits claims about the audit
  worlds and the governor rebuilds those worlds from their seeds and scores them.
- No heartbeat, an expired or replayed heartbeat, or a halt flag stops the run at its
  next step. A stopped workstation therefore stops the run, safely.

## Quick start (development)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## Governor on the workstation

```bash
prometheus-governor keygen --home ~/.prometheus-governor          # prints the public key
prometheus-governor policy --home ~/.prometheus-governor --run-id run-001 \
    --audit-seeds 1000-1049 --model-pin cortex=<hf-commit> --model-pin prover=<hf-commit>
prometheus-governor init   --home ~/.prometheus-governor --exchange <synced-run-folder> \
    --baseline <baseline-bundle>
prometheus-governor run    --home ~/.prometheus-governor --exchange <synced-run-folder>
prometheus-governor kill   --home ~/.prometheus-governor --exchange <synced-run-folder> --reason "..."
```

For unattended operation use `deploy/prometheus-governor.service` (systemd 250+ for
encrypted credentials: `systemd-creds encrypt`) with `deploy/rclone-sync.sh` on a timer.
The governor service itself is denied network access; only the rclone job talks to Drive.

## Known limits of this increment

- The static scan of agent-written Python is defence in depth, not the main barrier.
  The main barrier is the sandbox (unprivileged user, no network), still to be built
  and verified on Colab.
- Rules are strict by design: a false veto costs one generation, a false approval
  could cost the run.
- The Hidden-Law Lab currently covers finite algebraic structures with 8 laws. A
  majority-vote guesser scores about 66% on law claims; that is the floor the agent
  must beat.
