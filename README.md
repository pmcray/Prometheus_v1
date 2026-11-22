# Prometheus v1

A small, fully inspectable demonstrator of I.J. Good's ultraintelligence principles
combined with Ashby's ultrastability, Hofstadter's strange loops / analogy, and
Friston's free-energy principle: meaning as compression, recall as probabilistic
regeneration, assembly-style memory, active inference over sparse data, and a
self-improvement loop kept under control by a governor the agent cannot modify.

Design document: *Prometheus v1 Demonstrator: Design and Roadmap* (Claude Docs).

## Status: Phase 0 complete (code & local verification; Colab GPU bake-off ready)

| Component | State | Where |
| --- | --- | --- |
| Hash-chained audit log | Done, tested | `src/prometheus/common/eventlog.py` |
| Variant bundles (content-hashed harness changes) | Done, tested | `src/prometheus/common/variant.py` |
| Governor: Ed25519 keys, signed approvals & heartbeats, file locking | Done, tested | `src/prometheus/governor/` |
| Governor: fixed rules, audit scoring, proxy-gain check, kill switch | Done, tested | `governor/rules.py`, `governor/service.py` |
| Governor CLI, systemd units, rclone two-way copy script | Done; active on workstation | `governor/cli.py`, `deploy/` |
| Colab-side gate (heartbeat check, verified atomic install, persistent state) | Done, tested | `src/prometheus/runtime/gate.py` |
| Sandbox for agent-written code (AST + audit hook + rlimits + pipe IPC) | Done, tested | `src/prometheus/runtime/sandbox.py` |
| Hidden-Law Lab world generator, 8 laws, law shift | Done, tested | `src/prometheus/lab/` |
| Lean 4 ground-truth statement & `decide` proof layer | Done, **1,000 worlds verified** (`v4.34.1`) | `src/prometheus/lab/lean_layer.py`, `lean/verified_1000_worlds.json` |
| Resumable Colab runtime template | Done (ready for Colab GPU run) | `colab/prometheus_runtime.ipynb` |
| Foundation-model & prover bake-off harness + runner | Done, unit-tested (ready for Colab GPU) | `src/prometheus/harness/`, `src/prometheus/bakeoff/` |
| Observatory dashboard (read-only web UI + JSON API + CLI) | Done, tested | `src/prometheus/observatory/` |
| Cognitive core (Friston EFE, Good MDL/memory, Hofstadter transfer) | Next (Phase 1+) | — |

Run `pytest -q` for the full suite, or `pytest -q -k redteam` for the governor/gate/sandbox/prover red-team checks.

## Layout

```
src/prometheus/
  common/       canonical JSON + hashing, event log, variant bundles    (shared)
  governor/     runs on the workstation only: keys, tokens, policy, rules, service, CLI
  runtime/      runs on Colab: cryptographic gate + hardened subprocess sandbox
  lab/          Hidden-Law Lab: planted algebraic worlds, law catalogue, audit, Lean 4 layer
  harness/      model client (vLLM OpenAI API), prompts, episode & prover runners
  bakeoff/      candidate model registry + resumable MLflow bake-off runner
  observatory/  local read-only workstation dashboard (HTML + /api/snapshot + CLI)
lean/           pinned Lean 4 toolchain, Mathlib manifest, and 1,000-world proof certificate
colab/          resumable Colab runtime + bake-off notebook template
tests/          pytest suite (red-team cases named test_redteam_*)
deploy/         systemd units and rclone two-way copy script for the governor workstation
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
                                    logs/, checkpoints     --->   observatory (127.0.0.1)
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

## CLI Entry Points

```bash
# Governor (workstation only)
prometheus-governor keygen --home ~/.prometheus-governor
prometheus-governor policy --home ~/.prometheus-governor --run-id run-001 \
    --audit-seeds 1000-1049 --model-pin cortex=<hf-commit> --model-pin prover=<hf-commit> \
    --require-ready
prometheus-governor init   --home ~/.prometheus-governor --exchange <synced-run-folder> \
    --baseline <baseline-bundle>
prometheus-governor run    --home ~/.prometheus-governor --exchange <synced-run-folder>
prometheus-governor kill   --home ~/.prometheus-governor --exchange <synced-run-folder> --reason "..."

# Observatory dashboard (workstation only, read-only, binds 127.0.0.1)
prometheus-observatory serve    --exchange <synced-run-folder> --governor-home ~/.prometheus-governor
prometheus-observatory snapshot --exchange <synced-run-folder> --governor-home ~/.prometheus-governor

# Batch Lean 4 verification of Hidden-Law Lab ground-truth proofs
prometheus-verify-worlds --seeds 0-999 --workers 8 --out lean/verified_1000_worlds.json
```

For unattended operation use `deploy/prometheus-governor.service` (systemd 250+ for
encrypted credentials: `systemd-creds encrypt`) with `deploy/rclone-sync.sh` on
`deploy/rclone-sync.timer`. The governor service itself is denied network access; only
the rclone job talks to Drive.

## Known limits of Phase 0

- On Colab (no user namespaces), the sandbox relies on isolated subprocess execution
  (`python -I`, stripped env), static AST screening, a strict allowlist import hook,
  an irreversible CPython `sys.addaudithook`, and kernel rlimits (`RLIMIT_CPU`,
  `RLIMIT_AS`, `RLIMIT_NPROC`).
- Rules are strict by design: a false veto costs one generation, a false approval
  could cost the run.
- `lean/PrometheusLaws/lean-toolchain` currently pins `v4.34.1` for Lean core `decide`
  ground-truth proofs; if `Goedel-Prover-V2-8B` is selected for the prover role, align
  the prover's Lean/Mathlib toolchain (Lean 4.9) before running Phase 0 step 5b.
- The Hidden-Law Lab currently covers finite algebraic structures with 8 laws. A
  majority-vote guesser scores about 66% on law claims; that is the floor the agent
  must beat.
