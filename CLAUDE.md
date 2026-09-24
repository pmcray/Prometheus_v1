# Prometheus v1: context for Claude Code

## What this is
A small, fully inspectable demonstrator of I.J. Good's ultraintelligence principles
("Speculations Concerning the First Ultraintelligent Machine", 1966), combined with
Ashby's ultrastability (*Design for a Brain*), Hofstadter's strange loops (*GEB*) and
Friston's free-energy principle. The Chief Scientist is Paul (this machine's owner).
Built from scratch: Prometheus v0 code and definitions do not apply.

The full design is in the Claude Docs document "Prometheus v1 Demonstrator: Design and
Roadmap". Key decisions from it are summarised below; treat them as settled unless Paul
says otherwise.

## Settled design decisions
- **One environment:** the Hidden-Law Lab (`src/prometheus/lab/`): procedurally generated
  finite algebraic worlds with 8 hidden laws. The agent discovers laws by noisy, costed
  experiments, then states and proves them in Lean.
- **Foundation model runs locally, no web API, ever.** Open-weight, Apache 2.0, served by
  vLLM inside a Google Colab GPU runtime. Lead candidates: Qwen3.8-27B (cortex) and
  Goedel-Prover-V2-8B (Lean 4.9 proofs); Gemma 4 31B as alternative; gpt-oss-20b as a
  T4 fallback. Final pick comes from a Phase 0 bake-off. Weights are pinned to Hugging
  Face commit hashes, safetensors only, custom remote code disabled.
- **Self-improvement is bounded:** up to 10 gated generations per run. Variants may change
  only `harness/{prompts,mesh,memory_params,tools,evaluators}/`, never model weights,
  the governor, the audit suite, essential-variable bounds or the Markov-blanket schema.
- **Governor runs on this Linux laptop**, as a systemd service, outside Colab. It talks to
  Colab only through a Google Drive folder synced with rclone. It holds the only Ed25519
  signing key; Colab holds the public key. Colab installs nothing without a signed
  approval and halts when heartbeats stop, expire, are replayed or carry halt=true.
- **Audit scores are computed by the governor**, which rebuilds audit worlds from their
  seeds and checks the runtime's claims against ground truth. Proxy-only gains (evaluator
  score up, audit score not) are vetoed.
- **Stack:** Python 3.11+, PyTorch, vLLM, RxInfer.jl via juliacall (pymdp for prototypes),
  Lean 4.9 + matching Mathlib, MLflow file store, hash-chained JSONL logs on Drive.
  Everything on Colab must checkpoint to Drive and resume after a forced restart.

## Rules for working in this repo
- Never weaken a governor rule, the gate's checks or a `test_redteam_*` test to make
  something pass. If a rule blocks legitimate work, stop and ask Paul.
- Every new way of attacking the governor or gate gets a `test_redteam_*` test.
- Agent-generated code must never run on this laptop. The governor process needs no
  network (the systemd unit denies it); only the rclone job touches Drive.
- Never commit keys, passphrases or `state.json` (see `.gitignore`).
- Run `pytest -q` before every commit; all tests must pass.

## Commands
```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"
pytest -q                     # 53 tests at the start of this handover
pytest -q -k redteam          # the governor's red-team set (34 cases)
prometheus-governor --help
```

## Where things stand (Phase 0, which runs to 27 Nov 2026)
Done: audit log, variant bundles, governor (keys, signed approvals/heartbeats, rules,
audit scoring, kill switch, CLI), Colab-side gate, Hidden-Law Lab v0, systemd unit and
rclone script (both written, neither installed yet).

Next, in order:
1. Install and run the governor on this laptop: dedicated `governor` user, venv,
   keygen, systemd unit with an encrypted passphrase credential, rclone remote to Google
   Drive, rclone sync on a systemd timer. Confirm with Paul before any `sudo` step.
2. Lean layer: generate a Lean 4.9 statement for each law in each world (finite types,
   provable with `decide`), check ground truth compiles, and pin the Mathlib version.
   There is an existing `~/.elan` toolchain on this machine.
3. Colab runtime template: notebook that mounts Drive, loads the gate with the public
   key, checkpoints every episode, and survives a forced restart.
4. Sandbox for agent-written code on Colab (unprivileged user, no network, CPU/time
   limits); confirm what Colab actually permits.
5. Foundation-model bake-off on 200 Hidden-Law Lab tasks (Colab GPU).
6. Observatory dashboard reading the Drive logs on this laptop.

Phase 0 exit criteria: governor running here blocks 100% of the starter red-team set;
Drive approval/heartbeat round trip works and stopping the governor halts a live run;
1,000 worlds with verified ground-truth proofs; prover and foundation model chosen;
a run survives a forced runtime restart without losing state.
