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
pytest -q                     # 196 tests (including live Lean 4 compilation)
pytest -q -k redteam          # governor, gate, sandbox & prover red-team set (88 cases)
prometheus-governor --help
prometheus-observatory --help
prometheus-verify-worlds --help
```

## Where things stand (Phase 0, which runs to 27 Nov 2026)
Done (code, tests, and local workstation deployment):
- Audit log, variant bundles, governor (keys, signed approvals/heartbeats, file locking,
  rules, audit scoring, proxy-gain check, kill switch, CLI), and Colab-side gate with
  persistent Drive state.
- Workstation deployment: `governor` user, `prometheus-governor.service` (active, network
  denied), `rclone-sync.timer` / `rclone-sync.service` syncing with Google Drive.
  Confirm with Paul before any `sudo` step when updating deployed units/venv.
- Hidden-Law Lab v0 + Lean 4 ground-truth statement & `decide` proof layer (`v4.34.1`
  pinned in `lean/PrometheusLaws/`, 1,000 worlds verified in `lean/verified_1000_worlds.json`).
- Hardened subprocess sandbox (`python -I`, AST screen, import allowlist, audit hook,
  rlimits, dedicated result pipe).
- Resumable Colab runtime + bake-off notebook (`colab/prometheus_runtime.ipynb`),
  cortex & Lean 4 prover harnesses (`src/prometheus/harness/`), and MLflow bake-off runner.
- Local read-only Observatory dashboard (`src/prometheus/observatory/`).
- Local Phase 0 end-to-end test (`tests/test_phase0_e2e.py`).

Remaining for Phase 0 exit:
1. Run `colab/prometheus_runtime.ipynb` on a Colab GPU: live Drive approval/heartbeat
   round trip, forced runtime restart confirmation, and 200-episode foundation-model
   cortex bake-off (`qwen3-8b`, `gpt-oss-20b`, `gemma4-4b`).
2. Prover bake-off (step 5b) and Lean toolchain alignment (pin Lean 4.9 if selecting
   `Goedel-Prover-V2-8B`), then pin winning Hugging Face commit hashes in
   `src/prometheus/bakeoff/models.py` and the workstation `policy.json`.

Next after Phase 0: Phase 1+ cognitive core (Friston EFE, Good MDL/memory, Hofstadter transfer).

Phase 0 exit criteria: governor running here blocks 100% of the starter red-team set;
Drive approval/heartbeat round trip works and stopping the governor halts a live run;
1,000 worlds with verified ground-truth proofs; prover and foundation model chosen;
a run survives a forced runtime restart without losing state.
