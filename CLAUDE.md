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
pytest -q                     # 228 tests (including live Lean 4 compilation + Phase 1 core)
pytest -q -k redteam          # governor, gate, sandbox, core & prover red-team set
prometheus-governor --help
prometheus-observatory --help
prometheus-verify-worlds --help
```

## Where things stand (`v0.2.0` — Phase 0 live-verified, Phase 1 Cognitive Core complete)
Done (code, tests, workstation deployment, and live Colab Phase 0 verification):
- Audit log, variant bundles, governor (keys, signed approvals/heartbeats, file locking,
  rules, audit scoring, proxy-gain check, kill switch, CLI), and Colab-side gate with
  persistent Drive state (`INSTALL_ROOT` on local ext4 `/content/installed_harness`).
- Live Colab Phase 0 verification: `g1c` signed approval & install round-trip (`generation=1`),
  forced runtime restart recovery, governor-stop halt check, and 200-episode `qwen3-8b` baseline
  (63.94% raw LLM accuracy; pinned at `b968826d9c46dd6066d109eabc6255188de91218` in `models.py`).
- Workstation deployment: `governor` user, `prometheus-governor.service` (network denied),
  `rclone-sync.timer` / `rclone-sync.service` syncing with Google Drive.
  Confirm with Paul before any `sudo` step when updating deployed units/venv.
- Hidden-Law Lab v0 + Lean 4 ground-truth statement & `decide` proof layer (`v4.34.1`
  pinned in `lean/PrometheusLaws/`, 1,000 worlds verified in `lean/verified_1000_worlds.json`).
- Hardened subprocess sandbox (`python -I`, AST screen, import allowlist, audit hook,
  rlimits, dedicated result pipe).
- **Phase 1 Cognitive Core (`src/prometheus/core/`):** immutable Markov blanket (`RestrictedWorld`,
  `EssentialBounds`), pure-Python `ExactBayesianBackend` (120 consistent law profiles, incremental
  Friston EFE, forgetting factor), Good's MDL `AssemblyMemory` (probabilistic prior regeneration,
  atomic Drive checkpointing), strict immutable→mutable `harness/` loader, and hybrid
  `CognitiveAgent` (99.69% accuracy on seeds 0–199 with 36.59 mean budget; 100.0% on frozen audit
  seeds 1000–1049 in read-only memory mode; 5.130 bits mean MDL gain).
- Resumable Colab runtime notebook (`colab/prometheus_runtime.ipynb`) wired for Phase 1
  `CognitiveAgent`, hybrid Qwen3-8B advisor, Phase 1 audit evaluation, and `g2` proposal.
- Local read-only Observatory dashboard (`src/prometheus/observatory/`) with Phase 1 memory &
  cognitive telemetry cards and JSON API.

Optional follow-ups / Next (Phase 2+):
1. Run Phase 1 `CognitiveAgent` (Cell 4 → Cell 4c → Cell 5 → Cell 6 `g2`) on Colab and verify
   `g2` approval (`~1.000` audit score) + Observatory cognitive telemetry.
2. Optional deferred prover bake-off: run `goedel-prover-v2-8b` in Cell 4d (and align Lean 4.9 if
   selected), plus pin `model_pins` in workstation `policy.json` when starting `run-002`.
3. Phase 2+: Hofstadter structural analogy / cross-world transfer (`AnalogyTransferAdapter`),
   Ashby ultrastability monitor (`UltrastabilityMonitor`) driving bounded harness-mutation
   proposals, and law-shift adaptation benchmarks.
