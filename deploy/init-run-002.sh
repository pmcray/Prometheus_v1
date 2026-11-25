#!/usr/bin/env bash
# Workstation bootstrap helper for Phase 3 pinned multi-generation run (`run-002`).
#
# Creates a separate governor home & exchange directory for `run-002` with
# pinned `model_pins` (`qwen3-8b@b968826d9c46dd6066d109eabc6255188de91218`)
# without touching the existing `run-001` archive.
#
# Review this script carefully before running with `sudo` on the workstation.
set -euo pipefail

RUN_ID="${RUN_ID:-run-002}"
MODEL_PIN="${MODEL_PIN:-cortex=qwen3-8b@b968826d9c46dd6066d109eabc6255188de91218}"
AUDIT_SEEDS="${AUDIT_SEEDS:-1000-1049}"
MAX_GENERATIONS="${MAX_GENERATIONS:-10}"
GOV_HOME="${GOV_HOME:-/var/lib/prometheus-governor/home-${RUN_ID}}"
EXCHANGE="${EXCHANGE:-/var/lib/prometheus-governor/drive/${RUN_ID}}"
VENV="${VENV:-/var/lib/prometheus-governor/venv}"
KEY_SRC="${KEY_SRC:-/var/lib/prometheus-governor/home/governor_key.pem}"
PUB_SRC="${PUB_SRC:-/var/lib/prometheus-governor/home/governor_public_key.txt}"

echo "=== Bootstrapping ${RUN_ID} ==="
echo "  GOV_HOME   : ${GOV_HOME}"
echo "  EXCHANGE   : ${EXCHANGE}"
echo "  MODEL_PIN  : ${MODEL_PIN}"
echo "  AUDIT_SEEDS: ${AUDIT_SEEDS}"

mkdir -p "${GOV_HOME}" "${EXCHANGE}/proposals" "${EXCHANGE}/approvals" "${EXCHANGE}/logs"
chmod 700 "${GOV_HOME}"

if [[ ! -f "${GOV_HOME}/governor_key.pem" && -f "${KEY_SRC}" ]]; then
    cp -p "${KEY_SRC}" "${GOV_HOME}/governor_key.pem"
    cp -p "${PUB_SRC}" "${GOV_HOME}/governor_public_key.txt"
    echo "Copied existing Ed25519 signing key from ${KEY_SRC}"
fi

"${VENV}/bin/prometheus-governor" policy \
    --home "${GOV_HOME}" \
    --run-id "${RUN_ID}" \
    --audit-seeds "${AUDIT_SEEDS}" \
    --model-pin "${MODEL_PIN}" \
    --max-generations "${MAX_GENERATIONS}" \
    --require-ready

BASELINE_DIR="${GOV_HOME}/baseline"
"${VENV}/bin/python" - <<PY
from prometheus.core.multigen import build_pinned_baseline_bundle
pin_role, pin_val = "${MODEL_PIN}".split("=", 1)
lo, hi = (int(x) for x in "${AUDIT_SEEDS}".split("-", 1))
bundle = build_pinned_baseline_bundle(
    "${BASELINE_DIR}",
    run_id="${RUN_ID}",
    model_pins={pin_role: pin_val},
    audit_seeds=list(range(lo, hi + 1)),
    evaluate_audit=True,
)
print(f"Baseline bundle created: hash={bundle.variant_hash}")
PY

"${VENV}/bin/prometheus-governor" init \
    --home "${GOV_HOME}" \
    --exchange "${EXCHANGE}" \
    --baseline "${BASELINE_DIR}"

echo "Done! Next:"
echo "  1. Sync once: RUN_ID=${RUN_ID} /var/lib/prometheus-governor/rclone-sync.sh"
echo "  2. Point prometheus-governor.service / rclone-sync.service at ${RUN_ID} (or run alongside run-001)."
