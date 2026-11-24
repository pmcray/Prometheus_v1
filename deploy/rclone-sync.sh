#!/usr/bin/env bash
# Two-way copy between Google Drive and the governor's local exchange folder.
# Run from a systemd timer (every 30-60 s) as the `governor` user.
#
# One-time setup:  rclone config   -> create a remote named "gdrive" (Google Drive)
#
# Direction matters (NEVER use destructive `rclone sync` across a shared folder):
#   proposals/ and logs/                    Drive -> workstation  (written by Colab)
#   approvals/, heartbeat, baseline_hash    workstation -> Drive  (written by governor)
set -euo pipefail

RUN_ID="${RUN_ID:-run-001}"
REMOTE="${REMOTE:-gdrive:prometheus/${RUN_ID}}"
LOCAL="${LOCAL:-/var/lib/prometheus-governor/drive/${RUN_ID}}"

mkdir -p "${LOCAL}/proposals" "${LOCAL}/approvals" "${LOCAL}/logs"

# Ensure remote subdirectories exist so initial copy never aborts under set -e.
for sub in proposals approvals logs; do
    rclone mkdir "${REMOTE}/${sub}"
done

# Pull Colab outputs -> workstation
rclone copy "${REMOTE}/proposals" "${LOCAL}/proposals" --checksum
rclone copy "${REMOTE}/logs"      "${LOCAL}/logs"      --checksum
# Pull optional Phase 1 telemetry & checkpoints for local Observatory (never touches approvals/heartbeat)
rclone copy "${REMOTE}" "${LOCAL}" --max-depth 1 \
    --include "checkpoint.json" \
    --include "audit_checkpoint.json" \
    --include "phase1_checkpoint.json" \
    --include "memory_state.json" \
    --checksum

# Push governor outputs -> Drive
rclone copy "${LOCAL}/approvals" "${REMOTE}/approvals" --checksum
if [[ -f "${LOCAL}/baseline_hash.txt" ]]; then
    rclone copyto "${LOCAL}/baseline_hash.txt" "${REMOTE}/baseline_hash.txt" --checksum
fi
if [[ -f "${LOCAL}/heartbeat.json" ]]; then
    rclone copyto "${LOCAL}/heartbeat.json" "${REMOTE}/heartbeat.json" --checksum
fi
