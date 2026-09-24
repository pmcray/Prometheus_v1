#!/usr/bin/env bash
# Two-way sync between Google Drive and the governor's local exchange folder.
# Run from a systemd timer (every 30-60 s) as the `governor` user.
#
# One-time setup:  rclone config   -> create a remote named "gdrive" (Google Drive)
#
# Direction matters:
#   proposals/ and logs/   Drive -> workstation   (written by the Colab runtime)
#   approvals/, heartbeat  workstation -> Drive   (written by the governor)
set -euo pipefail

RUN_ID="${RUN_ID:-run-001}"
REMOTE="${REMOTE:-gdrive:prometheus/${RUN_ID}}"
LOCAL="${LOCAL:-/var/lib/prometheus-governor/drive/${RUN_ID}}"

mkdir -p "${LOCAL}/proposals" "${LOCAL}/approvals" "${LOCAL}/logs"

rclone copy "${REMOTE}/proposals" "${LOCAL}/proposals" --checksum
rclone copy "${REMOTE}/logs"      "${LOCAL}/logs"      --checksum
rclone copy "${LOCAL}/approvals"  "${REMOTE}/approvals" --checksum
rclone copyto "${LOCAL}/heartbeat.json" "${REMOTE}/heartbeat.json" --checksum 2>/dev/null || true
