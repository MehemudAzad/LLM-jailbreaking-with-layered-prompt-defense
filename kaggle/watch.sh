#!/usr/bin/env bash
# Live-watch a Kaggle kernel's status in your own terminal, independent of any
# background job Claude may have started -- Ctrl-C stops watching only, the
# Kaggle run itself keeps going.
#
# Usage: kaggle/watch.sh m4c_baseline_7b
set -euo pipefail
cd "$(dirname "$0")/.."

NAME="${1:?usage: kaggle/watch.sh <kernel-folder-name>, e.g. m4c_baseline_7b}"
META="kaggle/$NAME/kernel-metadata.json"
[ -f "$META" ] || { echo "no $META" >&2; exit 1; }

ID=$(python3 -c "import json; print(json.load(open('$META'))['id'])")
URL="https://www.kaggle.com/code/${ID/\//\/}"

echo "watching $ID"
echo "richer live view (actual cell output, not just status): $URL"
echo "Ctrl-C stops watching here; the Kaggle run is unaffected."
echo

FAILS=0
MAX_FAILS=10   # ~3-4 min of tolerance for local network blips before giving up
while true; do
  if STATUS=$(python3 -m kaggle kernels status "$ID" 2>&1); then
    FAILS=0
    echo "$(date +%H:%M:%S)  $STATUS"
    # only trust complete/error/cancel from a call that actually succeeded --
    # a connection exception's own text can contain the word "error" too.
    echo "$STATUS" | grep -qiE "complete|error|cancel" && { echo; echo "finished."; break; }
  else
    FAILS=$((FAILS + 1))
    echo "$(date +%H:%M:%S)  status check failed (local/network issue, likely NOT the Kaggle run) [$FAILS/$MAX_FAILS]" >&2
    if [ "$FAILS" -ge "$MAX_FAILS" ]; then
      echo "giving up watching after $MAX_FAILS consecutive failures -- check your network and re-run this script; it does not affect the Kaggle run itself." >&2
      exit 1
    fi
  fi
  sleep 20
done
