#!/usr/bin/env bash
# Push a Kaggle kernel, poll until it finishes, then download its output.
#
# Usage:
#   kaggle/run.sh m4b_regrade
#   kaggle/run.sh m4c_baseline_7b
#
# Uses `python3 -m kaggle` rather than the bare `kaggle` command, since pip
# installed the CLI's entry point outside PATH on this machine -- avoids
# needing to edit shell rc files just to run this.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

NO_PUSH=0
if [[ "${1:-}" == "--no-push" ]]; then
  NO_PUSH=1
  shift
fi
NAME="${1:?usage: kaggle/run.sh [--no-push] <kernel-folder-name>, e.g. m4b_regrade}"
DIR="kaggle/$NAME"
META="$DIR/kernel-metadata.json"
[ -f "$META" ] || { echo "no $META" >&2; exit 1; }

KAGGLE="python3 -m kaggle"
ID=$(python3 -c "import json; print(json.load(open('$META'))['id'])")

if [[ "$ID" == *"<KAGGLE_USERNAME>"* ]]; then
  echo "error: $META still has the <KAGGLE_USERNAME> placeholder" >&2
  exit 1
fi

if [[ "$NO_PUSH" == "1" ]]; then
  echo "== skipping push, resuming monitoring of $ID =="
else
  echo "== pushing $DIR -> $ID =="
  PUSH_OUT=$($KAGGLE kernels push -p "$DIR" 2>&1)
  echo "$PUSH_OUT"
  if echo "$PUSH_OUT" | grep -qi "does not resolve to the specified id"; then
    echo
    echo "warning: Kaggle derived a DIFFERENT slug from the title than the 'id'" >&2
    echo "field in $META. Check the URL printed above, fix 'id' to match it," >&2
    echo "then re-run with --no-push so this doesn't push a duplicate version." >&2
    exit 1
  fi
fi

echo
echo "== polling status every 30s (Ctrl-C stops watching, not the Kaggle run) =="
FAILS=0
MAX_FAILS=10   # ~5 min of tolerance for local network blips before giving up
while true; do
  if STATUS=$($KAGGLE kernels status "$ID" 2>&1); then
    FAILS=0
    echo "$(date +%H:%M:%S)  $STATUS"
    # only trust complete/error/cancel from a call that actually succeeded --
    # a connection exception's own text can contain the word "error" too.
    if echo "$STATUS" | grep -qiE "complete|error|cancel"; then
      break
    fi
  else
    FAILS=$((FAILS + 1))
    echo "$(date +%H:%M:%S)  status check failed (local/network issue, likely NOT the Kaggle run) [$FAILS/$MAX_FAILS]:" >&2
    echo "$STATUS" >&2
    if [ "$FAILS" -ge "$MAX_FAILS" ]; then
      echo "giving up after $MAX_FAILS consecutive failures -- check your network, then resume with:" >&2
      echo "  kaggle/run.sh --no-push $NAME" >&2
      exit 1
    fi
  fi
  sleep 30
done

OUT="logs/kaggle-pulls/$NAME"
mkdir -p "$OUT"
echo
echo "== downloading output to $OUT =="
$KAGGLE kernels output "$ID" -p "$OUT"
echo "done -> $OUT"
