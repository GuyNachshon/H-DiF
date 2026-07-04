#!/usr/bin/env bash
# Launch wrapper for the RunPod training pod: stages data onto local NVMe if present,
# runs training, and auto-stops the pod when the process exits (any exit code) so we
# don't pay for idle GPU time after a run finishes or crashes.
#
# Usage: scripts/launch_train.sh <config.yaml>
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
source .env

mkdir -p /workspace/logs

DATA_ROOT_ARGS=()
if [ -d /workspace/h-dif/data ] && [ ! -d /root/data ]; then
    echo "staging data: /workspace/h-dif/data -> /root/data (local NVMe)"
    time rsync -a /workspace/h-dif/data/ /root/data/
    DATA_ROOT_ARGS=(--data_root /root/data)
fi

echo "starting training: config=$1"
/root/venv/bin/python src/train.py --config "$1" "${DATA_ROOT_ARGS[@]}" 2>&1 | tee /workspace/logs/train.log
EXIT_CODE=${PIPESTATUS[0]}

echo "training exited with code $EXIT_CODE"
echo "$EXIT_CODE" > /workspace/logs/TRAIN_EXITED

# RUNPOD_POD_ID is auto-set inside RunPod containers. RUNPOD_API_KEY must be added
# to the pod's .env for auto-stop to work; without it we just leave the pod running.
if [ -n "${RUNPOD_POD_ID:-}" ] && [ ! -f /workspace/KEEP_ALIVE ]; then
    if [ -n "${RUNPOD_API_KEY:-}" ]; then
        echo "auto-stopping pod $RUNPOD_POD_ID"
        curl -s -X POST "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop" \
            -H "Authorization: Bearer $RUNPOD_API_KEY"
    else
        echo "RUNPOD_API_KEY not set, skipping auto-stop"
    fi
else
    echo "auto-stop skipped (KEEP_ALIVE present or not on RunPod)"
fi
