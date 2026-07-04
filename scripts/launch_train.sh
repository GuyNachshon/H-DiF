#!/usr/bin/env bash
# Launch wrapper for the RunPod training pod: stages data onto local NVMe if present,
# runs training, and auto-stops the pod when the process exits (any exit code) so we
# don't pay for idle GPU time after a run finishes or crashes.
#
# Usage: scripts/launch_train.sh <config.yaml>
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
# set -a so sourced keys are exported to the python child (wandb/HF need them).
# RUNPOD_POD_ID isn't visible in ssh/tmux sessions — put it in .env too for auto-stop.
set -a
source .env
set +a

mkdir -p /workspace/logs

DATA_ROOT_ARGS=()
if [ -d /workspace/h-dif/data ]; then
    # Stage only the train split on local NVMe (read every step; the full dataset
    # doesn't fit the 40GB container disk next to the venv). Val stays on the
    # volume via symlink — it's read once per val_every steps.
    if [ ! -f /root/data/.staged ]; then
        echo "staging train split: /workspace/h-dif/data/train -> /root/data/train (local NVMe)"
        mkdir -p /root/data
        time rsync -a --delete /workspace/h-dif/data/train/ /root/data/train/
        RSYNC_STATUS=$?
        if [ "$RSYNC_STATUS" -ne 0 ] || ! ln -sfn /workspace/h-dif/data/val /root/data/val; then
            rm -f /root/data/.staged
            echo "ERROR: data staging failed (rsync exit $RSYNC_STATUS)"
            exit 1
        fi
        touch /root/data/.staged
    fi
    DATA_ROOT_ARGS=(--data_root /root/data)
fi

export PYTHONUNBUFFERED=1  # never lose a traceback to block buffering through tee
echo "starting training: config=$1"
/root/venv/bin/python src/train.py --config "$1" "${DATA_ROOT_ARGS[@]}" 2>&1 | tee /workspace/logs/train.log
EXIT_CODE=${PIPESTATUS[0]}
TEE_STATUS=${PIPESTATUS[1]}
if [ "$TEE_STATUS" -ne 0 ]; then
    echo "warning: tee exited with code $TEE_STATUS, train.log may be incomplete"
fi

echo "training exited with code $EXIT_CODE"
echo "$EXIT_CODE" > /workspace/logs/TRAIN_EXITED

# RUNPOD_POD_ID is auto-set inside RunPod containers. RUNPOD_API_KEY must be added
# to the pod's .env for auto-stop to work; without it we just leave the pod running.
if [ -n "${RUNPOD_POD_ID:-}" ] && [ ! -f /workspace/KEEP_ALIVE ]; then
    if [ -n "${RUNPOD_API_KEY:-}" ]; then
        echo "auto-stopping pod $RUNPOD_POD_ID"
        if ! curl -sf --max-time 30 -X POST "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop" \
            -H "Authorization: Bearer $RUNPOD_API_KEY"; then
            echo "auto-stop request failed, retrying once in 10s"
            sleep 10
            if ! curl -sf --max-time 30 -X POST "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop" \
                -H "Authorization: Bearer $RUNPOD_API_KEY"; then
                echo "AUTO-STOP FAILED (code $?) — POD STILL RUNNING AND BILLING"
            fi
        fi
    else
        echo "RUNPOD_API_KEY not set, skipping auto-stop"
    fi
else
    echo "auto-stop skipped (KEEP_ALIVE present or not on RunPod)"
fi
