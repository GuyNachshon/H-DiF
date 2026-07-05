#!/usr/bin/env bash
# Runs the quality-recipe probes (probe_obj30, probe_obj110) sequentially via launch_train.sh.
# launch_train.sh auto-stops the pod on exit, which would kill the second run --
# KEEP_ALIVE is held until the last config so only the final exit stops the pod.
#
# Usage: scripts/run_probes.sh
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

CONFIGS=(config/probe_obj30.yaml config/probe_obj110.yaml)

touch /workspace/KEEP_ALIVE
for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    if [ "$i" -eq $(( ${#CONFIGS[@]} - 1 )) ]; then
        rm -f /workspace/KEEP_ALIVE
    fi
    echo "=== probe run: $cfg ==="
    scripts/launch_train.sh "$cfg"
    mv checkpoints "checkpoints-$(basename "$cfg" .yaml)"
done
