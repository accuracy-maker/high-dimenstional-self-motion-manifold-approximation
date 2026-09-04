#!/bin/bash
set -euo pipefail

STAGE="${1:-all}"
LOG_FILE="log.txt"

case "$STAGE" in
    data|train|eval|all) ;;
    *)
        echo "Usage: $0 [data|train|eval|all]" >&2
        exit 1
        ;;
esac

: > "$LOG_FILE"

# robot:task
ROBOTS=(
    "3R:planar"
    "franka_emika_panda:pose"
    "kuka_iiwa_14:pose"
)

for entry in "${ROBOTS[@]}"; do
    robot="${entry%%:*}"
    task="${entry##*:}"

    if [[ "$STAGE" == "data" || "$STAGE" == "all" ]]; then
        python -m assets.data_generation --robot "$robot" >> "$LOG_FILE" 2>&1
    fi

    if [[ "$STAGE" == "train" || "$STAGE" == "all" ]]; then
        python -m training.train --robot_name "$robot" >> "$LOG_FILE" 2>&1
    fi

    if [[ "$STAGE" == "eval" || "$STAGE" == "all" ]]; then
        python -m evaluation.eval --robot_name "$robot" --task "$task" >> "$LOG_FILE" 2>&1
    fi
done