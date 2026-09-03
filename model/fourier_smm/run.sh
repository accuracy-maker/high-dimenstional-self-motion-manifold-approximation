#!/bin/bash

LOG_FILE="log.txt"

: > "$LOG_FILE"

python -m main -r 3R -t planar -x -3.0 3.0 \
    --pos_res 0.005 \
    --stage all \
    --chunk 5000 \
    >> "$LOG_FILE"

python -m main -r panda -t pose -x -1.0 1.0 \
    -z -0.5 1.3 \
    --pos_res 0.1 \
    --stage all \
    --dirs 96 \
    --fft_cutoff 128 \
    --chunk 5000 \
    >> "$LOG_FILE"

python -m main -r iiwa -t pose -x -1.0 1.0 \
    -z -0.6 1.31 \
    --pos_res 0.1 \
    --stage all \
    --dirs 96 \
    --fft_cutoff 128 \
    --chunk 5000 \
    >> "$LOG_FILE"