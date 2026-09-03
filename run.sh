#!/bin/bash

LOG_FILE="log.txt"

: > "$LOG_FILE"

# data generation
python -m assets.data_generation --robot 3R >> "$LOG_FILE"

python -m assets.data_generation --robot franka_emika_panda >> "$LOG_FILE"

python -m assets.data_generation --robot kuka_iiwa_14 >> "$LOG_FILE"

# model training
python -m training.train --robot_name 3R >> "$LOG_FILE"

python -m training.train --robot_name franka_emika_panda >> "$LOG_FILE"

python -m training.train --robot_name kuka_iiwa_14 >> "$LOG_FILE"

# model evaluation
python -m evaluation.eval_3r_fm >> "$LOG_FILE"

python -m evaluation.eval_7r_pose --robot_name franka_emika_panda >> "$LOG_FILE"

python -m evaluation.eval_7r_pose --robot_name kuka_iiwa_14 >> "$LOG_FILE"
