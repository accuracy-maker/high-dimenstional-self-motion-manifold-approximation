"""
Evaluate the peformance of learned fourier SMMs approximator

metrics:
fk_err = 0.5(posistion_err / total_length + ori_err / pi)

compared with null-space projection method
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from realtime_smm.helpers.robot import Robot as RefRobot
from realtime_smm.learning import SMMNetworkBundle

from .pipeline import TASKConfig
from .robots.planar3r import planar3r


robot_name = "3R"
task = "planar"

TaskConfig = TASKConfig(
    robot_name=robot_name,
    task = task
)

folder_path = TaskConfig.get_saved_path
print(f"path is: {TaskConfig.get_saved_path}")

BUNDLE_NAME = "bundle"
X_RANGE, Y_RANGE = (-3.0, 3.0), (-3.0, 3.0)
N_TARGETS = 10_000
SAMPLES_PER_BRANCH = 128

# dls ik correction
CORRECTION_ITERS = 3

robot = TaskConfig.get_robot
BASE_DIR = folder_path
bundle = SMMNetworkBundle.load(name="3R", base_dir=BASE_DIR, device="cpu")

rng = np.random.default_rng(0)
xs, ys = [], []
while len(xs) < N_TARGETS:
    n_needed = N_TARGETS - len(xs)
    cx = rng.uniform(*X_RANGE, int(n_needed * 1.4) + 16)
    cy = rng.uniform(*Y_RANGE, int(n_needed * 1.4) + 16)
    keep = cx ** 2 + cy ** 2 <= 3.0 ** 2
    xs.extend(cx[keep][:n_needed])
    ys.extend(cy[keep][:n_needed])
xs, ys = np.array(xs), np.array(ys)

x_rep_chunks, q_raw_chunks = [], []
n_ok = 0
for i, (x, y) in enumerate(zip(xs, ys)):
    T = np.eye(4)
    T[0, 3], T[1, 3] = x, y
    ws = bundle(T, samples=SAMPLES_PER_BRANCH)
    if ws.status.name != "OK":
        continue
    n_ok += 1
    q = np.concatenate([b.data.astype(float) for b in ws.data], axis=0)
    x_rep_chunks.append(np.tile([x, y], (q.shape[0], 1)))
    q_raw_chunks.append(q)
    if (i + 1) % 2000 == 0:
        print(f"  ...{i + 1}/{N_TARGETS} targets")
x_rep = np.concatenate(x_rep_chunks)
q_raw = np.concatenate(q_raw_chunks)

T_target = np.tile(np.eye(4), (q_raw.shape[0], 1, 1))
T_target[:, 0, 3] = x_rep[:, 0]
T_target[:, 1, 3] = x_rep[:, 1]

ep, _ = robot.bk.fk_error_pct(q_raw, T_target)
# print(f"ep mean: {ep.mean()} | ep min: {ep.min()}")
err_pct = np.maximum(ep * 100.0, 1e-16)
print(f"Fourier-series SMM raw output ({n_ok}/{N_TARGETS} reachable targets, "
        f"{q_raw.shape[0]} configs): mean {err_pct.mean():.3g}% | "
        f"median {np.median(err_pct):.3g}% | max {err_pct.max():.3g}%")

q_corrected = robot.bk.ik_correct(q_raw, T_target, iters=CORRECTION_ITERS)
ep_c, _ = robot.bk.fk_error_pct(q_corrected, T_target)
err_corrected_pct = np.maximum(ep_c * 100.0, 1e-16)
print(f"  + {CORRECTION_ITERS} Jacobian IK-correction steps: "
        f"mean {err_corrected_pct.mean():.3g}% | median {np.median(err_corrected_pct):.3g}% | "
        f"max {err_corrected_pct.max():.3g}%")