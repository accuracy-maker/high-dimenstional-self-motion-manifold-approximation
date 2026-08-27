import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

import torch
from model.flow_matching import FMConfig, FlowMatching, load_data

import roboticstoolbox as rtb

def evaluate(cfg: FMConfig):
    # load robot
    robot = rtb.models.DH.Panda()
    print(robot)
    print("robot loaded")

    # load model
    train_set, test_set, norm = load_data(cfg)
    fm = FlowMatching(cfg, norm)
    fm.load()
    print(f'model loaded')

    # sampling
    robot_cfg = cfg.load_robot
    data = np.load(robot_cfg.save_path)

    # x = np.array([0.0, 1.5])
    x = data['xs'][1000]
    print(f"target pose: {x}")
    qs = fm.sample(x, n_samples=100, n_steps=100)
    # print("sampling qs:\n")
    # print(qs)


    # eval
    T = np.array(robot.fkine(qs).A)
    print(f"T shape: {T.shape}")

    # position
    position_errors = np.linalg.norm(T[:, :3, 3] - x[:3], axis=-1)
    print(f"position error mean: {position_errors.mean()} m")
    print(f"minimum position error: {position_errors.min()} m")
    total_length = 1.2

    # orientation
    R_true_6d = np.stack([x[3:6], x[6:9]], axis=1)  # (3, 2)
    R_pred_6d = T[:, :3, :2]                        # (N, 3, 2)

    R_true = np.column_stack([
        R_true_6d[:, 0],
        R_true_6d[:, 1],
        np.cross(R_true_6d[:, 0], R_true_6d[:, 1]),
    ])                                               # (3, 3)

    R_pred = np.concatenate([
        R_pred_6d,
        np.cross(
            R_pred_6d[:, :, 0],
            R_pred_6d[:, :, 1],
        )[:, :, None],
    ], axis=2)                                       # (N, 3, 3)

    R_error = R_true.T @ R_pred

    cos_angle = (
        np.trace(R_error, axis1=1, axis2=2) - 1.0
    ) / 2.0

    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    ori_errors = np.arccos(cos_angle)                # radians

    print(f"orientation error mean: {ori_errors.mean()} rad")
    print(f"minimum orientation error: {ori_errors.min()} rad")

    error = 0.5 * (position_errors / total_length + ori_errors / np.pi)
    print(f" mean error: {error.mean()}")
    print(f" min error: {error.min()}")



    # plot
    # plot_eval(qs, x, errors)
if __name__ == "__main__":
    cfg = FMConfig()
    evaluate(cfg)
