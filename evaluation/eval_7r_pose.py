import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

import torch
from model.flow_matching import FMConfig, FlowMatching

import roboticstoolbox as rtb

def evaluate(cfg: FMConfig):
    # load robot
    robot = rtb.models.DH.Panda()
    print("robot loaded")

    # load model
    fm = FlowMatching(cfg)
    fm.load()
    print(f'model loaded')

    # sampling
    robot_cfg = cfg.load_robot
    data = np.load(robot_cfg.save_path)
    # x = np.array([0.0, 1.5])
    x = data['xs'][0]
    print(f"target pose: {x}")
    qs = fm.sample(x, n_samples=100, n_steps=100)
    # print("sampling qs:\n")
    # print(qs)


    # eval
    T = robot.fkine(qs)
    # position
    position_errors = np.linalg.norm(T.t - x[None, :3], axis=1)
    # print(f"position error:\n {position_errors}")
    total_link_length = 1.3
   
    # orientation
    R_true = Rotation.from_quat(x[3:7]).as_matrix()
    R_pred = T.R

    R_error = R_true.T @ R_pred

    cos_angle = (np.trace(R_error, axis1=1, axis2=2) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    ori_errors = np.arccos(cos_angle)

    error = 0.5 * (position_errors / total_link_length + ori_errors /  np.pi)

    print(f" mean error: {error.mean()}")
    print(f" max error: {error.max()}")



    # plot
    # plot_eval(qs, x, errors)
if __name__ == "__main__":
    cfg = FMConfig()
    evaluate(cfg)
