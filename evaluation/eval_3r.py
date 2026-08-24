import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt

import torch
from model.flow_matching import FMConfig, FlowMatching

import roboticstoolbox as rtb

def evaluate(cfg: FMConfig):
    # load robot
    robot = rtb.models.DH.Planar3()
    print("robot loaded")

    # load model
    fm = FlowMatching(cfg)
    fm.load()
    print(f'model loaded')

    # sampling
    data = np.load(cfg.dataset_path)
    x = np.array([2, 2])
    print(f"target postion: {x}")
    qs = fm.sample(x, n_samples=100)
    print("sampling qs:\n")
    print(qs)

    # plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ax1 configuration space
    ax1.scatter(qs[:, 1], qs[:, 2], color = 'blue')
    ax1.set_xlabel(r'$\theta_2$ (rad)')
    ax1.set_ylabel(r'$\theta_3$ (rad)')
    ax1.set_xlim(cfg.Q_MIN, cfg.Q_MAX)
    ax1.set_ylim(cfg.Q_MIN, cfg.Q_MAX)

    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(True, alpha=0.3)

    # ax2 workspace
    theta = np.linspace(0, 2 * np.pi, 150)
    r = 3.0

    x_boundary = r * np.cos(theta)
    y_boundary = r * np.sin(theta)

    ax2.plot(
        x_boundary,
        y_boundary,
        color='red',
        linewidth=2,
        label='Workspace boundary'
    )

    ax2.scatter(
        x[0],
        x[1],
        color='blue',
        label='Samples'
    )

    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    ax2.set_xlim(-r-1, r+1)
    ax2.set_ylim(-r-1, r+1)

    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()


    # eval
    for i, q in enumerate(qs):
        pred_p = robot.fkine(q).t[:2]
        error = np.sqrt((x[0] - pred_p[0]) ** 2 + (x[1] - pred_p[1]) ** 2)
        print(f"error {i + 1} is {error}")



if __name__ == "__main__":
    cfg = FMConfig()
    evaluate(cfg)
