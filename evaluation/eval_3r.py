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
    # x = np.array([1, 1/2])
    x = data['ps'][10]
    print(f"target postion: {x}")
    qs = fm.sample(x, n_samples=1000, n_steps=100)
    print("sampling qs:\n")
    print(qs)


    # eval
    errors = np.linalg.norm(robot.fkine(qs).t[:,:2] - x[None, :], axis=1)

    print("Mean FK error:", errors.mean())
    print("Median FK error:", np.median(errors))
    print("Max FK error:", errors.max())
    print("95% FK error:", np.percentile(errors, 95))

    # plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))

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


    # histplot of error
    ax3.hist(errors, bins=20)

    plt.tight_layout()
    plt.show()


    



if __name__ == "__main__":
    cfg = FMConfig()
    evaluate(cfg)
