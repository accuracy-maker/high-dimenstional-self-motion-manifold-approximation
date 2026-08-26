"""
generate a (q, x) dataset from 3R manipulator

For 3R manipulator:
    configuration space: q = [theta_1, theta_2, theta_3]
    workspace: (x,y) (planar motion)

use robotics-toolbox-python for fast prototype
"""

from dataclasses import dataclass
from math import pi
import math
import numpy as np
from scipy.stats import qmc
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation # R -> quaternions

import roboticstoolbox as rtb


ROOT_PATH = Path(__file__).resolve().parents[0]
# print(f"root path: {ROOT_PATH}")


@dataclass(frozen=True)
class RobotConfig:
    name: str
    robot: object
    save_path: Path

    @property
    def q_min(self) -> np.ndarray:
        return self.robot.qlim[0]

    @property
    def q_max(self) -> np.ndarray:
        return self.robot.qlim[1]

    @property
    def q_dim(self) -> int:
        return self.robot.n


ROBOT_CONFIGS = {
    "3R": RobotConfig(
        name="3R",
        robot=rtb.models.DH.Planar3(),
        save_path=ROOT_PATH / "3Rplanar" / "planar3r.npz",
    ),
    "7R": RobotConfig(
        name="7R",
        robot=rtb.models.DH.Panda(),
        save_path=ROOT_PATH / "7R" / "7r.npz",
    ),
}


def get_robot_config(robot_name: str) -> RobotConfig:
    try:
        return ROBOT_CONFIGS[robot_name]
    except KeyError:
        raise NameError("The model name is invalid.")


def print_robot_config(config: RobotConfig):
    print(f"{config.name} manipulator is loaded")
    print(config.robot)
    print(f'minimum joint limits in rad:')
    print(config.q_min)
    print(f'maximum joint limits in rad:')
    print(config.q_max)
    print(f'dataset will be saved at: {config.save_path}')


def sobel_joint_samples(
    n_samples: int,
    config: RobotConfig,
    seed=None,
):
    # scramble: introduce randomness
    sampler = qmc.Sobol(
        d=config.q_dim,
        scramble=True,
        seed=seed,
    )

    # sobel generates samples 2^m
    m = math.ceil(math.log2(max(n_samples, 1)))

    # extract first n_samples
    u = sampler.random_base2(m=m)[:n_samples]

    return qmc.scale(
        u,
        config.q_min,
        config.q_max,
    )


def generate_dataset(
    n_samples: int,
    config: RobotConfig,
    seed=None,
):
    qs = sobel_joint_samples(
        n_samples=n_samples,
        config=config,
        seed=seed,
    )

    if config.name == "3R":
        # planar motion
        ps = config.robot.fkine(qs).t
        ps = ps[:, :2]
        np.savez(config.save_path, qs=qs, ps=ps)
        print(f"Saved {n_samples} samples to {config.save_path}")
        return qs, ps

    elif config.name == "7R":
        # full pose (x, y, z, quat) (7,)
        # transform matrix
        T = config.robot.fkine(qs)

        # position
        position = T.t # (N,3)

        # quaternions
        quat = Rotation.from_matrix(T.R).as_quat()

        # full pose
        pos = np.concatenate([position, quat], axis=1)
        assert pos.shape[1] == 7

        # save
        np.savez(config.save_path, qs=qs, pos=pos)
        print(f"Saved {n_samples} samples to {config.save_path}")
        return qs, pos

    else:
        raise NameError("The model name is invalid.")


def plot_space(
    config: RobotConfig,
    q: np.ndarray,
    p: np.ndarray,
):
    if config.name == "3R":
        # plot configuration space theta2-theta3 plane and workspace
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # theta2-theta3 plane
        ax1.scatter(q[:, 1], q[:, 2], color = 'blue')
        ax1.set_xlabel(r'$\theta_2$ (rad)')
        ax1.set_ylabel(r'$\theta_3$ (rad)')
        ax1.set_xlim(config.q_min[0], config.q_max[0])
        ax1.set_ylim(config.q_min[0], config.q_max[0])

        ax1.set_aspect('equal', adjustable='box')
        ax1.grid(True, alpha=0.3)
        
        # draw the workspace boundary
        # Draw theoretical workspace boundary
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

        # Plot FK positions
        ax2.scatter(
            p[:, 0],
            p[:, 1],
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
    
    elif config.name == "7R":
        # High-level layout:
        # left  = 3x3 joint-space histograms
        # right = 3D workspace
        fig = plt.figure(figsize=(16, 7))

        outer = fig.add_gridspec(
            1, 2,
            width_ratios=[1.2, 1.0],
            wspace=0.25,
        )

        # ---------------------------------------------------------
        # Left: 3x3 grid containing histograms for the 7 joints
        # ---------------------------------------------------------
        joint_grid = outer[0].subgridspec(
            3, 3,
            hspace=0.5,
            wspace=0.35,
        )

        for i in range(config.q_dim):
            ax = fig.add_subplot(joint_grid[i // 3, i % 3])

            ax.hist(q[:, i], bins=30)

            ax.set_title(rf"$q_{i+1}$")
            ax.set_xlabel("Joint angle (rad)")
            ax.set_ylabel("Count")
            ax.set_xlim(config.q_min[i], config.q_max[i])
            ax.grid(True, alpha=0.3)

        # Turn the two unused cells off
        for i in range(config.q_dim, 9):
            ax = fig.add_subplot(joint_grid[i // 3, i % 3])
            ax.axis("off")

        # ---------------------------------------------------------
        # Right: 3D workspace
        # ---------------------------------------------------------
        ax_workspace = fig.add_subplot(outer[1], projection="3d")

        ax_workspace.scatter(
            p[:, 0],
            p[:, 1],
            p[:, 2],
            s=15,
            # alpha=0.5,
        )

        ax_workspace.set_xlabel("x (m)")
        ax_workspace.set_ylabel("y (m)")
        ax_workspace.set_zlabel("z (m)")
        ax_workspace.set_title("Workspace")

        plt.show()
    
    else:
        raise NameError("The model name is invalid.")


if __name__ == "__main__":
    robot_name = "7R"
    config = get_robot_config(robot_name)
    print_robot_config(config)

    n = 10_000
    qs, pos = generate_dataset(
        n_samples=n,
        config=config,
        seed=42,
    )

    plot_space(config, qs, pos)