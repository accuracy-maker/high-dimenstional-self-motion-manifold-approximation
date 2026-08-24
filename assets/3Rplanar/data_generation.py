"""
generate a (q, x) dataset from 3R manipulator

For 3R manipulator:
    configuration space: q = [theta_1, theta_2, theta_3]
    workspace: (x,y) (planar motion)

use robotics-toolbox-python for fast prototype
"""

from math import pi
import math
import numpy as np
from scipy.stats import qmc
import matplotlib.pyplot as plt

import roboticstoolbox as rtb

robot = rtb.models.DH.Planar3()
print(robot)

Q_MIN = np.array([-pi, -pi, -pi])
Q_MAX = np.array([pi, pi, pi])

def sobel_joint_samples(n_samples: int, seed = None):
    # scramble: introduce randomness
    sampler = qmc.Sobol(d = 3, scramble = True, seed = seed)
    # sobel generates samples 2^m
    m = math.ceil(math.log2(max(n_samples, 1)))
    # extract first n_samples
    u = sampler.random_base2(m=m)[:n_samples]
    return qmc.scale(u, Q_MIN, Q_MAX)

def generate_dataset(
    n_samples: int,
    seed=None,
    save_path='dataset_3r.npz'
):
    qs = sobel_joint_samples(n_samples = n_samples, seed = seed)
    ps = robot.fkine(qs).t
    # planar motion
    ps = ps[:, :2]
    np.savez(save_path, qs=qs, ps=ps)
    print(f"Saved {n_samples} samples to {save_path}")
    return qs, ps

def plot_space(q, p):
    """ plot configuration space theta2-theta3 plane and workspace"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # theta2-theta3 plane
    ax1.scatter(q[:, 1], q[:, 2], color = 'blue')
    ax1.set_xlabel(r'$\theta_2$ (rad)')
    ax1.set_ylabel(r'$\theta_3$ (rad)')
    ax1.set_xlim(Q_MIN[0], Q_MAX[0])
    ax1.set_ylim(Q_MIN[0], Q_MAX[0])

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

n = 100000
qs, p = generate_dataset(n, 42, save_path="./planar3r.npz")
# plot_space(qs, p)