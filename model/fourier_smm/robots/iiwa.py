"""
Standard DH table of iiwa14
"""

import numpy as np
from realtime_smm import DHLink, TaskSpace
from ..fast_smm import FastRobot
from ..robots.panda import rotz, SE3

def iiwa14_standard_dh():
    """
    Standard DH parameters for KUKA LBR iiwa14.
    Convention:
        A_i = Rz(theta_i) @ Tz(d_i) @ Tx(a_i) @ Rx(alpha_i)

    Returns
    -------
    a : (7,)
    alpha : (7,)
    d : (7,)
    theta_offset : (7,)

    Actual joint angle:
        theta_i = q_i + theta_offset[i]
    """
    a = np.zeros(7)

    alpha = np.array([
        -np.pi / 2,
        +np.pi / 2,
        +np.pi / 2,
        -np.pi / 2,
        -np.pi / 2,
        +np.pi / 2,
        0.0,
    ])

    d = np.array([
        0.360,
        0.0,
        0.420,
        0.0,
        0.400,
        0.0,
        0.081,
    ])

    theta = np.array([
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ])

    return a, alpha, d, theta

a, alpha, d, theta = iiwa14_standard_dh()
# print(f"a = {a}")
# print(f"alpha = {alpha}")
# print(f"d = {d}")
# print(f"theta = {theta}")

def iiwa_links(joint_limits=(-np.pi, np.pi)):
    lo, hi = joint_limits
    return [DHLink(
        a = float(a[i]),
        alpha = float(alpha[i]),
        d = float(d[i]),
        theta=theta[i],
        lower_limit=lo,
        upper_limit=hi
    ) for i in range(7)]

TASKSPACE = TaskSpace.X | TaskSpace.Y | TaskSpace.Z | TaskSpace.SO3
IIWA_TOOL = SE3(
    rotz(np.pi),
    np.array([0.0, 0.0, 0.045])
)

def iiwa(joint_limits = (-np.pi, np.pi)) -> FastRobot:
    return FastRobot(iiwa_links(joint_limits), taskspace=TASKSPACE, tool=IIWA_TOOL)

