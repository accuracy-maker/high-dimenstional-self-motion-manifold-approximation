"""
standard DH parameters for Franka Emika Panda Robot

"""

import numpy as np
from realtime_smm import DHLink, TaskSpace
from ..fast_smm import FastRobot

"""
modified-DH table of roboticstoolbox.models.DH.Panda ...
check it out by:
    import roboticstoolbox as rtb
    robot = rtb.models.DH.Panda()
    print(robot)
DHRobot: Panda (by Franka Emika), 7 joints (RRRRRRR), dynamics, geometry, modified DH parameters
┌─────────┬────────┬─────┬───────┬─────────┬────────┐
│  aⱼ₋₁   │  ⍺ⱼ₋₁  │ θⱼ  │  dⱼ   │   q⁻    │   q⁺   │
├─────────┼────────┼─────┼───────┼─────────┼────────┤
│     0.0 │   0.0° │  q1 │ 0.333 │ -166.0° │ 166.0° │
│     0.0 │ -90.0° │  q2 │   0.0 │ -101.0° │ 101.0° │
│     0.0 │  90.0° │  q3 │ 0.316 │ -166.0° │ 166.0° │
│  0.0825 │  90.0° │  q4 │   0.0 │ -176.0° │  -4.0° │
│ -0.0825 │ -90.0° │  q5 │ 0.384 │ -166.0° │ 166.0° │
│     0.0 │  90.0° │  q6 │   0.0 │   -1.0° │ 215.0° │
│   0.088 │  90.0° │  q7 │ 0.107 │ -166.0° │ 166.0° │
└─────────┴────────┴─────┴───────┴─────────┴────────┘

┌──────┬───────────────────────────────────────┐
│ tool │ t = 0, 0, 0.1; rpy/xyz = -45°, 0°, 0° │
└──────┴───────────────────────────────────────┘

┌──────┬─────┬────────┬─────┬───────┬─────┬───────┬──────┐
│ name │ q0  │ q1     │ q2  │ q3    │ q4  │ q5    │ q6   │
├──────┼─────┼────────┼─────┼───────┼─────┼───────┼──────┤
│   qr │  0° │ -17.2° │  0° │ -126° │  0° │  115° │  45° │
│   qz │  0° │  0°    │  0° │  0°   │  0° │  0°   │  0°  │
└──────┴─────┴────────┴─────┴───────┴─────┴───────┴──────┘

the transform matrix of tool is:
print(robot.tool)

0.7071    0.7071    0         0         
-0.7071    0.7071   0         0         
0         0         1         0.103     
0         0         0         1      

"""
A_MDH = np.array([0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088])
ALPHA_MDH = np.array([0.0, -np.pi / 2, np.pi / 2, np.pi / 2,
                      -np.pi / 2, np.pi / 2, np.pi / 2])
D_MDH = np.array([0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.107])

# ... converted to the standard DH convention used by realtime_smm:
#     a_i = a_mdh[i+1], alpha_i = alpha_mdh[i+1], d_i = d_mdh[i]
A_STD = np.append(A_MDH[1:], 0.0)
ALPHA_STD = np.append(ALPHA_MDH[1:], 0.0)
D_STD = D_MDH.copy()

# 7Dof for 6-D pose
TASKSPACE = TaskSpace.X | TaskSpace.Y | TaskSpace.Z | TaskSpace.SO3

# help functions

# rotation wrt axis z by angle a
def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

# SE(3) matrix
def SE3(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T

# panda tool
PANDA_TOOL = SE3(rotz(-np.pi / 4.0), np.array([0.0, 0.0, 0.1034]))

# for fourier method, we should use the true joint limits
# because we need to trace the a closed curve
def panda_links(joint_limits=(-np.pi, np.pi)):
    lo, hi = joint_limits
    return [DHLink(a=float(A_STD[i]), alpha=float(ALPHA_STD[i]),
                   d=float(D_STD[i]), theta=0.0,
                   lower_limit=lo, upper_limit=hi) for i in range(7)]

def panda(joint_limits=(-np.pi, np.pi)) -> FastRobot:
    return FastRobot(panda_links(joint_limits), taskspace=TASKSPACE, tool=PANDA_TOOL)

def canonical_roll_frame(a_hat):
    """
    In terms of Panda or iiwa robots, the last joint just provides roll not local z-axis rotation
    We can reduce that dimention when sampling
    """
    a = np.asarray(a_hat, float)
    a = a / np.linalg.norm(a)
    ref = np.array([0.0, 0.0, 1.0])
    x = ref - np.dot(ref, a) * a
    if np.linalg.norm(x) < 1e-8:                     # a_hat parallel to world z
        ref = np.array([1.0, 0.0, 0.0])
        x = ref - np.dot(ref, a) * a
    x /= np.linalg.norm(x)
    y = np.cross(a, x)
    return np.column_stack([x, y, a])


def roll_offset(R):
    """
    Angle beta such that R @ Rz(-beta) has the canonical roll, i.e. the amount
    of tool-z rotation that must be removed from R (and later added to q7).
    """
    Rc = canonical_roll_frame(R[:, 2])
    M = Rc.T @ R # rotation about z by beta
    return float(np.arctan2(M[1, 0], M[0, 0]))

def fibonacci_sphere(n):
    """`n` approximately equispaced unit vectors on S^2."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    golden = np.pi * (1.0 + 5.0 ** 0.5)
    theta = golden * i
    return np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(phi)], axis=1)