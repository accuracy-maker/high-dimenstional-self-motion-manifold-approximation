"""
Test single and batched planar 3R manipulator
"""

from robots.planar3r import planar3r, planar3r_links, TASKSPACE

import numpy as np

import roboticstoolbox as rtb

from realtime_smm.helpers.robot import Robot as RefRobot

# help function
def section(t):
    print(f"\n--- {t} " + "-" * max(0, 60 - len(t)))

robot = planar3r()

# batched kinematics
bk = robot.bk
rng = np.random.default_rng(0)
# a batch of qs
Q = rng.uniform(-np.pi, np.pi, (20, 3))

section("kinematics")
ref = rtb.models.DH.Planar3()
e_fk = max(np.abs(ref.fkine(q).A - bk.fk(q[None])[0]).max() for q in Q)
e_J = max(np.abs(ref.jacob0(q) - bk.jacobian(q[None])[0]).max() for q in Q)
print(f"max |FK  - rtb.models.Panda|    = {e_fk:.2e}")
print(f"max |J   - rtb.jacob0|          = {e_J:.2e}")

Ts = bk.fk(Q)
assert Ts.shape[0] == 20, "Batched FK is not correct"

section("orientation frame bug in the released Robot.fk_err")

q = np.array([1.0, 1.0, 1.0])
T = bk.fk(q[None])[0]
seeds = rng.uniform(-np.pi, np.pi, (10, 3))

vanilla = RefRobot(planar3r_links(), taskspace=TASKSPACE)  
n_ok = sum(vanilla.ik(T, q0=s)[0] for s in seeds)
print(f"IK convergence, released fk_err  log(R_c^T R_d): {n_ok}/10")

_, ok = bk.ik(seeds, np.repeat(T[None], 10, axis=0))       # base-frame error
print(f"IK convergence, fixed   fk_err  log(R_d R_c^T): {int(ok.sum())}/10")

print("both are converged because there are not orientation components in planar 3R case")