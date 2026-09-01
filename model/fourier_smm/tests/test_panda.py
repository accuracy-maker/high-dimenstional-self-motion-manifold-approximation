"""
Test for Fastsmm Panda robot

"""

import time
import numpy as np
import roboticstoolbox as rtb
from realtime_smm.helpers.robot import Robot as RefRobot

from fast_smm import deformation
from robots.panda import (PANDA_TOOL, TASKSPACE, canonical_roll_frame, panda,
                   panda_links, roll_offset, rotz)

n_samples = 50

def section(t):
    print(f"\n--- {t} " + "-" * max(0, 60 - len(t)))

robot = panda()
bk = robot.bk
rng = np.random.default_rng(0)
Q = rng.uniform(-np.pi, np.pi, (n_samples, 7))

section("kinematics")
ref = rtb.models.DH.Panda()
e_fk = max(np.abs(ref.fkine(q).A - bk.fk(q[None])[0]).max() for q in Q)
e_J = max(np.abs(ref.jacob0(q) - bk.jacobian(q[None])[0]).max() for q in Q)
print(f"max |FK  - rtb.models.Panda|    = {e_fk:.2e}")
print(f"max |J   - rtb.jacob0|          = {e_J:.2e}")

Ts = bk.fk(Q)
print(f"Ts shape: {Ts.shape}")
assert Ts.shape[0] == n_samples, "Batched FK is not correct"

# forward_kinematics is built by official code
e_fk = max(np.abs(robot.forward_kinematics(q) - bk.fk(q[None])[0]).max() for q in Q)
e_J = max(np.abs(robot.jacobian(q) - bk.jacobian(q[None])[0]).max() for q in Q)
print(f"max |FK  - realtime_smm.Robot|  = {e_fk:.2e}")
print(f"max |J   - realtime_smm.Robot|  = {e_J:.2e}")

section("orientation-frame bug in the released Robot.fk_err")
q = np.array([0.0, -0.4, 0.0, -2.0, 0.0, 1.6, 0.5])
T = bk.fk(q[None])[0]
seeds = rng.uniform(-np.pi, np.pi, (10, 7))
vanilla = RefRobot(panda_links(), taskspace=TASKSPACE)    # released code, unpatched
n_ok = sum(vanilla.ik(T, q0=s)[0] for s in seeds)         # body-frame error
print(f"IK convergence, released fk_err  log(R_c^T R_d): {n_ok}/10")
_, ok = bk.ik(seeds, np.repeat(T[None], 10, axis=0))       # base-frame error
print(f"IK convergence, fixed   fk_err  log(R_d R_c^T): {int(ok.sum())}/10")

section("symmetries used for the grid reduction")
a, b = 0.37, -0.29
Rz4 = np.eye(4)
Rz4[:3, :3] = rotz(a)
Rt4 = np.eye(4)
Rt4[:3, :3] = rotz(b)
Q1 = Q.copy()
Q1[:, 0] += a
Q7 = Q.copy()
Q7[:, 6] += b
print(f"max |Rz(a).T(q) - T(q + a.e1)|  = {np.abs(Rz4 @ bk.fk(Q) - bk.fk(Q1)).max():.2e}")
print(f"max |T(q).Rz(b) - T(q + b.e7)|  = {np.abs(bk.fk(Q) @ Rt4 - bk.fk(Q7)).max():.2e}")
print("  (the second one is what the released package does not exploit)")

section("null space: eq. (6) minors vs QR")
J = bk.masked_jacobian(Q)
n_eq6, _ = bk.null_vector(J)
Qf, _ = np.linalg.qr(J.transpose(0, 2, 1), mode="complete")
n_qr = Qf[:, :, -1]
print(f"max |1 - |<n_eq6, n_qr>||        = "
      f"{np.abs(np.abs(np.einsum('ij,ij->i', n_eq6, n_qr)) - 1).max():.2e}")
print(f"max |J n| = {np.abs(J @ n_eq6[:, :, None]).max():.2e}")

section("batched SMM solver vs reference solver")
rng2 = np.random.default_rng(5)
Ts = bk.fk(rng2.uniform(-2.0, 2.0, (n_samples, 7)))
t0 = time.time()
fast = robot.workspace_smms_many(Ts, samples=128, step=0.05, smm_iters=1200,
                               seeds=40, chunk=8)
t_fast = (time.time() - t0) / len(Ts)
t0 = time.time()
slow = [RefRobot.workspace_smms(robot, Ts[i], samples=128, step=0.05,
                                sing_thresh=5e-3, smm_iters=1200)
        for i in range(len(Ts))]
t_slow = (time.time() - t0) / len(Ts)
print(f"branches  batched   = {[len(w.data) for w in fast]}")
print(f"branches  reference = {[len(w.data) for w in slow]}")
print(f"time/pose batched   = {t_fast:.3f} s")
print(f"time/pose reference = {t_slow:.3f} s   ->  {t_slow/t_fast:.0f}x")
for w, T_ in zip(fast, Ts):
    if w.status.name != "OK":
        continue
    q = np.concatenate([s.angle.astype(float) for s in w.data], axis=0)
    ep, eo = bk.fk_error_pct(q, np.repeat(T_[None], len(q), axis=0))
    print(f"  traced SMM FK error mean {100*0.5*(ep+eo).mean():.4f} %")
    break