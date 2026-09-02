"""
Test for Fastsmm Panda robot

"""

import time
import numpy as np
import mujoco
from realtime_smm.helpers.robot import Robot as RefRobot

from fast_smm import deformation
from robots.panda import (canonical_roll_frame, roll_offset, rotz)
from robots.iiwa import iiwa_links, iiwa


def section(t):
    print(f"\n--- {t} " + "-" * max(0, 60 - len(t)))

def mujoco_fk(model, data, q):
    T = np.eye(4)
    data.qpos[:] = q
    mujoco.mj_forward(model, data)

    ee_pos = data.site("attachment_site").xpos.copy()
    ee_mat = data.site("attachment_site").xmat.reshape(3, 3).copy()

    T[:3, 3] = ee_pos
    T[:3, :3] = ee_mat.reshape(3,3)

    return T

def mujoco_jaco(model, data, q):
    data.qpos[:] = q
    mujoco.mj_forward(model, data)

    site_id = model.site("attachment_site").id

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    mujoco.mj_jacSite(
        model,
        data,
        jacp,
        jacr,
        site_id,
    )

    J = np.vstack((jacp, jacr))

    return J

n_samples = 50
xml_path = "/home/z5506409/high-dimenstional-self-motion-manifold-approximation/assets/kuka_iiwa_14/iiwa14.xml"


robot = iiwa()
bk = robot.bk
rng = np.random.default_rng(0)
Q = rng.uniform(-np.pi, np.pi, (n_samples, 7))

section("kinematics")
# test if kinematics of FastSMM Robot is same with mujoco iiwa14

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

e_fk = max(np.abs(mujoco_fk(model,data,q) - bk.fk(q[None])[0]).max() for q in Q)
e_J = max(np.abs(mujoco_jaco(model,data,q) - bk.jacobian(q[None])[0]).max() for q in Q)
print(f"max |FK  - mujoco FK iiwa|    = {e_fk:.2e}")
print(f"max |J   - mujoco jaco iiwa|  = {e_J:.2e}")

