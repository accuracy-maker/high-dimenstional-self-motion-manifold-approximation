"""
This is an evaluation of generative IK for panda 7R arm

1. read the traget pose
2. generate a batch of config candidates
3. sort them based on error
4. pick and excute the best one
"""

# mujoco
import mujoco
import mujoco.viewer

# basic
import time
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt

# torch
import torch

# FM model
from model.flow_matching import FMConfig, FlowMatching, load_data


@dataclass
class SimulationConfig:
    gravity_compensation: bool = True
    dt: float = 0.002
    n_samples: int = 100
    n_ode_steps: int = 100


# load scene
xml_path = "assets/franka_emika_panda/scene.xml"
model = mujoco.MjModel.from_xml_path(xml_path)

data = mujoco.MjData(model)
eval_data = mujoco.MjData(model)

# load flow-matching model
fm_cfg = FMConfig(robot_name="franka_emika_panda")
_, _, norm = load_data(fm_cfg)

fm_model = FlowMatching(fm_cfg, norm)
fm_model.load()

# extract EE info
site_id = model.site("attachment_site").id

# gravity compensation
model.body_gravcomp[:] = float(SimulationConfig.gravity_compensation)

# read the dof and actuators
joint_names = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
]

dof_ids = np.array([
    model.joint(name).id
    for name in joint_names
])

actuator_ids = np.array([
    model.actuator(name).id
    for name in joint_names
])

# Initial joint configuration saved as a keyframe in the XML file.
key_name = "home"
key_id = model.key(key_name).id
q0 = model.key(key_name).qpos

# Mocap body we will control with our mouse.
mocap_name = "target"
mocap_id = model.body(mocap_name).mocapid[0]

# simulation dt
model.opt.timestep = SimulationConfig.dt

# pre-allocate vars
tgt_R_flat = np.zeros(9, dtype=np.float64)


def solve_best_q(tgt_pos, tgt_R):
    """Rank candidates offscreen and return only the best one."""

    x = np.concatenate([
        tgt_pos,
        tgt_R[:, 0],
        tgt_R[:, 1],
    ])

    # generate q candidates
    qs = fm_model.sample(
        x,
        n_samples=SimulationConfig.n_samples,
        n_steps=SimulationConfig.n_ode_steps,
    )

    # Compute FK error for all candidates
    eval_data.qpos[:] = data.qpos

    positions = np.zeros((qs.shape[0], 3))
    rotations = np.zeros((qs.shape[0], 3, 3))

    for i, q in enumerate(qs):
        eval_data.qpos[:7] = q
        mujoco.mj_forward(model, eval_data)

        positions[i] = eval_data.site_xpos[site_id]
        rotations[i] = eval_data.site_xmat[site_id].reshape(3, 3)

    # position error
    position_errors = np.linalg.norm(
        positions - tgt_pos[None, :],
        axis=1,
    )

    # orientation error
    R_error = tgt_R.T @ rotations

    cos_angle = (np.trace(R_error, axis1=1, axis2=2) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    orientation_errors = np.arccos(cos_angle)

    # combined error
    total_length = 1.2

    errors = 0.5 * (
        position_errors / total_length
        + orientation_errors / np.pi
    )

    # sort candidates
    return qs[int(np.argmin(errors))]


# viewer launch
with mujoco.viewer.launch_passive(
    model=model,
    data=data,
    show_left_ui=False,
    show_right_ui=False,
) as viewer:

    # reset the simulation
    mujoco.mj_resetDataKeyframe(model, data, key_id)

    # reset free camera
    mujoco.mjv_defaultFreeCamera(model, viewer.cam)

    # Enable site frame visualization.
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

    best_q = np.array(q0[:7], dtype=np.float64)
    last_target = None

    # control
    while viewer.is_running():
        step_start = time.time()

        tgt_pos = data.mocap_pos[mocap_id] # (3,) [x, y, z]
        tgt_quat = data.mocap_quat[mocap_id] # (4,) [w, x, y, z]

        target = np.concatenate([tgt_pos, tgt_quat])

        # re-solve only when the mocap target has moved
        if last_target is None or not np.allclose(target, last_target, atol=1e-6):
            mujoco.mju_quat2Mat(tgt_R_flat, tgt_quat)
            best_q = solve_best_q(tgt_pos, tgt_R_flat.reshape(3, 3))
            last_target = target

        # excute best q
        data.ctrl[actuator_ids] = best_q

        mujoco.mj_step(model, data)

        viewer.sync()

        time_until_next_step = (
            SimulationConfig.dt
            - (time.time() - step_start)
        )

        if time_until_next_step > 0:
            time.sleep(time_until_next_step)