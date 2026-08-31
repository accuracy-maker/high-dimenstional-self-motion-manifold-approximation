"""
This is an evaluation of generative IK for iiwa 7R arm

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

    # Cartesian impedance control gains
    impedance_pos: tuple = (100.0, 100.0, 100.0)
    impedance_ori: tuple = (50.0, 50.0, 50.0)

    # joint impedance control gains (nullspace)
    kp_null: tuple = (75.0, 75.0, 50.0, 50.0, 40.0, 25.0, 25.0)

    # damping ratio for both Cartesian and joint impedance control
    damping_ratio: float = 1.0

    # gains for the twist computation
    kpos: float = 0.95
    kori: float = 0.95

    # integration timestep
    integration_dt: float = 1.0

# load scene
xml_path = "assets/kuka_iiwa_14/scene.xml"
model = mujoco.MjModel.from_xml_path(xml_path)

data = mujoco.MjData(model)
eval_data = mujoco.MjData(model)

# load flow-matching model
fm_cfg = FMConfig(robot_name="kuka_iiwa_14")
_, _, norm = load_data(fm_cfg)

fm_model = FlowMatching(fm_cfg, norm)
fm_model.load()

# extract EE info
site_id = model.site("attachment_site").id

# impedance gains and damping
Kp = np.concatenate([SimulationConfig.impedance_pos, SimulationConfig.impedance_ori])
Kd = SimulationConfig.damping_ratio * 2 * np.sqrt(Kp)
Kp_null = np.asarray(SimulationConfig.kp_null)
Kd_null = SimulationConfig.damping_ratio * 2 * np.sqrt(Kp_null)

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

jac = np.zeros((6, model.nv))
twist = np.zeros(6)
site_quat = np.zeros(4)
site_quat_conj = np.zeros(4)
error_quat = np.zeros(4)
M_inv = np.zeros((model.nv, model.nv))
Mx = np.zeros((6, 6))

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
        # Cartesian impedance toward the mocap target
        dx = data.mocap_pos[mocap_id] - data.site(site_id).xpos
        twist[:3] = SimulationConfig.kpos * dx / SimulationConfig.integration_dt
        mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
        mujoco.mju_negQuat(site_quat_conj, site_quat)
        mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
        mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
        twist[3:] *= SimulationConfig.kori / SimulationConfig.integration_dt

        # task-space inertia and generalized forces
        mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
        mujoco.mj_solveM(model, data, M_inv, np.eye(model.nv))
        Mx_inv = jac @ M_inv @ jac.T
        if abs(np.linalg.det(Mx_inv)) >= 1e-2:
            Mx = np.linalg.inv(Mx_inv)
        else:
            Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)

        tau = jac.T @ Mx @ (Kp * twist - Kd * (jac @ data.qvel[dof_ids]))

        # joint task in the nullspace: track the FM solution
        Jbar = M_inv @ jac.T @ Mx
        ddq = Kp_null * (best_q - data.qpos[dof_ids]) - Kd_null * data.qvel[dof_ids]
        tau += (np.eye(model.nv) - jac.T @ Jbar.T) @ ddq

        # gravity compensation (iiwa motors are torque-controlled)
        if SimulationConfig.gravity_compensation:
            tau += data.qfrc_bias[dof_ids]

        np.clip(tau, *model.actuator_ctrlrange.T, out=tau)
        data.ctrl[actuator_ids] = tau[actuator_ids]

        mujoco.mj_step(model, data)

        viewer.sync()

        time_until_next_step = (
            SimulationConfig.dt
            - (time.time() - step_start)
        )

        if time_until_next_step > 0:
            time.sleep(time_until_next_step)