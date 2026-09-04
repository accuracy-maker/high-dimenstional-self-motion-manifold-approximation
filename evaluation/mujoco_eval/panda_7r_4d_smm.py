"""
This is testing the learned self-motion manifold

1. read target pose
2. generate qs candidates
3. clustering as 4-D SMM curves
4. trace the curve and visulaise
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

# clustering
from utils.smm_cluster import wrap_pi, cluster_torus, plot_clusters

@dataclass
class SimulationConfig:
    gravity_compensation: bool = True
    dt: float = 0.002
    n_samples: int = 1000
    n_ode_steps: int = 100
    control_dt: float = 0.2

# load scene
xml_path = "assets/franka_emika_panda/scene.xml"
model = mujoco.MjModel.from_xml_path(xml_path)

data = mujoco.MjData(model)
eval_data = mujoco.MjData(model)

# load flow-matching model
fm_cfg = FMConfig(robot_name="franka_emika_panda_position")
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


def approximate_smm(tgt_pos, fk_tol = 0.005):

    x = tgt_pos

    # generate q candidates
    qs = fm_model.sample(
        x,
        n_samples=SimulationConfig.n_samples,
        n_steps=SimulationConfig.n_ode_steps,
    )

    qs = wrap_pi(qs)

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

    # filter bad samples
    qs_filtered = qs[position_errors < fk_tol]

    # clustering
    labels, k, gap = cluster_torus(qs_filtered, eps = 1.0)

    print(f"labels: {labels}")
    print(f"k : {k}")
    print(f"gap: {gap}")

    return qs_filtered, labels, k, gap

# test
# tgt_pos = data.mocap_pos[mocap_id] # (3,) [x, y, z]
# tgt_quat = data.mocap_quat[mocap_id] # (4,) [w, x, y, z]

# target = np.concatenate([tgt_pos, tgt_quat])
# mujoco.mju_quat2Mat(tgt_R_flat, tgt_quat)
# qs_filtered, labels, k, gap = approximate_smm(tgt_pos, tgt_R_flat.reshape(3, 3))

# plot_clusters(qs_filtered, labels)



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

        target = np.asarray(tgt_pos)

        # re-solve only when the mocap target has moved
        if last_target is None or not np.allclose(target, last_target, atol=1e-6):
            qs_filtered, labels, k, gap = approximate_smm(tgt_pos)
            last_target = target.copy()

            # excute self-motion qs
            for label in np.unique(labels):
                qs_cluster = qs_filtered[labels == label]

                print(f"Executing cluster {label}")

                for q in qs_cluster:
                    if not viewer.is_running():
                        break

                    data.qpos[:7] = q
                    mujoco.mj_forward(model, data)

                    viewer.sync()
                    time.sleep(SimulationConfig.control_dt)



        viewer.sync()

        time_until_next_step = (
            SimulationConfig.dt
            - (time.time() - step_start)
        )

        if time_until_next_step > 0:
            time.sleep(time_until_next_step)