"""
Evaluate position accuracy of learned SMMs for TDCR.
Algorithm:
    1. pick one position in test set
    2. sample solutions from learned FM model
    3. compute predicted position by FK
    4. measure the distance between groundtruth and predictions
"""
import re
import math
import argparse
import time
from pathlib import Path
from tqdm import tqdm

import mujoco
import matplotlib.pyplot as plt
import numpy as np

from model.flow_matching import FMConfig, FlowMatching, load_data

from opencr_mujoco.tdcr_kinematics import MultiSegmentTDCRKinematics

# help functions
def find_tendon_actuators(model):
    """return tendon actuator IDs in segment-major order"""
    found = {}
    pattern = re.compile(r"^seg_(\d+)_ten_(\d+)$")

    for actuator_id in range(model.nu):
        name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id
            )
        match = pattern.match(name or "")

        if match:
            segment = int(match.group(1))
            tendon = int(match.group(2))
            found.setdefault(segment, {})[tendon] = actuator_id

    if not found:
        raise RuntimeError("No  seg_<segment>_ten_<tendon> actuators found")

    n_segments = max(found) + 1
    actuator_ids = []
    tendons_per_segment = []

    for segment in range(n_segments):
        if segment not in found:
            raise RuntimeError(f"Missing tendon actuator for segment {segment}")

        tendon_ids = found[segment]
        expected = list(range(len(tendon_ids)))

        if sorted(tendon_ids) != expected:
            raise RuntimeError(
                f"Tendon indices for segment {segment} are not contiguous: "
                f"{sorted(tendon_ids)}"
                )

        tendons_per_segment.append(len(tendon_ids))
        actuator_ids.extend(tendon_ids[i] for i in expected)

    return actuator_ids, tendons_per_segment

# pretension is the initial tensile force applied to every tendon when
# TDCR is in its neutral configuration
# mainly for maintaining the backbone stable
def read_pretension_keyframe(model):
    """return the pretension keyframe ID and its full atuator-control vector"""
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "pretension")
    if key_id < 0:
        raise RuntimeError("The XML does not contain a 'pretension' keyframe")
    return key_id, model.key_ctrl[key_id].copy()

def get_body_ids(model):
    body_ids = []

    for body_id in range(1, model.nbody):
        body_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id)
        # print(body_name)
        if body_name:
            body_ids.append(body_id)
    body_ids.sort()
    return body_ids

def sample_clarke(rng, radii_mm):
	q = np.zeros(2 * radii_mm.size, dtype=float)
	for segment, radius in enumerate(radii_mm):
		magnitude = radius * math.sqrt(float(rng.random()))
		angle = 2.0 * math.pi * float(rng.random())
		q[2 * segment : 2 * segment + 2] = magnitude * np.array(
		    [math.cos(angle), math.sin(angle)]
		)
	return q	

def tdcr_fk(model, data, qs=None):
    """
    initiate mujoco and data from xml path
    get the distance and offset of this tdcr
    compute the FK when it settles
    """

    # print(f"length of actuator_ids: {len(actuator_ids)}\n"
        # f"length of tendons per segment: {len(tendons_per_segment)}")

    # print(f"actuator_ids:\n {actuator_ids}\n",
        # f"tendons per segment:\n {tendons_per_segment}")


    # print(f"radii_mm: {radii_mm}")
    
    kin = MultiSegmentTDCRKinematics(
        n_tendons_per_segment=tendons_per_segment,
        tendon_distances_mm=distances,
        angle_offsets_rad_ccw=offsets,
        max_bending_angles_rad = MAX_BENDING_ANGLE_RAD
    )

    key_id, pretension_ctrl = read_pretension_keyframe(model)

    settle_steps = max(1, int(round(1.0 / model.opt.timestep)))
    print(f"settle steps:\n {settle_steps}")

    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "EE_pos")

    body_ids = get_body_ids(model)
    
    # compute tdcr position and tip position for each sampled q
    positions = []
    # rng = np.random.default_rng(42)
    # for i in range(10):
    for q in tqdm(qs, desc="compute FK"):
        # print(q)
        delta_tendons_m = kin.clark_to_tendons_mm(q) * 1e-3
        requested_ctrl = pretension_ctrl.copy()
        for actuator_id, delta in zip(actuator_ids,delta_tendons_m):
            requested_ctrl[actuator_id]  += delta
        
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        data.ctrl[:] = pretension_ctrl
        
        for actuator_id, delta in zip(actuator_ids, delta_tendons_m):
            data.ctrl[actuator_id] = pretension_ctrl[actuator_id] + delta
        mujoco.mj_forward(model, data)
        
        for _ in range(settle_steps):
            mujoco.mj_step(model, data)
        positions.append(np.array([data.xpos[body_id] for body_id in body_ids]))
    
    return positions

def plot_positions(positions):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for sample_id in range(positions.shape[0]):
        sample_positions = positions[sample_id]

        ax.plot(
            sample_positions[:, 0],
            sample_positions[:, 1],
            sample_positions[:, 2],
            "-o",
            linewidth=1.2,
            markersize=2,
            alpha=0.7,
        )

    ax.scatter(
        positions[:, 0, 0],
        positions[:, 0, 1],
        positions[:, 0, 2],
        color="green",
        s=30,
        label="Base",
    )

    ax.scatter(
        positions[:, -1, 0],
        positions[:, -1, 1],
        positions[:, -1, 2],
        color="red",
        s=30,
        label="Tip",
    )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("TDCR Skeletons")
    ax.legend()
    ax.set_box_aspect((1, 1, 1))

    plt.show()

def plot_clarke_coordinates(clarke_q, max_radii_mm):
    """
    clarke_q:
        Shape (6,) for one three-segment configuration,
        or shape (N, 6) for multiple configurations.

    max_radii_mm:
        Shape (3,), one maximum radius per segment.
    """
    clarke_q = np.asarray(clarke_q)

    if clarke_q.ndim == 1:
        clarke_q = clarke_q[None, :]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(12, 4),
        constrained_layout=True,
    )

    for segment, ax in enumerate(axes):
        x = clarke_q[:, 2 * segment]
        y = clarke_q[:, 2 * segment + 1]

        radius = max_radii_mm[segment]

        circle = plt.Circle(
            (0.0, 0.0),
            radius,
            fill=False,
            color="black",
            linestyle="--",
            label="Limit",
        )
        ax.add_patch(circle)

        ax.scatter(
            x,
            y,
            s=12,
            alpha=0.5,
        )

        # Draw the final vector if a single configuration is provided
        if len(x) == 1:
            ax.arrow(
                0.0,
                0.0,
                x[0],
                y[0],
                color="red",
                width=0.02 * radius,
                length_includes_head=True,
            )

        ax.axhline(0.0, color="gray", linewidth=0.8)
        ax.axvline(0.0, color="gray", linewidth=0.8)

        ax.set_aspect("equal")
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)
        ax.set_xlabel(r"$q_x$ [mm]")
        ax.set_ylabel(r"$q_y$ [mm]")
        ax.set_title(f"Segment {segment + 1}")
        ax.grid(True, alpha=0.3)

    plt.show()
               
MAX_BENDING_ANGLE_RAD = np.pi / 3
robot_name = "tdcr_3segs"

cfg = FMConfig(robot_name=robot_name)
print(f"cfg:\n {cfg}")
print()

robot_cfg = cfg.load_robot
print(f"robot config:\n {robot_cfg}")
print()

xml_path = robot_cfg.xml_path
print(f"xml path:\n {xml_path}")

# load model and data
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)  

actuator_ids, tendons_per_segment = find_tendon_actuators(model) 
n_segments = len(tendons_per_segment)

distances = [4.5, 4.5, 4.5]
offsets = [-3.0589, -2.5615, -2.0506]
distances = np.asarray(distances, dtype=float)
offsets = np.asarray(offsets, dtype=float)
# print(f"distances: {distances}\n",
    # f"offsets: {offsets}")

radii_mm = distances * MAX_BENDING_ANGLE_RAD

# load dataset
train, test, norm = load_data(cfg)
print(f"norm:\n {norm}")

# generate test samples
test_targets = test.tensors[1].detach().cpu().numpy()
test_targets = np.asarray(test_targets, dtype=np.float64)

x_c = np.asarray(norm["x_c"], dtype=np.float64)
x_h = np.asarray(norm["x_h"], dtype=np.float64)
test_targets = test_targets * x_h + x_c

rng = np.random.default_rng(42)  # fixed seed for reproducibility
num_targets = min(10, len(test_targets))

random_indices = rng.choice(
    len(test_targets),
    size=num_targets,
    replace=False,
)

test_targets = test_targets[random_indices]
if test_targets.ndim == 1:
    test_targets = test_targets[None, :]
print(f"test targets shape: {test_targets.shape}")

x = test_targets[0, :]
print(f"true target position: {x}")

# load fm model
fm = FlowMatching(cfg, norm)
fm.load()
print(fm)

qs = fm.sample(
    x,
    n_samples=20,
    n_steps=100
)
# print(f"qs:\n {qs}")
# plot_clarke_coordinates(
#     qs,
#     radii_mm,
# )



positions = tdcr_fk(model, data, qs)
positions = np.asarray(positions)
tip_positions = positions[:, -1, :]
print(f"tip positions:\n {tip_positions}")
        
# compute FK accuracy
target_errors = np.linalg.norm(tip_positions - x[None,:], axis=1)
print(f"target_errors mean:\n {np.mean(target_errors)}")

# plot position
plot_positions(positions)