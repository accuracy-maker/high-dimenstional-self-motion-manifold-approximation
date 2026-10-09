"""
Mujoco Simulation about Self-Motion for a 3 segments Tendon Driven Continuum Robots
for position-level task
"""

# mujoco
import mujoco
import mujoco.viewer

# basic
import time
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt

# Flow-matching model
from model.flow_matching import FMConfig, FlowMatching, load_data

# Tendon Driven Continuum Robot
from opencr_mujoco.tdcr_kinematics import MultiSegmentTDCRKinematics
from evaluation.eval_tdcr import find_tendon_actuators, read_pretension_keyframe, get_body_ids, tdcr_fk


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

# load data
train, test, norm = load_data(cfg)
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

x = test_targets[0, :]
print(f"true target position: {x}")

# load fm model
fm = FlowMatching(cfg, norm)
fm.load()		

qs = fm.sample(
    x,
    n_samples=10,
    n_steps=100
)

# viewer launch
with mujoco.viewer.launch_passive(
	model = model,
	data = data,
	show_left_ui = False,
	show_right_ui = False
) as viewer:
	# reset the simulation
	mujoco.mj_resetDataKeyframe(model, data, key_id)

	# reset free camera
	mujoco.mjv_defaultFreeCamera(model, viewer.cam)

	# enable site frame visulaisation
	viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

	# control
	while viewer.is_running():
		for q in qs:
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

			viewer.sync()

			time.sleep(0.5)