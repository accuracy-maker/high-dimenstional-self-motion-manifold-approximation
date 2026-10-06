"""Evaluate flow-matching IK for the 13-D Franka--TDCR representation.

The configuration stored by the dataset generator is

    q = [q_panda_1, ..., q_panda_7,
         c_0x, c_0y, c_1x, c_1y, c_2x, c_2y]

The first seven values are Panda joint angles.  The last six values are
Clark coordinates for three TDCR segments.  They are converted to the nine
tendon commands using the same OpenCR-MuJoCo kinematics class used by the
dataset generator.

The stored task-space convention is

    x = [px, py, pz, R00, R01, R10, R11, R20, R21].

Important evaluator behavior:

* The target sample is evaluated strictly.  A mismatch between the saved
  target pose and MuJoCo FK is an actual configuration error.
* Generated flow-matching samples are not assumed to be physically valid.
  Samples outside the Panda limits, outside the three Clark disks, or that
  make MuJoCo unstable are rejected individually.
* One bad generated sample therefore does not abort evaluation.  The valid
  sample count is reported because it is part of the learned-model quality.

Run from the project root, for example:

    python -m evaluation.eval_franka_tdcr \\
        --robot_name franka_tdcr \\
        --target-index 0 \\
        --n_samples 2000 \\
        --n_steps 100 \\
        --ramp-time 0.05 \\
        --settle-time 0.20

The ramp and settle times must match dataset generation.
"""

import argparse
from collections import Counter

import mujoco
import numpy as np
from tqdm import tqdm

from assets.data_generation import mujoco_fk
from model.flow_matching import FMConfig, FlowMatching, load_data
from opencr_mujoco.tdcr_kinematics import (
    ThreeTendonThreeSegmentTDCRKinematics,
)


N_PANDA = 7
N_TDCR_SEGMENTS = 3
N_TENDONS_PER_SEGMENT = 3
N_TENDONS = N_TDCR_SEGMENTS * N_TENDONS_PER_SEGMENT
N_CLARK = 2 * N_TDCR_SEGMENTS
N_REDUCED_Q = N_PANDA + N_CLARK
N_X = 9


class InvalidSimulation(RuntimeError):
    """Raised when a requested configuration cannot be simulated safely."""


MUJOCO_FATAL_ERROR = getattr(mujoco, "FatalError", RuntimeError)
SIMULATION_ERRORS = (
    InvalidSimulation,
    MUJOCO_FATAL_ERROR,
    FloatingPointError,
    OverflowError,
)


def get_panda_joint_info(model):
    """Return Panda IDs and qpos/qvel indices in Panda joint order."""

    joint_ids = []
    qpos_indices = []
    qvel_indices = []
    joint_names = []

    for joint_number in range(1, N_PANDA + 1):
        actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            f"panda_joint{joint_number}",
        )

        if actuator_id < 0:
            actuator_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                f"panda0_joint{joint_number}",
            )

        if actuator_id < 0:
            raise RuntimeError(
                f"Could not find Panda actuator for joint {joint_number}."
            )

        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id < 0:
            raise RuntimeError(
                f"Panda actuator {actuator_id} is not attached to a joint."
            )

        joint_ids.append(joint_id)
        qpos_indices.append(int(model.jnt_qposadr[joint_id]))
        qvel_indices.append(int(model.jnt_dofadr[joint_id]))
        joint_names.append(
            mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_id,
            )
        )

    return (
        np.asarray(joint_ids, dtype=int),
        np.asarray(qpos_indices, dtype=int),
        np.asarray(qvel_indices, dtype=int),
        joint_names,
    )


def get_panda_joint_limits(model, panda_joint_ids):
    """Read finite Panda limits in panda_joint1,...,panda_joint7 order."""

    lower = []
    upper = []

    for joint_id in panda_joint_ids:
        if not model.jnt_limited[joint_id]:
            name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                int(joint_id),
            )
            raise ValueError(
                f"Panda joint '{name}' does not have finite limits."
            )

        lo, hi = model.jnt_range[joint_id]
        lower.append(float(lo))
        upper.append(float(hi))

    return np.asarray(lower), np.asarray(upper)


def get_tendon_actuator_ids(model):
    """Return seg_0_ten_0,...,seg_2_ten_2 actuator IDs."""

    actuator_ids = []

    for segment_id in range(N_TDCR_SEGMENTS):
        for tendon_id in range(N_TENDONS_PER_SEGMENT):
            name = f"seg_{segment_id}_ten_{tendon_id}"
            actuator_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                name,
            )

            if actuator_id < 0:
                raise RuntimeError(
                    f"Could not find tendon actuator '{name}'."
                )

            actuator_ids.append(int(actuator_id))

    return np.asarray(actuator_ids, dtype=int)


def get_keyframe_id(model, name):
    """Return a named MuJoCo keyframe ID, or -1."""

    for key_id in range(model.nkey):
        key_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_KEY,
            key_id,
        )

        if key_name == name:
            return int(key_id)

    return -1


def get_initial_state(model):
    """Read the exact pretension state used by dataset generation."""

    key_id = get_keyframe_id(model, "pretension")
    if key_id < 0:
        raise RuntimeError(
            "The scene does not contain a keyframe named 'pretension'. "
            "Evaluation must use the same initialization as generation."
        )

    qpos = model.key_qpos[key_id].copy()
    qvel = model.key_qvel[key_id].copy()
    ctrl = model.key_ctrl[key_id].copy()

    if model.na > 0:
        act = model.key_act[key_id].copy()
    else:
        act = np.zeros(0, dtype=float)

    return qpos, qvel, act, ctrl


def reset_data(model, data, qpos, qvel, act, ctrl):
    """Reset all state that can affect an independent FK evaluation."""

    data.time = 0.0
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    data.ctrl[:] = ctrl

    if model.na > 0:
        data.act[:] = act

    data.qacc[:] = 0.0
    data.qfrc_applied[:] = 0.0
    data.xfrc_applied[:] = 0.0

    mujoco.mj_forward(model, data)


def get_robot_kinematics(robot_cfg):
    """Construct the same three-segment Clark model used by generation."""

    num_segments = getattr(
        robot_cfg,
        "clark_num_segments",
        N_TDCR_SEGMENTS,
    )
    if num_segments != N_TDCR_SEGMENTS:
        raise ValueError(
            "This evaluator expects exactly three TDCR segments."
        )

    tendon_distance_mm = getattr(
        robot_cfg,
        "clark_tendon_distance_mm",
        4.0,
    )
    max_bending_angle_deg = getattr(
        robot_cfg,
        "tdcr_limit_deg",
        45.0,
    )
    angle_offsets_deg = getattr(
        robot_cfg,
        "clark_angle_offsets_deg",
        (0.0, 30.0, 60.0),
    )

    if len(angle_offsets_deg) != N_TDCR_SEGMENTS:
        raise ValueError(
            "Expected exactly three Clark angle offsets."
        )

    return ThreeTendonThreeSegmentTDCRKinematics(
        tendon_distance_mm=tendon_distance_mm,
        angle_offset_rad_ccw=np.deg2rad(angle_offsets_deg),
        max_bending_angle_rad=np.deg2rad(max_bending_angle_deg),
    )


def clark_radius_mm(robot_cfg):
    """Return the radius of each physically valid Clark-coordinate disk."""

    tendon_distance_mm = float(
        getattr(robot_cfg, "clark_tendon_distance_mm", 4.0)
    )
    max_angle_deg = float(
        getattr(robot_cfg, "tdcr_limit_deg", 45.0)
    )
    return tendon_distance_mm * np.deg2rad(max_angle_deg)


def reduced_q_is_in_domain(
    q,
    panda_low,
    panda_high,
    clark_radius,
    tolerance=1e-9,
):
    """Check the Panda box and three Clark disks before calling mj_step."""

    q = np.asarray(q, dtype=float)
    if q.shape != (N_REDUCED_Q,):
        return False
    if not np.all(np.isfinite(q)):
        return False

    if np.any(q[:N_PANDA] < panda_low - tolerance):
        return False
    if np.any(q[:N_PANDA] > panda_high + tolerance):
        return False

    # Clark coordinates are polar bending coordinates.  The valid set is a
    # disk ||c_s|| <= d * theta_max, not the enclosing square.
    for segment_id in range(N_TDCR_SEGMENTS):
        start = N_PANDA + 2 * segment_id
        c = q[start:start + 2]
        if np.linalg.norm(c) > clark_radius + tolerance:
            return False

    return True


def get_tip_pose(model, data, robot_cfg):
    """Return [position, first two columns of the EE rotation matrix]."""

    if robot_cfg.ee_type == "body":
        ee_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            robot_cfg.ee_name,
        )
        if ee_id < 0:
            raise RuntimeError(
                f"End-effector body '{robot_cfg.ee_name}' was not found."
            )
        position = data.xpos[ee_id].copy()
        rotation = data.xmat[ee_id].reshape(3, 3)

    elif robot_cfg.ee_type == "site":
        ee_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            robot_cfg.ee_name,
        )
        if ee_id < 0:
            raise RuntimeError(
                f"End-effector site '{robot_cfg.ee_name}' was not found."
            )
        position = data.site_xpos[ee_id].copy()
        rotation = data.site_xmat[ee_id].reshape(3, 3)

    else:
        raise ValueError(
            f"Invalid ee_type '{robot_cfg.ee_type}'."
        )

    pose = np.concatenate(
        [position, rotation[:, :2].reshape(-1)]
    )

    if pose.shape != (N_X,) or not np.all(np.isfinite(pose)):
        raise InvalidSimulation("Tip pose contains NaN or Inf.")

    return pose


def state_is_valid(model, data, max_abs_qacc):
    """Check MuJoCo arrays for non-finite values or excessive acceleration."""

    arrays = [
        data.qpos,
        data.qvel,
        data.qacc,
        data.ctrl,
    ]
    if model.na > 0:
        arrays.append(data.act)

    if any(not np.all(np.isfinite(array)) for array in arrays):
        return False

    if data.qacc.size and np.max(np.abs(data.qacc)) > max_abs_qacc:
        return False

    return True


def compute_tendon_target(
    model,
    neutral_ctrl,
    tendon_actuator_ids,
    kinematics,
    q_clark,
    clip_controls=True,
):
    """Map six Clark coordinates to the nine OpenCR tendon controls."""

    q_clark = np.asarray(q_clark, dtype=float)
    if q_clark.shape != (N_CLARK,):
        raise InvalidSimulation(
            f"Expected Clark shape {(N_CLARK,)}, got {q_clark.shape}."
        )

    tendon_delta_mm = np.asarray(
        kinematics.clark_to_tendons_mm(q_clark),
        dtype=float,
    )
    if tendon_delta_mm.shape != (N_TENDONS,):
        raise InvalidSimulation(
            "Clark-to-tendon mapping returned shape "
            f"{tendon_delta_mm.shape}, expected {(N_TENDONS,)}."
        )

    if not np.all(np.isfinite(tendon_delta_mm)):
        raise InvalidSimulation(
            "Clark-to-tendon mapping returned NaN or Inf."
        )

    # This is the generator's convention: millimetres are converted to the
    # actuator-control units by 1e-3 and added to pretension controls.
    target = (
        neutral_ctrl[tendon_actuator_ids]
        + 1e-3 * tendon_delta_mm
    )

    for local_id, actuator_id in enumerate(tendon_actuator_ids):
        if not model.actuator_ctrllimited[actuator_id]:
            continue

        lo, hi = model.actuator_ctrlrange[actuator_id]
        if target[local_id] < lo or target[local_id] > hi:
            if not clip_controls:
                raise InvalidSimulation(
                    "Generated tendon command is outside ctrlrange."
                )
            target[local_id] = np.clip(target[local_id], lo, hi)

    if not np.all(np.isfinite(target)):
        raise InvalidSimulation(
            "Generated tendon command contains NaN or Inf."
        )

    return target


def simulate_franka_tdcr_configuration(
    q,
    robot_cfg,
    model,
    data,
    panda_qpos_indices,
    panda_qvel_indices,
    tendon_actuator_ids,
    kinematics,
    neutral_qpos,
    neutral_qvel,
    neutral_act,
    neutral_ctrl,
    ramp_steps,
    settle_steps,
    max_abs_qacc,
    clip_controls=True,
):
    """Simulate one reduced configuration with the generator's procedure."""

    q = np.asarray(q, dtype=float)
    if q.shape != (N_REDUCED_Q,):
        raise InvalidSimulation(
            f"Expected q shape {(N_REDUCED_Q,)}, got {q.shape}."
        )

    q_panda = q[:N_PANDA]
    q_clark = q[N_PANDA:]

    tendon_target = compute_tendon_target(
        model=model,
        neutral_ctrl=neutral_ctrl,
        tendon_actuator_ids=tendon_actuator_ids,
        kinematics=kinematics,
        q_clark=q_clark,
        clip_controls=clip_controls,
    )

    reset_data(
        model=model,
        data=data,
        qpos=neutral_qpos,
        qvel=neutral_qvel,
        act=neutral_act,
        ctrl=neutral_ctrl,
    )

    for step_id in range(ramp_steps + settle_steps):
        if ramp_steps > 0 and step_id < ramp_steps:
            alpha = (step_id + 1) / ramp_steps
        else:
            alpha = 1.0

        data.ctrl[:] = neutral_ctrl
        data.ctrl[tendon_actuator_ids] = (
            neutral_ctrl[tendon_actuator_ids]
            + alpha
            * (
                tendon_target
                - neutral_ctrl[tendon_actuator_ids]
            )
        )

        # Match the dataset generator: Panda is held at q_panda while the
        # TDCR dynamics settle under the tendon controls.
        data.qpos[panda_qpos_indices] = q_panda
        data.qvel[panda_qvel_indices] = 0.0

        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)

        if not state_is_valid(model, data, max_abs_qacc):
            raise InvalidSimulation(
                "MuJoCo state became unstable during FK simulation."
            )

    data.qpos[panda_qpos_indices] = q_panda
    data.qvel[panda_qvel_indices] = 0.0
    mujoco.mj_forward(model, data)

    if not state_is_valid(model, data, max_abs_qacc):
        raise InvalidSimulation("Final MuJoCo state is invalid.")

    return get_tip_pose(model, data, robot_cfg)


def franka_tdcr_fk(
    qs,
    robot_cfg,
    ramp_time=0.05,
    settle_time=0.20,
    max_abs_qacc=1e8,
    clip_controls=True,
    skip_invalid=False,
    return_valid_mask=False,
):
    """Evaluate reduced Franka-TDCR FK, optionally rejecting bad samples.

    When ``skip_invalid=True``, returned invalid rows are NaN.  If
    ``return_valid_mask=True``, the function returns ``(poses, mask)``.
    """

    qs = np.asarray(qs, dtype=float)
    if qs.ndim == 1:
        qs = qs[None, :]
    if qs.ndim != 2 or qs.shape[1] != N_REDUCED_Q:
        raise ValueError(
            f"Expected qs with shape (N, {N_REDUCED_Q}), got {qs.shape}."
        )

    if getattr(robot_cfg, "q_representation", "joint") != "panda_clark":
        raise ValueError(
            "franka_tdcr must use q_representation='panda_clark'."
        )

    model = robot_cfg.robot
    data = mujoco.MjData(model)

    (
        panda_joint_ids,
        panda_qpos_indices,
        panda_qvel_indices,
        _panda_joint_names,
    ) = get_panda_joint_info(model)
    panda_low, panda_high = get_panda_joint_limits(
        model,
        panda_joint_ids,
    )
    radius = clark_radius_mm(robot_cfg)

    tendon_actuator_ids = get_tendon_actuator_ids(model)
    kinematics = get_robot_kinematics(robot_cfg)
    neutral_qpos, neutral_qvel, neutral_act, neutral_ctrl = (
        get_initial_state(model)
    )

    timestep = float(model.opt.timestep)
    ramp_steps = int(np.ceil(ramp_time / timestep))
    settle_steps = max(1, int(np.ceil(settle_time / timestep)))

    poses = np.full(
        (len(qs), N_X),
        np.nan,
        dtype=np.float64,
    )
    valid_mask = np.zeros(len(qs), dtype=bool)
    reasons = Counter()

    iterator = tqdm(
        enumerate(qs),
        total=len(qs),
        desc="Franka-TDCR FK",
        leave=False,
    )

    for sample_id, q in iterator:
        if not reduced_q_is_in_domain(
            q=q,
            panda_low=panda_low,
            panda_high=panda_high,
            clark_radius=radius,
        ):
            reasons["outside_valid_domain"] += 1
            if not skip_invalid:
                raise InvalidSimulation(
                    f"q[{sample_id}] is outside the valid Panda+Clark "
                    "configuration domain."
                )
            continue

        try:
            poses[sample_id] = simulate_franka_tdcr_configuration(
                q=q,
                robot_cfg=robot_cfg,
                model=model,
                data=data,
                panda_qpos_indices=panda_qpos_indices,
                panda_qvel_indices=panda_qvel_indices,
                tendon_actuator_ids=tendon_actuator_ids,
                kinematics=kinematics,
                neutral_qpos=neutral_qpos,
                neutral_qvel=neutral_qvel,
                neutral_act=neutral_act,
                neutral_ctrl=neutral_ctrl,
                ramp_steps=ramp_steps,
                settle_steps=settle_steps,
                max_abs_qacc=max_abs_qacc,
                clip_controls=clip_controls,
            )
            valid_mask[sample_id] = True

        except SIMULATION_ERRORS as exc:
            if not skip_invalid:
                raise

            message = str(exc).lower()
            if "ctrlrange" in message or "control" in message:
                reasons["control_range"] += 1
            elif "tip pose" in message:
                reasons["nonfinite_tip_pose"] += 1
            else:
                reasons["mujoco_unstable"] += 1

    if skip_invalid:
        rejected = int(np.sum(~valid_mask))
        print(
            f"Rejected generated samples: {rejected}/{len(qs)}"
        )
        for reason, count in sorted(reasons.items()):
            print(f"  {reason}: {count}")

    if return_valid_mask:
        return poses, valid_mask
    return poses


def forward_kinematics(robot_cfg, qs, **kwargs):
    """Compute task-space outputs with the matching backend."""

    if robot_cfg.name == "franka_tdcr":
        return franka_tdcr_fk(qs, robot_cfg, **kwargs)

    if robot_cfg.backend == "mujoco":
        return mujoco_fk(qs, robot_cfg)

    if robot_cfg.backend == "rtb":
        T = np.asarray(robot_cfg.robot.fkine(qs).A)
        return T[:, :2, 3]

    raise ValueError(f"Invalid backend: {robot_cfg.backend}")


def rotation_from_6d_columns(x6):
    """Reconstruct a rotation matrix from [R00,R01,R10,R11,R20,R21]."""

    x6 = np.asarray(x6, dtype=float)
    if x6.shape[-1] != 6:
        raise ValueError(
            f"Expected final dimension 6, got {x6.shape}."
        )

    values = x6.reshape(x6.shape[:-1] + (3, 2))
    c1 = values[..., :, 0]
    c2 = values[..., :, 1]
    eps = 1e-12

    c1 = c1 / np.maximum(
        np.linalg.norm(c1, axis=-1, keepdims=True),
        eps,
    )
    c2 = c2 - np.sum(
        c1 * c2,
        axis=-1,
        keepdims=True,
    ) * c1
    c2 = c2 / np.maximum(
        np.linalg.norm(c2, axis=-1, keepdims=True),
        eps,
    )
    c3 = np.cross(c1, c2, axis=-1)

    return np.stack([c1, c2, c3], axis=-1)


def load_dataset_target(robot_cfg, target_index):
    """Load one target and validate the saved qs/xs convention."""

    with np.load(robot_cfg.save_path) as data:
        dataset_qs = np.asarray(data["qs"], dtype=np.float64)
        dataset_xs = np.asarray(data["xs"], dtype=np.float64)

    if dataset_qs.ndim != 2 or dataset_qs.shape[1] != N_REDUCED_Q:
        raise ValueError(
            f"Expected qs shape (N, {N_REDUCED_Q}), got {dataset_qs.shape}."
        )
    if dataset_xs.ndim != 2 or dataset_xs.shape[1] != N_X:
        raise ValueError(
            f"Expected xs shape (N, {N_X}), got {dataset_xs.shape}."
        )
    if dataset_qs.shape[0] != dataset_xs.shape[0]:
        raise ValueError("qs and xs contain different numbers of samples.")
    if not 0 <= target_index < len(dataset_qs):
        raise IndexError(
            f"target_index={target_index} is outside "
            f"[0, {len(dataset_qs) - 1}]."
        )

    return (
        dataset_qs[target_index:target_index + 1],
        dataset_xs[target_index:target_index + 1],
        dataset_qs,
        dataset_xs,
    )


def evaluate(
    cfg,
    n_samples,
    n_steps,
    target_index,
    ramp_time=None,
    settle_time=None,
    max_abs_qacc=1e8,
    clip_controls=True,
):
    """Run target validation, conditional sampling, and IK metrics."""

    robot_cfg = cfg.load_robot

    if robot_cfg.name != "franka_tdcr":
        raise ValueError(
            "This evaluator is intended for robot_name=franka_tdcr."
        )
    if robot_cfg.q_dim != N_REDUCED_Q:
        raise ValueError(
            f"franka_tdcr must report q_dim={N_REDUCED_Q}; "
            f"got {robot_cfg.q_dim}."
        )
    if robot_cfg.x_dim != N_X:
        raise ValueError(
            f"franka_tdcr must report x_dim={N_X}; "
            f"got {robot_cfg.x_dim}."
        )
    if getattr(robot_cfg, "q_representation", "joint") != "panda_clark":
        raise ValueError(
            "franka_tdcr must use q_representation='panda_clark'."
        )

    if ramp_time is None:
        ramp_time = getattr(robot_cfg, "tdcr_ramp_time", 0.05)
    if settle_time is None:
        settle_time = getattr(robot_cfg, "tdcr_settle_time", 0.20)

    print(f"Robot: {robot_cfg.name}")
    print(f"q dimension: {robot_cfg.q_dim}")
    print(f"x dimension: {robot_cfg.x_dim}")

    _, _, norm = load_data(cfg)
    fm = FlowMatching(cfg, norm)
    fm.load()
    print("Flow-matching model loaded")

    q_target, x_target, dataset_qs, dataset_xs = (
        load_dataset_target(robot_cfg, target_index)
    )

    print(f"Dataset qs shape: {dataset_qs.shape}")
    print(f"Dataset xs shape: {dataset_xs.shape}")
    print(f"Target index: {target_index}")
    print(f"Target q shape: {q_target.shape}")
    print(f"Target q: {q_target[0]}")
    print(f"Target x: {x_target[0]}")

    # Strict target check. This confirms EE_pos, Clark-to-tendon mapping,
    # pretension, ramp/settle settings, and the pose convention agree.
    x_target_fk = forward_kinematics(
        robot_cfg,
        q_target,
        ramp_time=ramp_time,
        settle_time=settle_time,
        max_abs_qacc=max_abs_qacc,
        clip_controls=clip_controls,
    )

    target_difference = float(np.max(np.abs(x_target_fk - x_target)))
    print(
        "Maximum dataset/FK target difference: "
        f"{target_difference:.6e}"
    )

    if not np.allclose(
        x_target_fk,
        x_target,
        atol=1e-8,
        rtol=1e-6,
    ):
        raise ValueError(
            "Dataset and evaluator FK disagree. Check pretension, "
            "Clark parameters, EE body, ramp/settle times, and pose "
            "convention."
        )

    generated_qs = np.asarray(
        fm.sample(
            x_target,
            n_samples=n_samples,
            n_steps=n_steps,
        ),
        dtype=float,
    )

    if generated_qs.ndim != 2 or generated_qs.shape[1] != N_REDUCED_Q:
        raise ValueError(
            "Flow sampler returned the wrong shape: "
            f"{generated_qs.shape}; expected (N, {N_REDUCED_Q})."
        )

    print(f"Generated qs shape: {generated_qs.shape}")

    # Keep the original sample count for reporting. Non-finite flow outputs
    # are rejected before entering MuJoCo.
    finite_q_mask = np.all(np.isfinite(generated_qs), axis=1)
    nonfinite_count = int(np.sum(~finite_q_mask))
    if nonfinite_count:
        print(
            f"Rejected non-finite generated q values: "
            f"{nonfinite_count}/{len(generated_qs)}"
        )

    generated_xs = np.full(
        (len(generated_qs), N_X),
        np.nan,
        dtype=float,
    )
    valid_mask = np.zeros(len(generated_qs), dtype=bool)

    finite_indices = np.flatnonzero(finite_q_mask)
    if len(finite_indices):
        candidate_xs, candidate_valid = forward_kinematics(
            robot_cfg,
            generated_qs[finite_indices],
            ramp_time=ramp_time,
            settle_time=settle_time,
            max_abs_qacc=max_abs_qacc,
            clip_controls=clip_controls,
            skip_invalid=True,
            return_valid_mask=True,
        )
        generated_xs[finite_indices] = candidate_xs
        valid_mask[finite_indices] = candidate_valid

    valid_qs = generated_qs[valid_mask]
    valid_xs = generated_xs[valid_mask]

    print(
        f"Valid generated samples: {len(valid_qs)}/{len(generated_qs)} "
        f"({100.0 * len(valid_qs) / max(len(generated_qs), 1):.2f}%)"
    )

    # A poor model may produce zero physically valid samples. Report that
    # condition cleanly instead of raising another exception on bad q.
    if len(valid_qs) == 0:
        print(
            "No valid generated configurations were available for IK "
            "error metrics. The target FK check passed, so the remaining "
            "problem is the learned sampler's support or stability."
        )
        return {
            "target_q": q_target,
            "target_x": x_target,
            "generated_qs": generated_qs,
            "generated_xs": generated_xs,
            "valid_mask": valid_mask,
        }

    print(f"Generated valid qs shape: {valid_qs.shape}")
    print(f"Generated valid xs shape: {valid_xs.shape}")

    position_errors = np.linalg.norm(
        valid_xs[:, :3] - x_target[0, :3],
        axis=1,
    )

    R_target = rotation_from_6d_columns(x_target[0, 3:9])
    R_predicted = rotation_from_6d_columns(valid_xs[:, 3:9])
    R_error = np.einsum(
        "ij,njk->nik",
        R_target.T,
        R_predicted,
    )

    cos_angle = (
        np.trace(R_error, axis1=1, axis2=2) - 1.0
    ) / 2.0
    orientation_errors = np.arccos(
        np.clip(cos_angle, -1.0, 1.0)
    )

    combined_errors = 0.5 * (
        position_errors / robot_cfg.x_max
        + orientation_errors / np.pi
    )

    print(f"Position error mean: {position_errors.mean():.6f} m")
    print(f"Position error minimum: {position_errors.min():.6f} m")
    print(f"Orientation error mean: {orientation_errors.mean():.6f} rad")
    print(
        f"Orientation error minimum: {orientation_errors.min():.6f} rad"
    )
    print(f"Combined error mean: {combined_errors.mean():.6f}")
    print(f"Combined error minimum: {combined_errors.min():.6f}")

    best_idx = int(np.argmin(combined_errors))
    print("\nBest IK solution")
    print(f"Index among valid samples: {best_idx}")
    print(f"Original generated index: {np.flatnonzero(valid_mask)[best_idx]}")
    print(f"q: {valid_qs[best_idx]}")
    print(f"x: {valid_xs[best_idx]}")
    print(f"Position error: {position_errors[best_idx]:.6f} m")
    print(f"Orientation error: {orientation_errors[best_idx]:.6f} rad")
    print(f"Combined error: {combined_errors[best_idx]:.6f}")

    return {
        "target_q": q_target,
        "target_x": x_target,
        "generated_qs": generated_qs,
        "generated_xs": generated_xs,
        "valid_mask": valid_mask,
        "valid_qs": valid_qs,
        "valid_xs": valid_xs,
        "position_errors": position_errors,
        "orientation_errors": orientation_errors,
        "combined_errors": combined_errors,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate flow-matching IK for Franka-TDCR with "
            "7 Panda + 6 Clark coordinates."
        )
    )
    parser.add_argument(
        "--robot_name",
        type=str,
        default="franka_tdcr",
        help="Robot name in ROBOT_CONFIGS.",
    )
    parser.add_argument(
        "--target-index",
        type=int,
        default=0,
        help="Target row in the saved qs/xs dataset.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of generated IK candidates.",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=100,
        help="Flow-matching Euler integration steps.",
    )
    parser.add_argument(
        "--ramp-time",
        type=float,
        default=None,
        help="Tendon-control ramp time; must match dataset generation.",
    )
    parser.add_argument(
        "--settle-time",
        type=float,
        default=None,
        help="TDCR settle time; must match dataset generation.",
    )
    parser.add_argument(
        "--max-abs-qacc",
        type=float,
        default=1e8,
        help="Reject a sample when max(abs(qacc)) exceeds this value.",
    )
    parser.add_argument(
        "--no-control-clipping",
        action="store_true",
        help="Reject rather than clip tendon controls outside ctrlrange.",
    )

    args = parser.parse_args()

    if args.n_samples <= 0:
        raise ValueError("--n_samples must be positive.")
    if args.n_steps <= 0:
        raise ValueError("--n_steps must be positive.")
    if args.ramp_time is not None and args.ramp_time < 0.0:
        raise ValueError("--ramp-time cannot be negative.")
    if args.settle_time is not None and args.settle_time <= 0.0:
        raise ValueError("--settle-time must be positive.")
    if args.max_abs_qacc <= 0.0:
        raise ValueError("--max-abs-qacc must be positive.")

    cfg = FMConfig(robot_name=args.robot_name)
    evaluate(
        cfg=cfg,
        n_samples=args.n_samples,
        n_steps=args.n_steps,
        target_index=args.target_index,
        ramp_time=args.ramp_time,
        settle_time=args.settle_time,
        max_abs_qacc=args.max_abs_qacc,
        clip_controls=not args.no_control_clipping,
    )


if __name__ == "__main__":
    main()
