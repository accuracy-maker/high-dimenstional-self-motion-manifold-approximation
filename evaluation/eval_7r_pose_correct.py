import argparse

import numpy as np

from model.flow_matching import (
    FMConfig,
    FlowMatching,
    load_data,
)

from assets.data_generation import mujoco_fk


def forward_kinematics(
    robot_cfg,
    qs: np.ndarray,
) -> np.ndarray:
    """
    Compute task-space forward kinematics.

    Returns:
        Position-only task: (N, 3)
        Pose task: (N, 9)
    """

    qs = np.asarray(
        qs,
        dtype=np.float64,
    )

    if robot_cfg.backend == "mujoco":
        return np.asarray(
            mujoco_fk(qs, robot_cfg),
            dtype=np.float64,
        )

    if robot_cfg.backend == "rtb":
        T = np.asarray(
            robot_cfg.robot.fkine(qs).A,
            dtype=np.float64,
        )

        return T[:, :2, 3]

    raise ValueError(
        f"Invalid backend: {robot_cfg.backend}"
    )


def numerical_task_jacobian(
    robot_cfg,
    q,
    eps=1e-6,
):
    """
    Compute the full task-space Jacobian numerically.

    For a 9-D pose representation, the Jacobian has shape (9, 7).
    Each column corresponds to one of the seven joint variables.
    """

    q = np.asarray(
        q,
        dtype=np.float64,
    )

    task_dimension = robot_cfg.x_dim
    n_joints = q.shape[0]

    J = np.zeros(
        (task_dimension, n_joints),
        dtype=np.float64,
    )

    for joint_index in range(n_joints):

        q_plus = q.copy()
        q_minus = q.copy()

        q_plus[joint_index] += eps
        q_minus[joint_index] -= eps

        x_plus = forward_kinematics(
            robot_cfg,
            q_plus[None, :],
        )[0]

        x_minus = forward_kinematics(
            robot_cfg,
            q_minus[None, :],
        )[0]

        J[:, joint_index] = (
            x_plus - x_minus
        ) / (2.0 * eps)

    return J


def jacobian_task_correct_mujoco(
    robot_cfg,
    qs,
    target_pose,
    iters=3,
    lam=1e-3,
    eps=1e-6,
):
    """
    Full 7-D joint correction using the complete task pose.

    The correction minimizes the complete task-space error:

        target_pose - forward_kinematics(q)

    For x_dim=9, this includes:

        - 3-D position
        - 6-D orientation representation

    Parameters
    ----------
    robot_cfg : robot configuration
    qs : np.ndarray
        Joint configurations with shape (N, 7).
    target_pose : np.ndarray
        Target pose with shape (9,).
    iters : int
        Number of correction iterations.
    lam : float
        Damping coefficient.
    eps : float
        Finite-difference step size.

    Returns
    -------
    np.ndarray
        Corrected joint configurations with shape (N, 7).
    """

    qs = np.array(
        qs,
        dtype=np.float64,
        copy=True,
    )

    target_pose = np.asarray(
        target_pose,
        dtype=np.float64,
    )

    if target_pose.shape != (robot_cfg.x_dim,):
        raise ValueError(
            "target_pose must have shape "
            f"({robot_cfg.x_dim},), "
            f"but received {target_pose.shape}."
        )

    n_joints = qs.shape[1]
    task_dimension = robot_cfg.x_dim
    identity = np.eye(task_dimension)

    for iteration in range(iters):

        for sample_index in range(qs.shape[0]):

            q = qs[sample_index]

            current_pose = forward_kinematics(
                robot_cfg,
                q[None, :],
            )[0]

            task_error = (
                target_pose - current_pose
            )

            J = numerical_task_jacobian(
                robot_cfg,
                q,
                eps=eps,
            )

            # Damped least-squares update:
            #
            # dq = J^T (J J^T + lambda^2 I)^(-1) e
            #
            JJt = J @ J.T + (lam**2) * identity

            dq = J.T @ np.linalg.solve(
                JJt,
                task_error,
            )

            qs[sample_index] += dq

        print(
            f"Completed full-pose correction "
            f"iteration {iteration + 1}/{iters}"
        )

    return qs


def rotation_from_6d_rows(
    x6: np.ndarray,
) -> np.ndarray:
    """
    Recover a rotation matrix from the first two
    stored rows of a 6-D rotation representation.
    """

    x6 = np.asarray(
        x6,
        dtype=np.float64,
    )

    rows = x6.reshape(
        *x6.shape[:-1],
        2,
        3,
    )

    r1 = rows[..., 0, :]
    r2 = rows[..., 1, :]
    r3 = np.cross(r1, r2)

    return np.stack(
        [
            r1,
            r2,
            r3,
        ],
        axis=-2,
    )


def evaluate(
    cfg: FMConfig,
):
    """Evaluate Flow Matching with full 7-D joint correction."""

    robot_cfg = cfg.load_robot

    if robot_cfg.backend != "mujoco":
        raise ValueError(
            "This script requires the MuJoCo backend, "
            f"but got {robot_cfg.backend}."
        )

    if robot_cfg.x_dim != 9:
        raise ValueError(
            "This script expects a 9-D pose task, "
            f"but got x_dim={robot_cfg.x_dim}."
        )

    # ------------------------------------------------------
    # Load model
    # ------------------------------------------------------

    _, _, norm = load_data(cfg)

    fm = FlowMatching(
        cfg,
        norm,
    )

    fm.load()

    print("Model loaded.")

    # ------------------------------------------------------
    # Target pose
    # ------------------------------------------------------

    q_target = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=np.float64,
    )

    x = forward_kinematics(
        robot_cfg,
        q_target,
    )

    target_pose = x[0]

    print(f"Target pose: {target_pose}")
    print(f"Target pose shape: {target_pose.shape}")

    # ------------------------------------------------------
    # Generate samples
    # ------------------------------------------------------

    qs_raw = fm.sample(
        x,
        n_samples=100,
        n_steps=100,
    )

    qs_raw = np.asarray(
        qs_raw,
        dtype=np.float64,
    )

    print(
        f"Generated configurations shape: "
        f"{qs_raw.shape}"
    )

    # ------------------------------------------------------
    # Evaluate raw samples
    # ------------------------------------------------------

    xs_raw = forward_kinematics(
        robot_cfg,
        qs_raw,
    )

    raw_task_errors = np.linalg.norm(
        xs_raw - target_pose[None, :],
        axis=1,
    )

    raw_position_errors = np.linalg.norm(
        xs_raw[:, :3] - target_pose[:3],
        axis=1,
    )

    raw_orientation_errors = np.linalg.norm(
        xs_raw[:, 3:9] - target_pose[3:9],
        axis=1,
    )

    print(
        f"Raw task error mean: "
        f"{raw_task_errors.mean():.6e}"
    )

    print(
        f"Raw position error mean: "
        f"{raw_position_errors.mean():.6e} m"
    )

    print(
        f"Raw orientation representation error mean: "
        f"{raw_orientation_errors.mean():.6e}"
    )

    # ------------------------------------------------------
    # Full 7-D joint correction
    # ------------------------------------------------------

    qs_corrected = jacobian_task_correct_mujoco(
        robot_cfg=robot_cfg,
        qs=qs_raw,
        target_pose=target_pose,
        iters=3,
        lam=1e-3,
        eps=1e-6,
    )

    print(
        "Applied three full-pose Jacobian "
        "correction iterations to all seven joints."
    )

    # ------------------------------------------------------
    # Evaluate corrected samples
    # ------------------------------------------------------

    xs = forward_kinematics(
        robot_cfg,
        qs_corrected,
    )

    task_errors = np.linalg.norm(
        xs - target_pose[None, :],
        axis=1,
    )

    position_errors = np.linalg.norm(
        xs[:, :3] - target_pose[:3],
        axis=1,
    )

    orientation_representation_errors = np.linalg.norm(
        xs[:, 3:9] - target_pose[3:9],
        axis=1,
    )

    print()
    print(
        f"Corrected task error mean: "
        f"{task_errors.mean():.6e}"
    )

    print(
        f"Corrected task error minimum: "
        f"{task_errors.min():.6e}"
    )

    print(
        f"Corrected position error mean: "
        f"{position_errors.mean():.6e} m"
    )

    print(
        f"Corrected position error minimum: "
        f"{position_errors.min():.6e} m"
    )

    print(
        f"Corrected orientation representation "
        f"error mean: "
        f"{orientation_representation_errors.mean():.6e}"
    )

    # ------------------------------------------------------
    # Geometric orientation error
    # ------------------------------------------------------

    R_true = rotation_from_6d_rows(
        target_pose[3:9],
    )

    R_pred = rotation_from_6d_rows(
        xs[:, 3:9],
    )

    R_error = R_true.T @ R_pred

    cos_angle = (
        np.trace(
            R_error,
            axis1=1,
            axis2=2,
        )
        - 1.0
    ) / 2.0

    cos_angle = np.clip(
        cos_angle,
        -1.0,
        1.0,
    )

    orientation_errors = np.arccos(
        cos_angle,
    )

    print(
        f"Geometric orientation error mean: "
        f"{orientation_errors.mean():.6e} rad"
    )

    print(
        f"Geometric orientation error minimum: "
        f"{orientation_errors.min():.6e} rad"
    )

    # ------------------------------------------------------
    # Combined normalized error
    # ------------------------------------------------------

    total_length = robot_cfg.x_max

    combined_error = 0.5 * (
        position_errors / total_length
        + orientation_errors / np.pi
    )

    print(
        f"Combined error mean: "
        f"{combined_error.mean():.6e}"
    )

    print(
        f"Combined error minimum: "
        f"{combined_error.min():.6e}"
    )

    # ------------------------------------------------------
    # Best corrected sample
    # ------------------------------------------------------

    best_idx = np.argmin(combined_error)

    print("\nBest corrected sample")
    print(f"Index: {best_idx}")
    print(f"Configuration: {qs_corrected[best_idx]}")

    print(
        f"Position error: "
        f"{position_errors[best_idx]:.6e} m"
    )

    print(
        f"Orientation error: "
        f"{orientation_errors[best_idx]:.6e} rad"
    )

    print(
        f"Combined error: "
        f"{combined_error[best_idx]:.6e}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a MuJoCo 7R pose Flow Matching "
            "model with full-pose Jacobian correction."
        )
    )

    parser.add_argument(
        "--robot_name",
        type=str,
        default="kuka_iiwa_14",
        help="Robot name in ROBOT_CONFIGS.",
    )

    args = parser.parse_args()

    cfg = FMConfig(
        robot_name=args.robot_name,
    )

    evaluate(cfg)