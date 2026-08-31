import numpy as np

from model.flow_matching import FMConfig, FlowMatching, load_data

from assets.data_generation import mujoco_fk


def forward_kinematics(robot_cfg, qs: np.ndarray) -> np.ndarray:
    """(N, x_dim) task-space FK with the same backend that generated the data"""

    if robot_cfg.backend == "mujoco":
        return mujoco_fk(qs, robot_cfg)

    if robot_cfg.backend == "rtb":
        T = np.array(robot_cfg.robot.fkine(qs).A)
        return T[:, :2, 3]  # planar 3R: (N, 2)

    raise ValueError(f"Invalid backend: {robot_cfg.backend}")


def rotation_from_6d_rows(x6: np.ndarray) -> np.ndarray:
    """
    Recover rotation matrix from the first two rows.

    Input:
        x6: (..., 6)

    Stored representation:
        [R00, R01, R02,
         R10, R11, R12]

    Output:
        (..., 3, 3)
    """

    x6 = np.asarray(x6, dtype=float)

    # first two rows
    rows = x6.reshape(*x6.shape[:-1], 2, 3)

    r1 = rows[..., 0, :]
    r2 = rows[..., 1, :]

    # third row
    r3 = np.cross(r1, r2)

    R = np.stack(
        [
            r1,
            r2,
            r3,
        ],
        axis=-2,
    )

    return R


def evaluate(cfg: FMConfig):

    # load robot config + model
    robot_cfg = cfg.load_robot

    if robot_cfg.x_dim != 9:
        raise ValueError(
            f"eval_7r_pose expects a pose task (x_dim=9), "
            f"got {robot_cfg.x_dim}"
        )

    _, _, norm = load_data(cfg)

    fm = FlowMatching(
        cfg,
        norm,
    )

    fm.load()

    print("model loaded")


    # target pose from FK
    q_target = np.array([
        [1, 1, 1, 1, 1, 1, 1]
    ], dtype=float)

    x = forward_kinematics(
        robot_cfg,
        qs=q_target,
    )

    print(f"target pose: {x}")
    print(f"target pose shape: {x.shape}")


    # sampling
    qs = fm.sample(
        x,
        n_samples=100,
        n_steps=100,
    )

    print(f"generated qs shape: {qs.shape}")


    # eval: FK with the same backend as the dataset
    xs = forward_kinematics(
        robot_cfg,
        qs,
    )

    print(f"xs shape: {xs.shape}")


    # position
    position_errors = np.linalg.norm(
        xs[:, :3] - x[0, :3],
        axis=1,
    )

    print(
        f"position error mean: "
        f"{position_errors.mean()} m"
    )

    print(
        f"minimum position error: "
        f"{position_errors.min()} m"
    )


    # orientation: recover the rotation matrix from the first two stored rows
    R_true = rotation_from_6d_rows(
        x[0, 3:9]
    )

    R_pred = rotation_from_6d_rows(
        xs[:, 3:9]
    )

    print(f"R_true shape: {R_true.shape}")
    print(f"R_pred shape: {R_pred.shape}")


    # relative rotation
    #
    # R_error = R_true^T @ R_pred
    #
    # shape:
    #   R_true: (3, 3)
    #   R_pred: (N, 3, 3)
    #
    # result:
    #   (N, 3, 3)
    R_error = R_true.T @ R_pred


    # rotation angle
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

    ori_errors = np.arccos(
        cos_angle
    )


    print(
        f"orientation error mean: "
        f"{ori_errors.mean()} rad"
    )

    print(
        f"minimum orientation error: "
        f"{ori_errors.min()} rad"
    )


    # combined error
    total_length = robot_cfg.x_max

    error = 0.5 * (
        position_errors / total_length
        +
        ori_errors / np.pi
    )


    print(
        f"mean error: "
        f"{error.mean()}"
    )

    print(
        f"min error: "
        f"{error.min()}"
    )


    # best sample
    best_idx = np.argmin(error)

    print("\nBest sample")
    print(f"index: {best_idx}")
    print(f"q: {qs[best_idx]}")
    print(
        f"position error: "
        f"{position_errors[best_idx]} m"
    )
    print(
        f"orientation error: "
        f"{ori_errors[best_idx]} rad"
    )
    print(
        f"combined error: "
        f"{error[best_idx]}"
    )


if __name__ == "__main__":

    cfg = FMConfig(
        robot_name="kuka_iiwa_14"
    )

    evaluate(cfg)