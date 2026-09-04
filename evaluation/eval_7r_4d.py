"""
Evaluation of flow-matching model for 7R robot doing 3-D positioning job

"""
import argparse
import numpy as np

from model.flow_matching import FMConfig, load_data, FlowMatching

from .eval_7r_pose import  forward_kinematics

from assets.data_generation import mujoco_fk

def evaluate_4d(cfg: FMConfig):
     # load robot config + model
    robot_cfg = cfg.load_robot

    if robot_cfg.x_dim != 3:
        raise ValueError(
            f"eval_7r_4d expects a position task (x_dim=3), "
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

    xs = forward_kinematics(
        robot_cfg,
        qs,
    )

    print(f"xs shape: {xs.shape}")

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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training flow-matching model")
    parser.add_argument(
        "--robot_name",
        type=str,
        default="3R",
        help="robot name in ROBOT_CONFIGS",
    )
    args = parser.parse_args()

    cfg = FMConfig(
        robot_name=args.robot_name
    )

    evaluate_4d(cfg)