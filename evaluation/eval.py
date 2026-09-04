"""
overall evaluation of robots:
1. robots supports: 3R, 7R (Panda, iiwa)
2. metrics:
    - ep
    - eo
    - 0.5 * (ep + eo)
    - inference speed (time)
"""

import time
import numpy as np
import argparse
# for 3R 
from spatialmath import SE3

# ODE
from model.ode import *
from evaluation.eval_3r_ode import wrapped_curve_for_plot

# flow matching
from model.flow_matching import FMConfig, FlowMatching, load_data

# smm cluster
from utils.smm_cluster import wrap_pi, filter_samples, cluster_torus

# fourier smm
from model.fourier_smm.pipeline import TASKConfig
from realtime_smm.learning import SMMNetworkBundle

# utils
from assets.data_generation import RobotConfig, get_robot_config

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


def plot_all_methods(
    ode_components,
    fm_qs,
    fourier_qs,
    save_path,
    joint_indices=(0, 1, 2),
):
    """
    Pair plot comparing ODE-traced SMM, Flow Matching samples,
    and Fourier samples.

    diagonal:
        histogram comparison

    off-diagonal:
        FM samples + ODE curves + Fourier samples
    """

    joint_indices = list(joint_indices)
    n = len(joint_indices)

    fig, axes = plt.subplots(
        n,
        n,
        figsize=(20, 15),
        sharex="col",
    )

    # wrap samples
    fm_qs_wrapped = wrap_to_pi(fm_qs)

    ode_qs = np.concatenate(
        [component.q for component in ode_components],
        axis=0,
    )
    ode_qs_wrapped = wrap_to_pi(ode_qs)

    fourier_qs_wrapped = wrap_to_pi(fourier_qs)

    # common histogram bins
    bins = np.linspace(
        -np.pi,
        np.pi,
        31,
    )

    for row in range(n):
        for col in range(n):

            ax = axes[row, col]

            joint_y = joint_indices[row]
            joint_x = joint_indices[col]

            # diagonal: histogram
            if row == col:

                ax.hist(
                    fm_qs_wrapped[:, joint_x],
                    bins=bins,
                    density=True,
                    alpha=0.45,
                    color="tab:blue",
                    label="FM samples",
                )

                ax.hist(
                    ode_qs_wrapped[:, joint_x],
                    bins=bins,
                    density=True,
                    alpha=0.45,
                    color="tab:orange",
                    label="ODE samples",
                )

                ax.hist(
                    fourier_qs_wrapped[:, joint_x],
                    bins=bins,
                    density=True,
                    alpha=0.45,
                    color="tab:green",
                    label="Fourier samples",
                )

                ax.set_xlim(
                    -np.pi,
                    np.pi,
                )

            # off-diagonal
            else:

                # FM
                ax.scatter(
                    fm_qs_wrapped[:, joint_x],
                    fm_qs_wrapped[:, joint_y],
                    s=18,
                    marker="o",
                    color="tab:blue",
                    alpha=0.35,
                    linewidths=0.5,
                    edgecolors="none",
                    label=(
                        "FM samples"
                        if row == 0 and col == 1
                        else None
                    ),
                )

                # ODE
                for component_index, component in enumerate(
                    ode_components,
                    start=1,
                ):
                    wrapped = wrapped_curve_for_plot(
                        component.q
                    )

                    ax.plot(
                        wrapped[:, joint_x],
                        wrapped[:, joint_y],
                        linewidth=3.0,
                        color="tab:orange",
                        alpha=0.5,
                        label=(
                            "ODE SMM"
                            if (
                                row == 0
                                and col == 1
                                and component_index == 1
                            )
                            else None
                        ),
                    )

                    # ODE initial point
                    # ax.scatter(
                    #     wrap_to_pi(
                    #         component.q[0, joint_x]
                    #     ),
                    #     wrap_to_pi(
                    #         component.q[0, joint_y]
                    #     ),
                    #     marker="X",
                    #     s=100,
                    #     color="tab:red",
                    #     edgecolors="black",
                    #     linewidths=0.8,
                    #     zorder=5,
                    #     label=(
                    #         "ODE initial point"
                    #         if (
                    #             row == 0
                    #             and col == 1
                    #             and component_index == 1
                    #         )
                    #         else None
                    #     ),
                    # )

                # Fourier
                # ax.scatter(
                #     fourier_qs_wrapped[:, joint_x],
                #     fourier_qs_wrapped[:, joint_y],
                #     s=18,
                #     marker="x",
                #     color="tab:green",
                #     alpha=0.25,
                #     linewidths=0.5,
                #     label=(
                #         "Fourier samples"
                #         if row == 0 and col == 1
                #         else None
                #     ),
                # )

                ax.set_xlim(
                    -np.pi,
                    np.pi,
                )

                ax.set_ylim(
                    -np.pi,
                    np.pi,
                )

            # x labels only on bottom row
            if row == n - 1:
                ax.set_xlabel(
                    rf"$\theta_{{{joint_x + 1}}}$"
                )
            else:
                ax.tick_params(
                    labelbottom=False
                )

            # y labels only on first column
            if col == 0 and row != col:
                ax.set_ylabel(
                    rf"$\theta_{{{joint_y + 1}}}$"
                )

            ax.grid(
                True,
                alpha=0.2,
            )

    axes[0, 1].legend()
    axes[0, 0].legend()

    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )




@dataclass
class ODEConfig:
    RK5_step_size: float = 0.05
    minimum_steps: int = 30
    maximum_steps: int = 20_000
    singularity_tol: float = 1e-9
    closure_tol: float = 0.05


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training flow-matching model")    
    parser.add_argument(
        "--robot_name",
        type=str,
        default="3R",
        help="robot name in ROBOT_CONFIGS",
    )

    parser.add_argument(
        "--task",
        type=str,
        help="tasks: planar, pose",
        default="planar"
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="seed of random generation",
        default=42
    )

    args = parser.parse_args()

    # load fm model
    cfg = FMConfig(robot_name=args.robot_name)
    print("FM configs are loaded")

    # robot config
    if args.robot_name == "3R":
        robot_cfg = cfg.load_robot
        print("Robot configs are loaded")

    elif args.robot_name in ("franka_emika_panda", "kuka_iiwa_14"):
        robot_cfg = get_robot_config(robot_name=args.robot_name)
        print("Robot configs are loaded")

    else:
        raise NameError("robot name is invalid")
    

    train, test, norm = load_data(cfg)
    fm = FlowMatching(cfg, norm)
    fm.load()
    print(f'model loaded')

    metrics = {
        # ep
        "ep_ode": 0.0,
        "ep_fm": 0.0,
        "ep_fourier": 0.0,
        # eo
        "eo_ode": 0.0,
        "eo_fm": 0.0,
        "eo_fourier": 0.0,

        # overall err
        "err_ode": 0.0,
        "err_fm": 0.0,
        "err_fourier": 0.0,

        # inference speed
        "inference_speed_ode": 0.0,
        "inference_speed_fm": 0.0,
        "inference_speed_fourier": 0.0,
    }


    # target
    rng = np.random.default_rng()
    # n = len(test)
    # idx = rng.integers(low=0, high=(n + 1), size=args.n_targets)

    # print(test.tensors[1][:3].numpy())

    # print(f"len of idx: {len(idx)}")
    xs = test.tensors[1].numpy()
    x = xs[100]
    # print(f"len of xs: {len(xs)}")
    # print(xs[:3])

    # inference
    if args.robot_name == "3R":
        ## ODE
        ode_cfg = ODEConfig()
        t_ode_s = time.time()
        T = SE3(x[0], x[1], 0.0)
        mask = [1, 1, 0, 0, 0, 1]
        sol = robot_cfg.robot.ikine_LM(T, mask=mask)
        q0 = sol.q
        seeds = rrr_component_seed(robot_cfg, q0)
        components = search_smm_components(
                robot_cfg,
                seeds,
                step_size=ode_cfg.RK5_step_size,
                closure_tolerance=ode_cfg.closure_tol,
                minimum_steps=ode_cfg.minimum_steps,
                maximum_steps=ode_cfg.maximum_steps,
                singularity_tolerance=ode_cfg.singularity_tol,
            )
        
        # metrics['eo_ode'] = components.mean_orientation_error

        t_ode = time.time() - t_ode_s

        ep_ode = np.mean([c.mean_position_error for c in components])
        print(f"pos err mean: {ep_ode}")
        # print(f"ori err mean: {components.mean_orientation_error}")

        metrics['ep_ode'] = ep_ode / robot_cfg.x_max
        metrics["err_ode"] = ep_ode / robot_cfg.x_max
        metrics['inference_speed_ode'] = t_ode

        # fm
        t_fm_s = time.time()
        qs = fm.sample(x, n_samples=2000, n_steps=100)
        qs_wrapped = wrap_pi(qs)
        qs_keep, keep_frac = filter_samples(
            qs = qs_wrapped,
            x = x,
            robot_cfg=robot_cfg,
            fk_tol=0.02
        )
        labels, k, gap = cluster_torus(qs_keep, eps = 1.0)

        t_fm = time.time() - t_fm_s

        T_pred = np.array(robot_cfg.robot.fkine(qs_keep).A)
        # print(f"T_pred shape: {T_pred.shape}")
        # print(f"x shape: {x.shape}")
        ep_fm = np.linalg.norm(T_pred[:, :2, 3] - x[:2], axis=-1)
        ep_fm = ep_fm.mean() / robot_cfg.x_max

        metrics['ep_fm'] = ep_fm
        metrics['err_fm'] = ep_fm
        metrics['inference_speed_fm'] = t_fm

        # fourier
        bundle = SMMNetworkBundle.load(
            path = "model/fourier_smm/results/3R_planar/3R/bundle.pt"
        )
        t_fourier_s = time.time()
        ws = bundle(T, samples = 2000)
        qs_fourier = np.concatenate(
            [
                b.angle.astype(float)
                for b in ws.data
            ],
            axis=0
        )
        t_fourier = time.time() - t_fourier_s

        print(f"ODE inference time per sample: {t_ode:.5f}")
        print(f"FM inference time per sample: {t_fm:.5f}")
        print(f"Fourier inference time per sample: {t_fourier:.5f}")

        fourier_taskcfg = TASKConfig(
            robot_name = args.robot_name,
            task = args.task
        )

        robot = fourier_taskcfg.get_robot
        T_target = np.asarray(T.A)
        ep_fourier, _ = robot.bk.fk_error_pct(
                qs_fourier,
                T_target[None,:,:]
            )

        metrics['ep_fourier'] = ep_fourier.mean()
        metrics['err_fourier'] = ep_fourier.mean()
        metrics['inference_speed_fm'] = t_fourier

        # print(f"metrics:\n {metrics}")
        print()
        print("metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.6e}")

        # plot
        fig_path = f"evaluation/{args.robot_name}_{args.task}.png"
        plot_all_methods(
            ode_components=components,
            fm_qs = qs_keep,
            fourier_qs=qs_fourier,
            save_path=fig_path,
            joint_indices=(0,1,2)
        )


    elif args.robot_name in ("franka_emika_panda", "kuka_iiwa_14"):
        T = np.eye(4)
        T[:3, 3] = x[:3]
        R = rotation_from_6d_rows(
            x[3:9]
        )
        T[:3,:3] =R

        # ode
        ode_cfg = ODEConfig()
        t_ode_s = time.time()
        q_g = random_joint_configuration(
            robot_cfg=robot_cfg,
            rng=rng
        )

        sol = IKSolution(
            q = q_g,
            success=False
        )

        while not sol.success:
            sol = solve_ik_from_seed(
                robot_cfg=robot_cfg,
                q0 = sol.q,
                x = T,
                ilimit=100,
                slimit=1,
                tol=1e-10,
                joint_limits=False
            )

        q0 = sol.q

        seeds = generate_ik_seeds(
                    robot_cfg=robot_cfg,
                    x = T,
                    q0=q0,
                )

        components = search_smm_components(
            robot_cfg,
            seeds,
            step_size=ode_cfg.RK5_step_size,
            closure_tolerance=ode_cfg.closure_tol,
            minimum_steps=ode_cfg.minimum_steps,
            maximum_steps=ode_cfg.maximum_steps,
            singularity_tolerance=ode_cfg.singularity_tol,
        )

        t_ode = time.time() - t_ode_s

        ep_ode = np.mean([c.mean_position_error for c in components])
        eo_ode = np.mean([c.mean_orientation_error for c in components])
        # print(f"pos err mean: {ep_ode}")
        # print(f"ori err mean: {components.mean_orientation_error}")

        metrics['ep_ode'] = ep_ode / robot_cfg.x_max
        metrics['eo_ode'] = eo_ode / np.pi
        metrics["err_ode"] = 0.5 * (ep_ode / robot_cfg.x_max + eo_ode / np.pi)
        metrics['inference_speed_ode'] = t_ode

        # fm
        t_fm_s = time.time()
        qs = fm.sample(x, n_samples=2000, n_steps=100)
        qs_wrapped = wrap_pi(qs)
        qs_keep, keep_frac = filter_samples(
            qs = qs_wrapped,
            x = T,
            robot_cfg=robot_cfg,
            fk_tol=0.02
        )
    
        labels, k, gap = cluster_torus(qs_keep, eps = 1.0)

        t_fm = time.time() - t_fm_s

        xs = forward_kinematics(robot_cfg, qs_keep)
        ep_fm = np.linalg.norm(
                xs[:, :3] - x[:3],
                axis=1,
            )
        
        ep_fm = ep_fm.mean() / robot_cfg.x_max

        # eo 
        
        R_pred = rotation_from_6d_rows(
            xs[:, 3:9]
        )

        R_error = R.T @ R_pred
        
        
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
    
        eo_fm = np.arccos(
            cos_angle
        )

        eo_fm = eo_fm.mean() / np.pi

        metrics['ep_fm'] = ep_fm
        metrics['eo_fm'] = eo_fm
        metrics['err_fm'] = 0.5 * (ep_fm + eo_fm)
        metrics['inference_speed_fm'] = t_fm
        

        # fourier
        ## load bundle
        bundle = SMMNetworkBundle.load(
            path = "model/fourier_smm/results/panda_pose/panda/bundle.pt"
        )
        t_fourier_s = time.time()
        ws = bundle(T, samples = 2000)
        qs_fourier = np.concatenate(
            [
                b.angle.astype(float)
                for b in ws.data
            ],
            axis=0
        )
        t_fourier = time.time() - t_fourier_s

        fourier_taskcfg = TASKConfig(
            robot_name = args.robot_name,
            task = args.task
        )

        robot = fourier_taskcfg.get_robot
        T_target = T
        ep_fourier, eo_fourier = robot.bk.fk_error_pct(
                qs_fourier,
                T_target[None,:,:]
            )

        metrics['ep_fourier'] = ep_fourier.mean()
        metrics['eo_fourier'] = eo_fourier.mean()

        metrics['err_fourier'] = 0.5 * (metrics['ep_fourier'] + metrics['eo_fourier'])
        metrics['inference_speed_fm'] = t_fourier

        print(f"ODE inference time per sample: {t_ode:.5f}")
        print(f"FM inference time per sample: {t_fm:.5f}")
        print(f"Fourier inference time per sample: {t_fourier:.5f}")

        print()
        print("metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.6e}")

        # plot
        fig_path = f"evaluation/{args.robot_name}_{args.task}.png"
        plot_all_methods(
            ode_components=components,
            fm_qs = qs_keep,
            fourier_qs=qs_fourier,
            save_path=fig_path,
            joint_indices=(0,1,2,3,4,5,6)
        )


    else:
        raise NameError("robot name is invalid")

        