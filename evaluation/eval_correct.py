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
from dataclasses import dataclass
import numpy as np
import argparse
import json
import mujoco
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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


def numerical_position_jacobian(robot_cfg, q, eps=1e-6):
    q = np.asarray(q, dtype=np.float64)
    J = np.zeros((3, q.size), dtype=np.float64)
    for j in range(q.size):
        qp, qm = q.copy(), q.copy()
        qp[j] += eps
        qm[j] -= eps
        xp = forward_kinematics(robot_cfg, qp[None, :])[0, :3]
        xm = forward_kinematics(robot_cfg, qm[None, :])[0, :3]
        J[:, j] = (xp - xm) / (2.0 * eps)
    return J


# def jacobian_position_correct(robot_cfg, qs, target_pos,
#                               iters=3, lam=1e-3, eps=1e-6):
#     """Apply three damped-least-squares position corrections."""
#     qs = np.array(qs, dtype=np.float64, copy=True)
#     target_pos = np.asarray(target_pos, dtype=np.float64)
#     eye = np.eye(2 if robot_cfg.backend == "rtb" else 3)

#     for _ in range(iters):
#         for i in range(len(qs)):
#             q = qs[i]
#             current = forward_kinematics(robot_cfg, q[None, :])[0]

#             if robot_cfg.backend == "rtb":
#                 current = current[:2]
#                 J = np.asarray(robot_cfg.robot.jacob0(q))[:2, :]
#             elif robot_cfg.backend == "mujoco":
#                 current = current[:3]
#                 J = numerical_position_jacobian(robot_cfg, q, eps)
#             else:
#                 raise ValueError(f"Unsupported backend: {robot_cfg.backend}")

#             e = target_pos - current
#             JJt = J @ J.T + (lam ** 2) * eye
#             qs[i] += J.T @ np.linalg.solve(JJt, e)
#     return qs


def jacobian_position_correct(
    robot_cfg,
    qs,
    target_pos,
    iters=3,
    lam=1e-3,
):
    """
    Batched damped-least-squares position correction for Planar3R.

    qs:
        Shape (N, 3)

    target_pos:
        Shape (2,)
    """

    qs = np.array(
        qs,
        dtype=np.float64,
        copy=True,
    )

    target_pos = np.asarray(
        target_pos,
        dtype=np.float64,
    )

    if qs.ndim != 2 or qs.shape[1] != 3:
        raise ValueError(
            f"Expected qs with shape (N, 3), got {qs.shape}."
        )

    if target_pos.shape != (2,):
        raise ValueError(
            f"Expected target_pos with shape (2,), "
            f"got {target_pos.shape}."
        )

    # Planar3R link lengths
    a = np.asarray(
        robot_cfg.robot.a,
        dtype=np.float64,
    )

    a1, a2, a3 = a[:3]

    identity = np.eye(2)

    for iteration in range(iters):

        q1 = qs[:, 0]
        q2 = qs[:, 1]
        q3 = qs[:, 2]

        q12 = q1 + q2
        q123 = q1 + q2 + q3

        s1 = np.sin(q1)
        c1 = np.cos(q1)

        s12 = np.sin(q12)
        c12 = np.cos(q12)

        s123 = np.sin(q123)
        c123 = np.cos(q123)

        # Batched planar FK
        x_current = (
            a1 * c1
            + a2 * c12
            + a3 * c123
        )

        y_current = (
            a1 * s1
            + a2 * s12
            + a3 * s123
        )

        current_position = np.stack(
            [
                x_current,
                y_current,
            ],
            axis=1,
        )

        # Batched analytical position Jacobian, shape (N, 2, 3)
        J = np.empty(
            (qs.shape[0], 2, 3),
            dtype=np.float64,
        )

        J[:, 0, 0] = (
            -a1 * s1
            -a2 * s12
            -a3 * s123
        )

        J[:, 1, 0] = (
            a1 * c1
            +a2 * c12
            +a3 * c123
        )

        J[:, 0, 1] = (
            -a2 * s12
            -a3 * s123
        )

        J[:, 1, 1] = (
            a2 * c12
            +a3 * c123
        )

        J[:, 0, 2] = -a3 * s123
        J[:, 1, 2] = a3 * c123

        # Batched position error, shape (N, 2)
        error = (
            target_pos[None, :]
            - current_position
        )

        # Batched damped least-squares solve
        JJt = J @ J.transpose(0, 2, 1)

        JJt += (lam**2) * identity[None, :, :]

        dq = (
            J.transpose(0, 2, 1)
            @ np.linalg.solve(
                JJt,
                error[:, :, None],
            )
        )[:, :, 0]

        qs += dq

        print(
            f"Completed Planar3R correction "
            f"iteration {iteration + 1}/{iters}",
            flush=True,
        )

    return qs

def _rotation_log(R):
    theta = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1e-8:
        return 0.5 * v
    return theta * v / (2.0 * np.sin(theta))


def mujoco_pose_correct(robot_cfg, qs, target_pose, iters=3, lam=1e-3):
    """Fast full-pose DLS correction using MuJoCo's analytical Jacobian."""
    model = robot_cfg.robot
    data = mujoco.MjData(model)
    qpos_ids = [model.jnt_qposadr[jid] for jid in robot_cfg._jnt_ids]
    dof_ids = [model.jnt_dofadr[jid] for jid in robot_cfg._jnt_ids]
    target_R = rotation_from_6d_rows(target_pose[3:9])
    I6 = np.eye(6)
    qs = np.array(qs, dtype=np.float64, copy=True)
    for iteration in range(iters):
        for i, q in enumerate(qs):
            data.qpos[:] = 0.0
            data.qpos[qpos_ids] = q
            mujoco.mj_forward(model, data)
            jp = np.zeros((3, model.nv))
            jr = np.zeros((3, model.nv))
            if robot_cfg.ee_type == "site":
                ee_id = model.site(robot_cfg.ee_name).id
                mujoco.mj_jacSite(model, data, jp, jr, ee_id)
                pos = data.site_xpos[ee_id].copy()
                R = data.site_xmat[ee_id].reshape(3, 3).copy()
            else:
                ee_id = model.body(robot_cfg.ee_name).id
                mujoco.mj_jacBody(model, data, jp, jr, ee_id)
                pos = data.xpos[ee_id].copy()
                R = data.xmat[ee_id].reshape(3, 3).copy()
            J = np.vstack((jp[:, dof_ids], jr[:, dof_ids]))
            e = np.concatenate((target_pose[:3] - pos, _rotation_log(target_R @ R.T)))
            qs[i] += J.T @ np.linalg.solve(J @ J.T + lam**2 * I6, e)
        print(f"Completed pose correction iteration {iteration + 1}/{iters}", flush=True)
    return qs


def forward_kinematics(robot_cfg, qs: np.ndarray) -> np.ndarray:
    """(N, x_dim) task-space FK with the same backend that generated the data"""

    qs = np.asarray(qs, dtype=np.float64)
    if qs.ndim == 1:
        qs = qs[None, :]

    if robot_cfg.backend == "mujoco":
        return np.asarray(mujoco_fk(qs, robot_cfg), dtype=np.float64)

    if robot_cfg.backend == "rtb":
        return np.asarray(
            [robot_cfg.robot.fkine(q).A[:2, 3] for q in qs],
            dtype=np.float64,
        )

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
    Pair plot comparing ODE-traced SMM with FM and Fourier samples.

    Diagonal:
        Empty

    Lower triangle:
        FM samples + ODE curves

    Upper triangle:
        Fourier samples + ODE curves
    """

    # Times for the figure text, to match the paper body font.  Times New
    # Roman is absent on Linux, so fall back to its metric-compatible clones
    # rather than silently landing on DejaVu; "stix" gives the theta labels
    # matching Times-style math.
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Nimbus Roman",
            "Liberation Serif",
            "Times",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    joint_indices = list(joint_indices)
    n = len(joint_indices)

    fig, axes = plt.subplots(
        n,
        n,
        figsize=(20, 15),
        sharex="col",
        squeeze=False,
    )

    fm_qs_wrapped = wrap_pi(fm_qs)
    fourier_qs_wrapped = wrap_pi(fourier_qs)

    for row in range(n):
        for col in range(n):

            ax = axes[row, col]

            joint_y = joint_indices[row]
            joint_x = joint_indices[col]

            # Empty diagonal
            if row == col:
                ax.axis("off")
                continue

            # ODE curves
            for component in ode_components:
                wrapped = wrapped_curve_for_plot(component.q)

                ax.plot(
                    wrapped[:, joint_x],
                    wrapped[:, joint_y],
                    color="tab:orange",
                    linewidth=3.0,
                    alpha=0.65,
                )

            # Lower triangle: FM samples
            if row > col:
                ax.scatter(
                    fm_qs_wrapped[:, joint_x],
                    fm_qs_wrapped[:, joint_y],
                    s=22,
                    marker="o",
                    color="tab:blue",
                    alpha=0.55,
                    linewidths=0.5,
                    edgecolors="none",
                )

            # Upper triangle: Fourier samples
            elif row < col:
                ax.scatter(
                    fourier_qs_wrapped[:, joint_x],
                    fourier_qs_wrapped[:, joint_y],
                    s=22,
                    marker="x",
                    color="tab:green",
                    alpha=0.55,
                    linewidths=0.5,
                )

            ax.set_xlim(-np.pi, np.pi)
            ax.set_ylim(-np.pi, np.pi)

            # X-axis labels
            if row == n - 1:
                ax.set_xlabel(
                    rf"$\theta_{{{joint_x + 1}}}$",
                    fontsize=30,
                )
            else:
                ax.tick_params(
                    axis="x",
                    labelbottom=False,
                )

            # Y-axis labels
            if col == 0:
                ax.set_ylabel(
                    rf"$\theta_{{{joint_y + 1}}}$",
                    fontsize=30,
                )
            else:
                ax.tick_params(
                    axis="y",
                    labelleft=False,
                )

            # Hide the numeric tick values; the theta axis labels carry the
            # meaning and the panels only need to show the manifold shape.
            ax.tick_params(
                axis="both",
                labelbottom=False,
                labelleft=False,
            )

            ax.grid(
                True,
                alpha=0.25,
            )

    # Clear, custom legend handles
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="tab:blue",
            markeredgecolor="none",
            markersize=16,
            alpha=0.8,
            label="FM samples",
        ),
        Line2D(
            [0],
            [0],
            color="tab:orange",
            linewidth=6.0,
            alpha=0.8,
            label="ODE SMM",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="None",
            color="tab:green",
            markersize=17,
            markeredgewidth=3.0,
            alpha=0.8,
            label="Fourier samples",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.98),
        fontsize=28,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor="black",
        borderpad=0.8,
        handlelength=2.5,
        handletextpad=0.8,
        columnspacing=2.0,
    )

    # Leave space for the legend
    fig.subplots_adjust(
        top=0.88,
        wspace=0.08,
        hspace=0.08,
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)


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
        # corrected FM metrics
        "ep_c": 0.0,
        "eo_c": 0.0,
        "err_c": 0.0,
        "correction_time": 0.0,
        "ep_fourier_c": 0.0,
        "eo_fourier_c": 0.0,
        "err_fourier_c": 0.0,
        "fourier_correction_time": 0.0,

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
        t_fm = time.time() - t_fm_s
        qs_wrapped = wrap_pi(qs)
        qs_keep, keep_frac = filter_samples(
            qs = qs_wrapped,
            x = x,
            robot_cfg=robot_cfg,
            fk_tol=0.02
        )
        labels, k, gap = cluster_torus(qs_keep, eps = 1.0)

        

        T_pred = np.array(robot_cfg.robot.fkine(qs_keep).A)
        # print(f"T_pred shape: {T_pred.shape}")
        # print(f"x shape: {x.shape}")
        ep_fm = np.linalg.norm(T_pred[:, :2, 3] - x[:2], axis=-1)
        ep_fm = ep_fm.mean() / robot_cfg.x_max

        t_c_s = time.time()
        qs_corrected = jacobian_position_correct(
            robot_cfg, qs_keep, x[:2], iters=3
        )
        t_c = time.time() - t_c_s
        T_corrected = np.asarray(
            robot_cfg.robot.fkine(qs_corrected).A
        )
        ep_c = np.linalg.norm(
            T_corrected[:, :2, 3] - x[:2], axis=-1
        ).mean() / robot_cfg.x_max

        metrics['ep_fm'] = ep_fm
        metrics['err_fm'] = ep_fm
        metrics['inference_speed_fm'] = t_fm
        metrics['ep_c'] = ep_c
        metrics['eo_c'] = 0.0
        metrics['err_c'] = ep_c
        metrics['correction_time'] = t_c

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

        T_fourier_targets = np.repeat(
            T_target[None, :, :],
            qs_fourier.shape[0],
            axis=0,
        )

        # Uncorrected Fourier metrics
        ep_fourier, _ = robot.bk.fk_error_pct(
            qs_fourier,
            T_fourier_targets,
        )

        metrics["ep_fourier"] = ep_fourier.mean()
        metrics["eo_fourier"] = 0.0
        metrics["err_fourier"] = metrics["ep_fourier"]
        metrics["inference_speed_fourier"] = t_fourier


        # ------------------------------------------------------
        # Fourier position-only correction for Planar3R
        # ------------------------------------------------------

        t_fourier_c_s = time.time()

        qs_fourier_corrected = np.array(
            qs_fourier,
            dtype=np.float64,
            copy=True,
        )

        target_position = T_target[:2, 3]

        for _ in range(3):

            T_current = robot.bk.fk(
                qs_fourier_corrected,
            )

            current_position = T_current[:, :2, 3]

            J_full = robot.bk.jacobian(
                qs_fourier_corrected,
            )

            # Planar3R uses only x/y position rows
            J_position = J_full[:, :2, :]

            position_error = (
                target_position[None, :]
                - current_position
            )

            JJt = (
                J_position
                @ J_position.transpose(0, 2, 1)
            )

            JJt[:, 0, 0] += 1e-3**2
            JJt[:, 1, 1] += 1e-3**2

            dq = (
                J_position.transpose(0, 2, 1)
                @ np.linalg.solve(
                    JJt,
                    position_error[:, :, None],
                )
            )[:, :, 0]

            qs_fourier_corrected += dq

        t_fourier_c = time.time() - t_fourier_c_s


        # Corrected Fourier position error
        T_fourier_corrected = robot.bk.fk(
            qs_fourier_corrected,
        )

        ep_fourier_c = np.linalg.norm(
            T_fourier_corrected[:, :2, 3]
            - target_position[None, :],
            axis=1,
        ) / robot.bk.L

        metrics["ep_fourier_c"] = ep_fourier_c.mean()

        # Planar3R has no orientation task
        metrics["eo_fourier_c"] = 0.0
        metrics["err_fourier_c"] = metrics["ep_fourier_c"]
        metrics["fourier_correction_time"] = t_fourier_c

        # save dict as json
        json_path = f"evaluation/{args.robot_name}_{args.task}.eval_metrics.json"
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"evaluation results saved to {json_path}")

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

        print(f"test pose:\n {T}")

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
        t_fm = time.time() - t_fm_s
        qs_wrapped = wrap_pi(qs)
        qs_keep, keep_frac = filter_samples(
            qs = qs_wrapped,
            x = T,
            robot_cfg=robot_cfg,
            fk_tol=0.02
        )
    
        labels, k, gap = cluster_torus(qs_keep, eps = 1.0)
        
        xs = forward_kinematics(robot_cfg, qs_keep)
        ep_fm = np.linalg.norm(
                xs[:, :3] - x[:3],
                axis=1,
            )
        
        ep_fm = ep_fm.mean() / robot_cfg.x_max

        t_c_s = time.time()
        qs_corrected = mujoco_pose_correct(
            robot_cfg, qs_keep, x, iters=3
        )
        t_c = time.time() - t_c_s
        xs_corrected = forward_kinematics(robot_cfg, qs_corrected)
        ep_c = np.linalg.norm(
            xs_corrected[:, :3] - x[:3], axis=1
        ).mean() / robot_cfg.x_max

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

        R_pred_c = rotation_from_6d_rows(xs_corrected[:, 3:9])
        R_error_c = R.T @ R_pred_c
        cos_angle_c = (
            np.trace(R_error_c, axis1=1, axis2=2) - 1.0
        ) / 2.0
        eo_c = np.mean(
            np.arccos(np.clip(cos_angle_c, -1.0, 1.0))
        ) / np.pi

        metrics['ep_fm'] = ep_fm
        metrics['eo_fm'] = eo_fm
        metrics['err_fm'] = 0.5 * (ep_fm + eo_fm)
        metrics['inference_speed_fm'] = t_fm
        metrics['ep_c'] = ep_c
        metrics['eo_c'] = eo_c
        metrics['err_c'] = 0.5 * (ep_c + eo_c)
        metrics['correction_time'] = t_c
        

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
        metrics["inference_speed_fourier"] = t_fourier

        T_fourier_targets = np.repeat(
            T_target[None, :, :], qs_fourier.shape[0], axis=0
        )
        t_fourier_c_s = time.time()
        qs_fourier_corrected = robot.bk.ik_correct(
            qs_fourier, T_fourier_targets, iters=3, lam=1e-3
        )
        metrics['fourier_correction_time'] = time.time() - t_fourier_c_s
        ep_fc, eo_fc = robot.bk.fk_error_pct(
            qs_fourier_corrected, T_fourier_targets
        )
        metrics['ep_fourier_c'] = ep_fc.mean()
        metrics['eo_fourier_c'] = eo_fc.mean()
        metrics['err_fourier_c'] = 0.5 * (
            metrics['ep_fourier_c'] + metrics['eo_fourier_c']
        )

        print(f"ODE inference time per sample: {t_ode:.5f}")
        print(f"FM inference time per sample: {t_fm:.5f}")
        print(f"Fourier inference time per sample: {t_fourier:.5f}")

        print()
        print("metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.6e}")

        # save dict as json
        json_path = f"evaluation/{args.robot_name}_{args.task}.eval_metrics.json"
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"evaluation results saved to {json_path}")

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

        
