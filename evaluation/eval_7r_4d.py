"""Evaluate a flow-matching model on a 7R 3-D positioning task.

For every test target position, the script generates multiple joint
configurations, computes their forward-kinematics positions, collects the
position errors, and saves a figure containing:

    left  - histogram of FK position errors
    right - 3-D workspace scatter plot
"""

import argparse
import time
from pathlib import Path

import mujoco
import matplotlib.pyplot as plt
import numpy as np

from model.flow_matching import FMConfig, FlowMatching, load_data

from .eval_7r_pose import forward_kinematics


def wrap_to_pi(q):
    """Wrap joint angles to the interval [-pi, pi]."""
    return (np.asarray(q) + np.pi) % (2.0 * np.pi) - np.pi


def correct_positions_mujoco(
    robot_cfg,
    qs,
    target_positions,
    iters=3,
    lam=1e-2,
    max_step=0.25,
):
    """
    Apply stable MuJoCo analytical position-Jacobian correction.

    The update is applied to every configuration independently.
    A maximum joint-space step is used to prevent divergence.
    """

    model = robot_cfg.robot
    data = mujoco.MjData(model)

    qpos_ids = [
        model.jnt_qposadr[jid]
        for jid in robot_cfg._jnt_ids
    ]

    dof_ids = [
        model.jnt_dofadr[jid]
        for jid in robot_cfg._jnt_ids
    ]

    qs = np.array(
        qs,
        dtype=np.float64,
        copy=True,
    )

    target_positions = np.asarray(
        target_positions,
        dtype=np.float64,
    )

    if target_positions.ndim == 1:
        target_positions = np.repeat(
            target_positions[None, :],
            qs.shape[0],
            axis=0,
        )

    if target_positions.shape != (qs.shape[0], 3):
        raise ValueError(
            "target_positions must have shape "
            f"({qs.shape[0]}, 3), but received "
            f"{target_positions.shape}."
        )

    if robot_cfg.ee_type == "site":
        ee_id = model.site(
            robot_cfg.ee_name
        ).id
    elif robot_cfg.ee_type == "body":
        ee_id = model.body(
            robot_cfg.ee_name
        ).id
    else:
        raise ValueError(
            f"Invalid ee_type: {robot_cfg.ee_type}"
        )

    eye3 = np.eye(3)

    for iteration in range(iters):

        for sample_index in range(qs.shape[0]):

            q = qs[sample_index]

            # Set the current joint configuration
            data.qpos[:] = 0.0
            data.qpos[qpos_ids] = q

            mujoco.mj_forward(
                model,
                data,
            )

            # Get the end-effector position
            if robot_cfg.ee_type == "site":
                current_position = (
                    data.site_xpos[ee_id].copy()
                )
            else:
                current_position = (
                    data.xpos[ee_id].copy()
                )

            # Analytical MuJoCo position Jacobian
            jacp = np.zeros(
                (3, model.nv),
                dtype=np.float64,
            )

            jacr = np.zeros(
                (3, model.nv),
                dtype=np.float64,
            )

            if robot_cfg.ee_type == "site":
                mujoco.mj_jacSite(
                    model,
                    data,
                    jacp,
                    jacr,
                    ee_id,
                )
            else:
                mujoco.mj_jacBody(
                    model,
                    data,
                    jacp,
                    jacr,
                    ee_id,
                )

            J = jacp[:, dof_ids]

            position_error = (
                target_positions[sample_index]
                - current_position
            )

            # Damped least-squares update
            JJt = J @ J.T + lam**2 * eye3

            dq = J.T @ np.linalg.solve(
                JJt,
                position_error,
            )

            # Clip the joint-space update
            dq_norm = np.linalg.norm(dq)

            if dq_norm > max_step:
                dq *= max_step / dq_norm

            qs[sample_index] += dq

        print(
            f"Completed correction iteration "
            f"{iteration + 1}/{iters}",
            flush=True,
        )

    return qs


def evaluate_4d(
    cfg: FMConfig,
    n_samples: int = 2000,
    n_steps: int = 100,
    output_path: str = "evaluation/eval_7r_4d.png",
):
    """Evaluate all test target positions and save the visualization."""

    output_path = Path(output_path)
    robot_cfg = cfg.load_robot

    if robot_cfg.x_dim != 3:
        raise ValueError(
            "eval_7r_4d expects a position task with x_dim=3, "
            f"but got x_dim={robot_cfg.x_dim}."
        )

    train, test, norm = load_data(cfg)

    # The existing dataset layout stores target/task variables in tensor 1.
    # load_data returns *normalised* tensors, while FlowMatching.sample()
    # normalises its conditioning input itself and forward_kinematics()
    # works in metres, so the targets must be mapped back to raw units here.
    test_targets = test.tensors[1].detach().cpu().numpy()
    test_targets = np.asarray(test_targets, dtype=np.float64)

    x_c = np.asarray(norm["x_c"], dtype=np.float64)
    x_h = np.asarray(norm["x_h"], dtype=np.float64)
    test_targets = test_targets * x_h + x_c

    rng = np.random.default_rng(42)  # fixed seed for reproducibility
    num_targets = min(1000, len(test_targets))

    random_indices = rng.choice(
        len(test_targets),
        size=num_targets,
        replace=False,
    )

    test_targets = test_targets[random_indices]

    if test_targets.ndim == 1:
        test_targets = test_targets[None, :]

    # Keep only the 3-D position component.
    test_targets = test_targets[:, :3]

    fm = FlowMatching(
        cfg,
        norm,
    )

    fm.load()
    print("Model loaded.")
    print(f"Number of test target positions: {len(test_targets)}")

    all_target_positions = []
    all_predicted_positions = []
    all_position_errors = []
    all_best_configs = []
    skipped_targets = 0
    skipped_samples = 0

    for target_index, target_position in enumerate(test_targets):
        target_position = np.asarray(
            target_position,
            dtype=np.float64,
        )

        # print(f"target_position = {target_position}")

        # Skip invalid test targets.
        if not np.all(np.isfinite(target_position)):
            print(
                f"Skipping target {target_index}: "
                "target contains NaN or Inf."
            )
            skipped_targets += 1
            continue

        qs = fm.sample(
            target_position,
            n_samples=n_samples,
            n_steps=n_steps,
        )

        qs = np.asarray(qs, dtype=np.float64)

        # print(f"qs: {qs}")

        # Keep only complete finite joint configurations.
        finite_q_mask = np.all(
            np.isfinite(qs),
            axis=1,
        )

        invalid_q_count = np.count_nonzero(
            ~finite_q_mask
        )

        if invalid_q_count > 0:
            print(
                f"Target {target_index}: skipping "
                f"{invalid_q_count} invalid joint samples."
            )

        skipped_samples += invalid_q_count
        qs = qs[finite_q_mask]

        if len(qs) == 0:
            print(
                f"Skipping target {target_index}: "
                "all generated joint samples are invalid."
            )
            skipped_targets += 1
            continue

        predicted_positions = forward_kinematics(
            robot_cfg,
            qs=qs,
        )

        predicted_positions = np.asarray(
            predicted_positions,
            dtype=np.float64,
        )[:, :3]

        # Keep only finite FK results.
        finite_fk_mask = np.all(
            np.isfinite(predicted_positions),
            axis=1,
        )

        invalid_fk_count = np.count_nonzero(
            ~finite_fk_mask
        )

        if invalid_fk_count > 0:
            print(
                f"Target {target_index}: skipping "
                f"{invalid_fk_count} invalid FK results."
            )

        skipped_samples += invalid_fk_count
        predicted_positions = predicted_positions[
            finite_fk_mask
        ]
        qs = qs[finite_fk_mask]

        if len(predicted_positions) == 0:
            skipped_targets += 1
            continue

        target_errors = np.linalg.norm(
            predicted_positions - target_position[None, :],
            axis=1,
        )

        # Keep only finite error values.
        finite_error_mask = np.isfinite(target_errors)
        skipped_samples += np.count_nonzero(
            ~finite_error_mask
        )

        predicted_positions = predicted_positions[
            finite_error_mask
        ]
        target_errors = target_errors[
            finite_error_mask
        ]
        qs = qs[finite_error_mask]

        if len(target_errors) == 0:
            skipped_targets += 1
            continue

        # Keep only the configuration with the lowest FK error for this target.
        best_index = np.argmin(target_errors)
        best_config = qs[best_index]
        best_predicted_position = predicted_positions[best_index]
        best_position_error = target_errors[best_index]

        all_target_positions.append(target_position)
        all_predicted_positions.append(best_predicted_position)
        all_position_errors.append(best_position_error)
        all_best_configs.append(best_config)

        if (
            target_index == 0
            or (target_index + 1) % 10 == 0
            or target_index == len(test_targets) - 1
        ):
            print(
                f"Processed {target_index + 1}/"
                f"{len(test_targets)} targets"
            )

    if not all_position_errors:
        raise RuntimeError(
            "No finite evaluation results were produced."
        )

    target_positions = np.concatenate(
        all_target_positions,
        axis=0,
    ).reshape(-1, 3)

    best_configs = np.stack(
        all_best_configs,
        axis=0,
    )

    predicted_positions = np.stack(
        all_predicted_positions,
        axis=0,
    )

    position_errors = np.asarray(
        all_position_errors,
        dtype=np.float64,
    )

    # Convert the FK error from self-length units to percentage of the
    # robot's maximum self-length/workspace scale.
    position_errors = (
        100.0 * position_errors / robot_cfg.x_max
    )

    print()
    print(f"Valid target poses: {len(position_errors)}")
    print(f"Skipped targets: {skipped_targets}")
    print(f"Skipped samples: {skipped_samples}")
    print(
        f"Mean position error: "
        f"{position_errors.mean():.6f} %"
    )
    print(
        f"Median position error: "
        f"{np.median(position_errors):.6f} %"
    )
    print(
        f"Minimum position error: "
        f"{position_errors.min():.6f} %"
    )
    print(
        f"Maximum position error: "
        f"{position_errors.max():.6f} %"
    )

    plot_evaluation(
        target_positions=target_positions,
        best_configs=best_configs,
        predicted_positions=predicted_positions,
        position_errors=position_errors,
        output_path=output_path,
    )

    # Correct the selected best configuration for every target pose.
    correction_start = time.perf_counter()
    corrected_configs = correct_positions_mujoco(
        robot_cfg=robot_cfg,
        qs=best_configs,
        target_positions=target_positions,
        iters=3,
        lam=1e-1,
        max_step=0.25,
    )
    correction_time = time.perf_counter() - correction_start

    corrected_positions = np.asarray(
        forward_kinematics(robot_cfg, corrected_configs),
        dtype=np.float64,
    )[:, :3]

    corrected_errors = 100.0 * np.linalg.norm(
        corrected_positions - target_positions,
        axis=1,
    ) / robot_cfg.x_max

    corrected_output_path = (
        output_path.with_name(
            output_path.stem + "_corrected" + output_path.suffix
        )
    )

    plot_evaluation(
        target_positions=target_positions,
        best_configs=corrected_configs,
        predicted_positions=corrected_positions,
        position_errors=corrected_errors,
        output_path=corrected_output_path,
    )

    print(f"Three-iteration correction time: {correction_time:.6f} s")
    print(f"Corrected mean position error: {corrected_errors.mean():.6e} %")

    return {
        "target_positions": target_positions,
        "predicted_positions": predicted_positions,
        "position_errors": position_errors,
        "best_configs": best_configs,
        "corrected_configs": corrected_configs,
        "corrected_positions": corrected_positions,
        "corrected_position_errors": corrected_errors,
        "correction_time": correction_time,
    }


def plot_evaluation(
    target_positions: np.ndarray,
    best_configs: np.ndarray,
    predicted_positions: np.ndarray,
    position_errors: np.ndarray,
    output_path: str,
):
    """Create the FK-error histogram and workspace scatter plot."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        # Times New Roman is not installed on Linux; Nimbus Roman and
        # Liberation Serif are metric-compatible clones of it, so the figure
        # matches the paper body text instead of falling back to DejaVu.
        "font.serif": [
            "Times New Roman",
            "Nimbus Roman",
            "Liberation Serif",
            "Times",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "stix",
        # ICRA/IEEE body text is 10 pt Times; keep figure text at or just
        # above that so it stays legible at 1:1 print size.
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "black",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # IEEE two-column width.
    fig = plt.figure(figsize=(7.16, 3.5))

    # --------------------------------------------------------
    # Left: FK error histogram of the best sample per target
    # --------------------------------------------------------

    ax_error = fig.add_subplot(1, 2, 1)

    ax_error.hist(
        position_errors,
        bins=35,
        color="#0072B2",
        edgecolor="black",
        linewidth=0.4,
        alpha=0.85,
    )

    ax_error.axvline(
        position_errors.mean(),
        color="#D55E00",
        linestyle="--",
        linewidth=1.4,
        label=f"Mean: {position_errors.mean():.4f} %",
    )
    ax_error.set_xlim(0, max(position_errors.max() * 1.05, 1e-9))
    ax_error.set_xlabel("FK position error (%)")
    ax_error.set_ylabel("Number of target poses")
    ax_error.grid(
        axis="y",
        linestyle="--",
        linewidth=0.5,
        alpha=0.5,
    )

    ax_error.legend(
        frameon=True,
        facecolor="white",
        edgecolor="0.7",
        framealpha=0.9,
        borderpad=0.35,
        handlelength=1.5,
        loc="upper right",
    )

    ax_error.spines["top"].set_visible(False)
    ax_error.spines["right"].set_visible(False)

    # --------------------------------------------------------
    # Right: 3-D workspace scatter plot
    # --------------------------------------------------------

    ax_workspace = fig.add_subplot(
        1,
        2,
        2,
        projection="3d",
    )

    # Plot test target positions.
    unique_targets = np.unique(
        target_positions,
        axis=0,
    )

    # Shrink the markers as the target count grows, so the cloud stays
    # readable whether the run uses ten targets or a thousand.
    if len(unique_targets) <= 50:
        marker_size, marker_width = 28.0, 1.2
    elif len(unique_targets) <= 300:
        marker_size, marker_width = 14.0, 0.8
    else:
        marker_size, marker_width = 6.0, 0.5

    ax_workspace.scatter(
        unique_targets[:, 0],
        unique_targets[:, 1],
        unique_targets[:, 2],
        s=marker_size,
        color="#D55E00",
        marker="x",
        linewidths=marker_width,
        alpha=0.75,
        label="Target positions",
    )

    # ax_workspace.set_xlabel("$x$ (m)", labelpad=1)
    # ax_workspace.set_ylabel("$y$ (m)", labelpad=1)
    # ax_workspace.set_zlabel("$z$ (m)", labelpad=1)
    ax_workspace.legend(
        frameon=True,
        facecolor="white",
        edgecolor="0.7",
        framealpha=0.9,
        borderpad=0.35,
        markerscale=2.5,
        loc="upper left",
    )

    # Use equal axis limits so the workspace geometry is not distorted.
    all_points = np.vstack(
        [predicted_positions, unique_targets]
    )

    lower = all_points.min(axis=0)
    upper = all_points.max(axis=0)
    center = 0.5 * (lower + upper)
    radius = 0.5 * np.max(upper - lower)
    radius = max(radius, 1e-6)

    ax_workspace.set_xlim(
        center[0] - radius,
        center[0] + radius,
    )
    ax_workspace.set_ylim(
        center[1] - radius,
        center[1] + radius,
    )
    ax_workspace.set_zlim(
        center[2] - radius,
        center[2] + radius,
    )

    ax_workspace.view_init(
        elev=22,
        azim=-58,
    )

    ax_workspace.set_xticklabels([])
    ax_workspace.set_yticklabels([])
    ax_workspace.set_zticklabels([])

    ax_workspace.set_xlabel("")
    ax_workspace.set_ylabel("")
    ax_workspace.set_zlabel("")

    fig.tight_layout(pad=0.5, w_pad=0.8)

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
    )

    print(f"Saved figure to: {output_path}")
    print(f"Saved vector figure to: {pdf_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a 7R position-only flow-matching model "
            "and plot FK errors and workspace coverage."
        )
    )

    parser.add_argument(
        "--robot_name",
        type=str,
        default="3R",
        help="Robot name in ROBOT_CONFIGS.",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of joint samples per test target.",
    )

    parser.add_argument(
        "--n_steps",
        type=int,
        default=100,
        help="Number of Flow Matching integration steps.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/eval_7r_4d.png",
        help="Output figure path.",
    )

    args = parser.parse_args()

    cfg = FMConfig(
        robot_name=args.robot_name,
    )

    evaluate_4d(
        cfg,
        n_samples=args.n_samples,
        n_steps=args.n_steps,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
