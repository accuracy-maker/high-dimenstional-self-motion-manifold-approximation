import argparse

import matplotlib.pyplot as plt
import numpy as np
import roboticstoolbox as rtb

from model.flow_matching import FMConfig, FlowMatching, load_data


def jacobian_position_correct(
    robot,
    q,
    target_pos,
    iters=3,
    lam=1e-3,
):
    """
    Damped least-squares Newton correction for planar position.

    Parameters
    ----------
    robot : roboticstoolbox robot
        Robot model.
    q : np.ndarray
        Joint configurations with shape (N, 3).
    target_pos : np.ndarray
        Target position with shape (2,) or (N, 2).
    iters : int
        Number of correction iterations.
    lam : float
        Damping coefficient.

    Returns
    -------
    np.ndarray
        Corrected joint configurations.
    """

    q = np.array(
        q,
        dtype=np.float64,
        copy=True,
    )

    target_pos = np.asarray(
        target_pos,
        dtype=np.float64,
    )

    if target_pos.ndim == 1:
        target_pos = np.repeat(
            target_pos[None, :],
            q.shape[0],
            axis=0,
        )

    if target_pos.shape != (q.shape[0], 2):
        raise ValueError(
            "target_pos must have shape (2,) or (N, 2), "
            f"but received {target_pos.shape}."
        )

    eye2 = np.eye(2)

    for _ in range(iters):

        for i in range(q.shape[0]):

            # Position Jacobian in the world frame
            J = np.asarray(
                robot.jacob0(q[i]),
                dtype=np.float64,
            )[:2, :]

            current_pos = np.asarray(
                robot.fkine(q[i]).A,
                dtype=np.float64,
            )[:2, 3]

            error = target_pos[i] - current_pos

            JJt = J @ J.T + (lam**2) * eye2

            q[i] += J.T @ np.linalg.solve(
                JJt,
                error,
            )

    return q


def plot_eval(
    robot,
    qs,
    x,
    errors,
    save_path="evaluation/eval_3r_fm_correct.png",
):
    """
    Plot configuration space, workspace, and FK-error histogram.
    """

    fig, (ax1, ax2, ax3) = plt.subplots(
        1,
        3,
        figsize=(12, 5),
    )

    # ------------------------------------------------------
    # Configuration space
    # ------------------------------------------------------

    ax1.scatter(
        qs[:, 1],
        qs[:, 2],
        color="blue",
        s=10,
        alpha=0.6,
    )

    ax1.set_xlabel(r"$\theta_2$ (rad)")
    ax1.set_ylabel(r"$\theta_3$ (rad)")

    ax1.set_xlim(
        -np.pi - 1,
        np.pi + 1,
    )
    ax1.set_ylim(
        -np.pi - 1,
        np.pi + 1,
    )

    ax1.set_aspect(
        "equal",
        adjustable="box",
    )

    ax1.grid(
        True,
        alpha=0.3,
    )

    # ------------------------------------------------------
    # Workspace
    # ------------------------------------------------------

    theta = np.linspace(
        0,
        2 * np.pi,
        150,
    )

    radius = 3.0

    x_boundary = radius * np.cos(theta)
    y_boundary = radius * np.sin(theta)

    ax2.plot(
        x_boundary,
        y_boundary,
        color="red",
        linewidth=2,
        label="Workspace boundary",
    )

    fk_positions = np.array(
        [
            robot.fkine(q).A[:2, 3]
            for q in qs
        ]
    )

    ax2.scatter(
        fk_positions[:, 0],
        fk_positions[:, 1],
        color="blue",
        s=10,
        alpha=0.5,
        label="Corrected samples",
    )

    ax2.scatter(
        x[0],
        x[1],
        color="black",
        marker="x",
        s=80,
        linewidths=2,
        label="Target",
    )

    ax2.set_xlabel("$x$ (m)")
    ax2.set_ylabel("$y$ (m)")

    ax2.set_xlim(
        -radius - 1,
        radius + 1,
    )
    ax2.set_ylim(
        -radius - 1,
        radius + 1,
    )

    ax2.set_aspect(
        "equal",
        adjustable="box",
    )

    ax2.grid(
        True,
        alpha=0.3,
    )

    ax2.legend()

    # ------------------------------------------------------
    # FK-error histogram
    # ------------------------------------------------------

    ax3.hist(
        errors,
        bins=20,
        color="steelblue",
        edgecolor="black",
        alpha=0.8,
    )

    ax3.set_xlabel("FK position error (m)")
    ax3.set_ylabel("Number of samples")

    ax3.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Plot saved to: {save_path}")

    plt.show()
    plt.close(fig)


def evaluate(cfg: FMConfig):

    # ------------------------------------------------------
    # Load robot
    # ------------------------------------------------------

    robot = rtb.models.DH.Planar3()

    print("Robot loaded.")

    # ------------------------------------------------------
    # Load Flow Matching model
    # ------------------------------------------------------

    _, _, norm = load_data(cfg)

    fm = FlowMatching(
        cfg,
        norm,
    )

    fm.load()

    print("Model loaded.")

    # ------------------------------------------------------
    # Target position
    # ------------------------------------------------------

    x = np.array(
        [0.0, 0.7],
        dtype=np.float64,
    )

    print(f"Target position: {x}")

    # ------------------------------------------------------
    # Generate Flow Matching samples
    # ------------------------------------------------------

    qs_raw = fm.sample(
        x,
        n_samples=1000,
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
    # Evaluate raw configurations
    # ------------------------------------------------------

    raw_fk_positions = np.array(
        [
            robot.fkine(q).A[:2, 3]
            for q in qs_raw
        ]
    )

    raw_errors = np.linalg.norm(
        raw_fk_positions - x[None, :],
        axis=1,
    )

    print(
        f"Raw error mean: "
        f"{raw_errors.mean():.6f} m"
    )

    print(
        f"Raw error minimum: "
        f"{raw_errors.min():.6f} m"
    )

    # ------------------------------------------------------
    # Apply three Jacobian correction iterations
    # ------------------------------------------------------

    qs_corrected = jacobian_position_correct(
        robot=robot,
        q=qs_raw,
        target_pos=x,
        iters=3,
        lam=1e-3,
    )

    print(
        "Applied 3 Jacobian "
        "position-correction iterations."
    )

    # ------------------------------------------------------
    # Evaluate corrected configurations
    # ------------------------------------------------------

    corrected_fk_positions = np.array(
        [
            robot.fkine(q).A[:2, 3]
            for q in qs_corrected
        ]
    )

    errors = np.linalg.norm(
        corrected_fk_positions - x[None, :],
        axis=1,
    )

    print(
        f"Corrected error mean: "
        f"{errors.mean():.6f} m"
    )

    print(
        f"Corrected error minimum: "
        f"{errors.min():.6f} m"
    )

    print(
        f"Corrected error maximum: "
        f"{errors.max():.6f} m"
    )

    total_link_length = 3.0

    print(
        "Mean corrected FK error: "
        f"{errors.mean() / total_link_length * 100:.2f} %"
    )

    print(
        "Maximum corrected FK error: "
        f"{errors.max() / total_link_length * 100:.2f} %"
    )

    # ------------------------------------------------------
    # Plot results
    # ------------------------------------------------------

    plot_eval(
        robot=robot,
        qs=qs_corrected,
        x=x,
        errors=errors,
        save_path="evaluation/eval_3r_fm_correct.png",
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Flow Matching with Jacobian "
            "position correction for Planar3."
        )
    )

    parser.add_argument(
        "--robot_name",
        type=str,
        default="3R",
        help="Robot name in ROBOT_CONFIGS.",
    )

    args = parser.parse_args()

    cfg = FMConfig(
        robot_name=args.robot_name,
    )

    evaluate(cfg)