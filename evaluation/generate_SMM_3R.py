"""Generate the illustrative self-motion-manifold figure for the planar 3R arm.

Two task-space positions are traced with the null-space ODE and drawn as:

    (a) task space          - workspace boundary, singularity image, targets
    (b) joint space T^3     - the self-motion manifolds on the 3-torus
    (c) projection to q2-q3 - the same manifolds seen in the q2-q3 plane

The first target lies outside the singularity image and has a single
connected self-motion manifold; the second lies inside it and has two.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from spatialmath import SE3

from assets.data_generation import get_robot_config
from model.ode import (
    SMMTrace,
    rrr_component_seed,
    search_smm_components,
    wrap_to_pi,
)

from .eval_3r_ode import wrapped_curve_for_plot


# ODE tracing settings, matching evaluation/eval_correct.py
RK5_STEP_SIZE = 0.05
MINIMUM_STEPS = 30
MAXIMUM_STEPS = 20_000
SINGULARITY_TOL = 1e-9
CLOSURE_TOL = 0.05

# Colours for the two targets.
COLOR_A = "#3B3BA5"
COLOR_B = "#CC2222"


def singularity_radii(robot_cfg) -> tuple[float, float]:
    """Radii of the task-space images of the singular configurations.

    A planar 3R arm is singular when the three links are collinear, so the
    critical values form circles of radius |+-a1 +-a2 +-a3|.
    """

    lengths = np.asarray(robot_cfg.robot.a, dtype=float)[:3]

    radii = {
        abs(float(s1 * lengths[0] + s2 * lengths[1] + s3 * lengths[2]))
        for s1 in (1.0, -1.0)
        for s2 in (1.0, -1.0)
        for s3 in (1.0, -1.0)
    }

    return max(radii), min(radii)


def trace_target(robot_cfg, position) -> list[SMMTrace]:
    """Trace every self-motion-manifold component of one planar target."""

    position = np.asarray(position, dtype=float)

    # Position-only IK: constrain x, y and the planar rotation is free.
    solution = robot_cfg.robot.ikine_LM(
        SE3(position[0], position[1], 0.0),
        mask=[1, 1, 0, 0, 0, 1],
    )

    if not solution.success:
        raise RuntimeError(
            f"IK failed for target {position}; pick a reachable position."
        )

    seeds = rrr_component_seed(robot_cfg, solution.q)

    return search_smm_components(
        robot_cfg,
        seeds,
        step_size=RK5_STEP_SIZE,
        closure_tolerance=CLOSURE_TOL,
        minimum_steps=MINIMUM_STEPS,
        maximum_steps=MAXIMUM_STEPS,
        singularity_tolerance=SINGULARITY_TOL,
    )


def apply_icra_style():
    """IEEE/ICRA figure typography: Times at body-text size."""

    plt.rcParams.update({
        "font.family": "serif",
        # Times New Roman is absent on most Linux systems; Nimbus Roman and
        # Liberation Serif are metric-compatible clones of it.
        "font.serif": [
            "Times New Roman",
            "Nimbus Roman",
            "Liberation Serif",
            "Times",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "black",
        "axes.facecolor": "white",
        "axes.grid": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def plot_task_space(ax, robot_cfg, target_a, target_b):
    """Panel (a): workspace boundary, singularity image and the two targets."""

    outer_radius, inner_radius = singularity_radii(robot_cfg)
    angle = np.linspace(0.0, 2.0 * np.pi, 400)

    ax.plot(
        outer_radius * np.cos(angle),
        outer_radius * np.sin(angle),
        color="black",
        linewidth=1.2,
    )

    ax.plot(
        inner_radius * np.cos(angle),
        inner_radius * np.sin(angle),
        color="0.55",
        linewidth=0.9,
        linestyle="--",
    )

    ax.fill(
        inner_radius * np.cos(angle),
        inner_radius * np.sin(angle),
        color="0.9",
        zorder=0,
    )

    ax.annotate(
        "workspace boundary",
        xy=(0.0, outer_radius),
        xytext=(0.0, outer_radius * 0.72),
        ha="center",
        color="0.35",
        fontsize=8,
    )

    ax.annotate(
        "singularity image",
        xy=(0.0, -inner_radius),
        xytext=(0.0, -inner_radius - 0.55),
        ha="center",
        color="0.35",
        fontsize=8,
    )

    for position, color, label in (
        (target_a, COLOR_A, "a"),
        (target_b, COLOR_B, "b"),
    ):
        ax.plot(
            position[0],
            position[1],
            marker="o",
            markersize=4.5,
            color=color,
            linestyle="None",
        )

        ax.annotate(
            f"${label}$",
            xy=position,
            xytext=(position[0] - 0.18, position[1] + 0.30),
            color=color,
        )

    limit = outer_radius * 1.15
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")

    ticks = [-outer_radius, 0.0, outer_radius]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("(a) task space")

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_joint_space(ax, components_a, components_b):
    """Panel (b): the manifolds drawn on the 3-torus."""

    for components, color in (
        (components_a, COLOR_A),
        (components_b, COLOR_B),
    ):
        for component in components:
            curve = wrapped_curve_for_plot(component.q)

            ax.plot(
                curve[:, 0],
                curve[:, 1],
                curve[:, 2],
                color=color,
                linewidth=1.3,
            )

    ticks = [-np.pi, 0.0, np.pi]
    tick_labels = [r"$-\pi$", "$0$", r"$\pi$"]

    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_zlim(-np.pi, np.pi)

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)

    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    ax.set_zticklabels(tick_labels)

    ax.set_xlabel("$q_1$", labelpad=-1)
    ax.set_ylabel("$q_2$", labelpad=-1)

    # mplot3d places the z-label at the far edge of the projected bounding
    # box, where it drifts into the neighbouring panel; place it by hand.
    ax.set_zlabel("")
    ax.text2D(
        0.97,
        0.62,
        "$q_3$",
        transform=ax.transAxes,
        ha="left",
        va="center",
    )

    ax.tick_params(pad=0.5)
    ax.view_init(elev=22, azim=-58)
    ax.set_title(r"(b) joint space $\mathbb{T}^3$")


def plot_projection(ax, components_a, components_b):
    """Panel (c): the same manifolds projected onto the q2-q3 plane."""

    for components, color in (
        (components_a, COLOR_A),
        (components_b, COLOR_B),
    ):
        for component in components:
            curve = wrapped_curve_for_plot(component.q)

            ax.plot(
                curve[:, 1],
                curve[:, 2],
                color=color,
                linewidth=1.3,
            )

    # Label each set of curves at the centroid of its longest component.
    for components, color, label in (
        (components_a, COLOR_A, "a"),
        (components_b, COLOR_B, "b"),
    ):
        if not components:
            continue

        longest = max(components, key=lambda c: len(c.q))
        centre = wrap_to_pi(longest.q[:, 1:3]).mean(axis=0)

        ax.annotate(
            f"${label}$",
            xy=centre,
            color=color,
        )

    ticks = [-np.pi, 0.0, np.pi]
    tick_labels = [r"$-\pi$", "$0$", r"$\pi$"]

    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("$q_2$")
    ax.set_ylabel("$q_3$")
    ax.set_title("(c) projection to $q_2$-$q_3$")

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def generate_figure(
    target_a=(1.80, 0.60),
    target_b=(0.55, 0.30),
    output_path="evaluation/SMM_3R.png",
):
    """Trace both targets and save the three-panel figure."""

    robot_cfg = get_robot_config("3R")

    components_a = trace_target(robot_cfg, target_a)
    components_b = trace_target(robot_cfg, target_b)

    print(f"target a = {target_a}: {len(components_a)} component(s)")
    print(f"target b = {target_b}: {len(components_b)} component(s)")

    for label, components in (("a", components_a), ("b", components_b)):
        for index, component in enumerate(components, start=1):
            print(
                f"  {label}{index}: {len(component.q)} samples | "
                f"closed={component.closed} | "
                f"max position error={component.max_position_error:.2e}"
            )

    apply_icra_style()

    # IEEE two-column width.
    fig = plt.figure(figsize=(7.16, 2.60))

    ax_task = fig.add_subplot(1, 3, 1)
    ax_joint = fig.add_subplot(1, 3, 2, projection="3d")
    ax_proj = fig.add_subplot(1, 3, 3)

    plot_task_space(ax_task, robot_cfg, target_a, target_b)
    plot_joint_space(ax_joint, components_a, components_b)
    plot_projection(ax_proj, components_a, components_b)

    fig.subplots_adjust(
        left=0.06,
        right=0.985,
        bottom=0.17,
        top=0.90,
        wspace=0.48,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=300, facecolor="white")

    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, facecolor="white")

    print(f"Saved figure to: {output_path}")
    print(f"Saved vector figure to: {pdf_path}")

    plt.close(fig)

    return components_a, components_b


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate the planar 3R self-motion-manifold illustration."
        )
    )

    parser.add_argument(
        "--target_a",
        type=float,
        nargs=2,
        default=(1.80, 0.60),
        help="Target outside the singularity image.",
    )

    parser.add_argument(
        "--target_b",
        type=float,
        nargs=2,
        default=(0.55, 0.30),
        help="Target inside the singularity image.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/SMM_3R.png",
        help="Output figure path.",
    )

    args = parser.parse_args()

    generate_figure(
        target_a=tuple(args.target_a),
        target_b=tuple(args.target_b),
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
