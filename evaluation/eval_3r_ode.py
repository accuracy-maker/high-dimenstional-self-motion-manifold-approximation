import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Iterable, Sequence
from model.ode import *
import roboticstoolbox as rtb

# plot functions
def wrapped_curve_for_plot(values: np.ndarray) -> np.ndarray:
    """Wrap a curve and insert NaNs at +/-pi discontinuities."""

    wrapped = wrap_to_pi(np.asarray(values, dtype=float)).copy()
    if wrapped.ndim != 2:
        raise ValueError("values must be a two-dimensional trajectory array")
    jumps = np.any(np.abs(np.diff(wrapped, axis=0)) > np.pi, axis=1)
    if np.any(jumps):
        rows: list[np.ndarray] = []
        for index, row in enumerate(wrapped):
            rows.append(row)
            if index < jumps.size and jumps[index]:
                rows.append(np.full(wrapped.shape[1], np.nan))
        wrapped = np.asarray(rows)
    return wrapped

def plot_results(
    robot: Any,
    components: Sequence[SMMTrace],
    x: np.ndarray,
    title: str,
    output_path: Path | None,
    show: bool
) -> None:

    fig, axes = plt.subplots(2, 2, figsize=(12,5))

    joint_pairs = [(0,1), (0,2), (1,2)]

    # first 3 figures: configuration space
    for axis, (joint_a, joint_b) in zip(axes.flat[:3], joint_pairs):
        for component_index, component in enumerate(components, start=1):
            wrapped = wrapped_curve_for_plot(component.q)
            axis.plot(
                wrapped[:, joint_a],
                wrapped[:, joint_b],
                linewidth=1.6,
                label=f"component {component_index}",
            )
            axis.scatter(
                wrap_to_pi(component.q[0, joint_a]),
                wrap_to_pi(component.q[0, joint_b]),
                marker="x",
                s=45,
            )
        axis.set_xlabel(fr"$q_{joint_a + 1}$ [rad]")
        axis.set_ylabel(fr"$q_{joint_b + 1}$ [rad]")
        axis.set_xlim(-np.pi, np.pi)
        axis.set_ylim(-np.pi, np.pi)
        axis.grid(True, alpha=0.3)

    axes.flat[0].legend(loc="lower right")

    # last figure: workspace
    task_axis = axes.flat[3]
    for component_index, component in enumerate(components, start=1):
        sample_count = min(35, component.q.shape[0])
        sample_indices = np.linspace(0, component.q.shape[0] - 1, sample_count, dtype=int)
        for sample_index in sample_indices:
            points = joint_points(robot, component.q[sample_index])
            task_axis.plot(points[:, 0], points[:, 1], marker="o", markersize=2, alpha=0.16)
        first_points = joint_points(robot, component.q[0])
        task_axis.plot(
            first_points[:, 0],
            first_points[:, 1],
            marker="o",
            linewidth=2.0,
            label=f"seed {component_index}",
        )

    task_axis.scatter(x[0], x[1], marker="+", s=110, linewidths=2.0, label="target")
    task_axis.scatter(0.0, 0.0, marker="s", s=110, label="base")
    task_axis.set_xlabel("x")
    task_axis.set_ylabel("y")
    # task_axis.set_aspect("equal", adjustable="box")
    task_axis.grid(True, alpha=0.3)
    task_axis.legend(loc="lower right")

    # figure setup
    fig.suptitle(title)
    fig.tight_layout()

    # save figs
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=330, bbox_inches="tight")
        print(f"saved figure: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

@dataclass
class ODEConfig:
    RK5_step_size: float = 0.05
    minimum_steps: int = 30
    maximum_steps: int = 20_000
    singularity_tol: float = 1e-9
    closure_tol: float = 0.05

# main function
if __name__ == "__main__":

    ROOT_PATH = Path(__file__).resolve().parents[1]
    # print(f"root path: {ROOT_PATH}")

    config = ODEConfig()

    q_example_deg = {
        "outer": np.array([-35.0, 40.0, 15.0]), # out of circle with radius 1 (r > 1)
        "inner":  np.array([-170.0, 150.0, 70.0]), # r < 1
    }

    q0_deg = q_example_deg['inner']
    q0 = np.deg2rad(q0_deg)

    # 3R with length 3
    robot = build_planar_3r([1, 1, 1]) 

    # target
    x = target(robot, q0)

    seeds = rrr_component_seed(robot, q0)

    components = search_smm_components(
        robot,
        seeds,
        step_size=config.RK5_step_size,
        closure_tolerance=config.closure_tol,
        minimum_steps=config.minimum_steps,
        maximum_steps=config.maximum_steps,
        singularity_tolerance=config.singularity_tol,
    )

    print(robot)
    print(f"Initial configuration [deg]: {q0_deg}")
    print(f"Target position: [{x[0]:.9f}, {x[1]:.9f}]")
    print(f"Q* contains {len(seeds)} unique elbow-toggle seeds")
    print(f"Detected connected SMM components: {len(components)}")
    print("-" * 20)
    print()

    # each component
    for index, component in enumerate(components, start=1):
        approximate_length = config.RK5_step_size * max(0, component.q.shape[0] - 1)
        print(
            f"  component {index}: samples={component.q.shape[0]}, "
            f"closed={component.closed}, approximate length={approximate_length:.3f}, "
            f"max position error={component.max_position_error:.3e}, "
            f"min sigma={component.minimum_task_singular_value:.3e}\n"
            f"    stop: {component.stop_reason}"
        )

    # plot
    plot_results(
        robot,
        components,
        x,
        title="1-D SMM by ODE",
        output_path= Path(ROOT_PATH, "figures/ode_3r.png"),
        show=True
    )

