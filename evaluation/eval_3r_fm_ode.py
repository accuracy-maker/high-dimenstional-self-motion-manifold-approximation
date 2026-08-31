"""
Compare ODE method with Flow-matching method for 3R planar

"""

import numpy as np
import matplotlib.pyplot as plt

# ODE
from model.ode import *
from evaluation.eval_3r_ode import wrapped_curve_for_plot

# flow matching
from model.flow_matching import FMConfig, FlowMatching, load_data

# smm cluster
from utils.smm_cluster import wrap_pi, filter_samples, cluster_torus

from assets.data_generation import RobotConfig, get_robot_config

@dataclass
class ODEConfig:
    RK5_step_size: float = 0.05
    minimum_steps: int = 30
    maximum_steps: int = 20_000
    singularity_tol: float = 1e-9
    closure_tol: float = 0.05

def plot_ode_fm(
    ode_components,
    fm_qs,
    joint_indices=(0, 1, 2),
):
    """
    Pair plot comparing ODE-traced SMM and Flow Matching samples.

    diagonal:
        histogram of all FM samples vs all ODE samples

    off-diagonal:
        FM samples + ODE SMM curves
    """

    joint_indices = list(joint_indices)
    n = len(joint_indices)

    fig, axes = plt.subplots(
        n,
        n,
        figsize=(20, 15),
        sharex="col",
    )

    # wrap FM samples
    fm_qs_wrapped = wrap_to_pi(fm_qs)

    # collect all ODE samples from all components
    ode_qs = np.concatenate(
        [component.q for component in ode_components],
        axis=0,
    )
    ode_qs_wrapped = wrap_to_pi(ode_qs)

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

                # all FM samples
                ax.hist(
                    fm_qs_wrapped[:, joint_x],
                    bins=bins,
                    density=True,
                    alpha=0.4,
                    label="FM samples",
                )

                # all ODE samples
                ax.hist(
                    ode_qs_wrapped[:, joint_x],
                    bins=bins,
                    density=True,
                    alpha=0.4,
                    label="ODE samples",
                )

                ax.set_xlim(
                    -np.pi,
                    np.pi,
                )

            # off-diagonal: scatter + ODE curves
            else:

                # fm
                ax.scatter(
                    fm_qs_wrapped[:, joint_x],
                    fm_qs_wrapped[:, joint_y],
                    s=8,
                    alpha=0.25,
                    label=(
                        "FM samples"
                        if row == 0 and col == 1
                        else None
                    ),
                )

                # ode
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
                        linewidth=2.0,
                        label=(
                            f"ODE component {component_index}"
                            if row == 0 and col == 1
                            else None
                        ),
                    )

                    # initial point
                    ax.scatter(
                        wrap_to_pi(
                            component.q[0, joint_x]
                        ),
                        wrap_to_pi(
                            component.q[0, joint_y]
                        ),
                        marker="x",
                        s=50,
                    )

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
    plt.show()

if __name__ == "__main__":

    # load fm model
    cfg = FMConfig(robot_name="3R")
    robot_cfg = cfg.load_robot
    print(f"FM and Robot configs are loaded")

    _, _, norm = load_data(cfg)
    fm = FlowMatching(cfg, norm)
    fm.load()
    print(f'model loaded')

    # target
    q_example_deg = {
            "outer": np.array([-35.0, 40.0, 15.0]), # out of circle with radius 1 (r > 1)
            "inner":  np.array([-170.0, 150.0, 70.0]), # r < 1
        }

    q0_deg = q_example_deg['inner']
    q0 = np.deg2rad(q0_deg)

    robot_cfg = get_robot_config(robot_name="3R")

    x = target(robot_cfg=robot_cfg, q = q0)
    print(f"target is: {x}")

    # ode
    ode_cfg = ODEConfig()

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

    # fm
    qs = fm.sample(x, n_samples=2000, n_steps=100)
    qs_wrapped = wrap_pi(qs)
    qs_keep, keep_frac = filter_samples(
        qs = qs_wrapped,
        x = x,
        robot_cfg=robot_cfg,
        fk_tol=0.02
    )

    labels, k, gap = cluster_torus(qs_keep, eps = 1.0)

    plot_ode_fm(
        components,
        qs_keep,
        joint_indices=(0, 1, 2),
    )
    

    
