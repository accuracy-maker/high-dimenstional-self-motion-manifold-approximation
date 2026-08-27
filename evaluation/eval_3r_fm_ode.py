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

@dataclass
class ODEConfig:
    RK5_step_size: float = 0.05
    minimum_steps: int = 30
    maximum_steps: int = 20_000
    singularity_tol: float = 1e-9
    closure_tol: float = 0.05

def plot_ode_fm(ode_components, fm_qs, labels):
    fig, ax = plt.subplots(figsize=(12, 5))

    # ode
    for component_index, component in enumerate(ode_components, start=1):
        wrapped = wrapped_curve_for_plot(component.q)
        ax.plot(
            wrapped[:, 1],
            wrapped[:, 2],
            linewidth=1.6,
            label=f"component {component_index}"
        )

        ax.scatter(
            wrap_to_pi(component.q[0, 1]),
            wrap_to_pi(component.q[0, 2]),
            marker="x",
            s=110,
        )


    # fm
    for lb in np.unique(labels):
        mask = labels == lb

        ax.scatter(
            fm_qs[mask, 1],
            fm_qs[mask, 2],
            label=f"Cluster {lb}",
            alpha=0.3
        )

    ax.set_xlabel(r'$\theta_2$ (rad)')
    ax.set_ylabel(r'$\theta_3$ (rad)')

    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()

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

    q0_deg = q_example_deg['outer']
    q0 = np.deg2rad(q0_deg)

    x = target(robot = robot_cfg.robot, q = q0)
    print(f"target is: {x}")

    # ode
    ode_cfg = ODEConfig()

    seeds = rrr_component_seed(robot_cfg.robot, q0)

    components = search_smm_components(
        robot_cfg.robot,
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
        ode_components=components,
        fm_qs=qs_keep,
        labels=labels
    )

    

    
