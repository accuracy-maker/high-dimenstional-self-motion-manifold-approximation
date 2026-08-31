"""
Compare ODE method between Flow-matching and ODE for 7R KUKA IIWA 14 robotic arm

"""

import numpy as np
import matplotlib.pyplot as plt

# ODE
from model.ode import *

# flow matching
from model.flow_matching import FMConfig, FlowMatching, load_data

# smm cluster
from utils.smm_cluster import wrap_pi, filter_samples, cluster_torus

from assets.data_generation import RobotConfig, get_robot_config

from evaluation.eval_3r_fm_ode import plot_ode_fm

@dataclass
class ODEConfig:
    RK5_step_size: float = 0.05
    minimum_steps: int = 30
    maximum_steps: int = 20_000
    singularity_tol: float = 1e-9
    closure_tol: float = 0.05

if __name__ == "__main__":

    # load fm model
    cfg = FMConfig(robot_name="kuka_iiwa_14")
    print(f"FM configs are loaded")

    _, _, norm = load_data(cfg)
    fm = FlowMatching(cfg, norm)
    fm.load()
    print(f'model loaded')

    # target
    q0 = np.array([1,1,1, 1,1,1,1])

    robot_cfg = get_robot_config(robot_name="kuka_iiwa_14")

    T = target(robot_cfg=robot_cfg, q = q0)
    print(f"target is: {T}")

    p = T[:3, 3]
    R = T[:3, :3]

    x = np.concatenate(
        [
            p,
            R.reshape(-1)[:6]
        ]
    )

    # ode
    ode_cfg = ODEConfig()

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

    # fm
    qs = fm.sample(x, n_samples=2000, n_steps=100)
    qs_wrapped = wrap_pi(qs)
    qs_keep, keep_frac = filter_samples(
        qs = qs_wrapped,
        x = T,
        robot_cfg=robot_cfg,
        fk_tol=0.02
    )

    labels, k, gap = cluster_torus(qs_keep, eps = 1.0)

    plot_ode_fm(
        components,
        qs_keep,
        joint_indices=(0, 1, 2, 3, 4, 5, 6),
    )