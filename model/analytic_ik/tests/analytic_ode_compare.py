"""Compare ODE and analytical self-motion manifolds for KUKA iiwa.

The ODE uses the MuJoCo end-effector site frame, while the analytical
solver uses the DH frame 7.  For the current model, the measured fixed
transform is

    T^7_site =
        [[-1, 0, 0, 0   ],
         [ 0,-1, 0, 0   ],
         [ 0, 0, 1, 0.045],
         [ 0, 0, 0, 1   ]].

Consequently,

    T^0_site = T^0_7 @ T^7_site,
    T^0_7    = T^0_site @ inv(T^7_site).

Analytical IK is evaluated using T^0_7.  ODE tracing remains in the
MuJoCo site coordinates.
"""

from collections import defaultdict
from dataclasses import dataclass
from itertools import product

import matplotlib.pyplot as plt
import numpy as np


# Analytical IK
from model.analytic_ik.analytic_ik_7dof import Analytical_IK_7DoF

# ODE and robot utilities
from model.ode import *  # noqa: F401,F403
from evaluation.eval_3r_ode import wrapped_curve_for_plot


# ---------------------------------------------------------------------------
# Frame conversion
# ---------------------------------------------------------------------------

T_7_SITE = np.array(
    [
        [-1.0, 0.0, 0.0, 0.000],
        [0.0, -1.0, 0.0, 0.000],
        [0.0, 0.0, 1.0, 0.045],
        [0.0, 0.0, 0.0, 1.000],
    ],
    dtype=float,
)


def site_pose_to_dh_pose(T_site):
    """Convert a MuJoCo site pose into the analytical DH frame-7 pose."""

    T_site = np.asarray(T_site, dtype=float)

    if T_site.shape != (4, 4):
        raise ValueError(
            f"Expected a pose with shape (4, 4), got {T_site.shape}."
        )

    return T_site @ np.linalg.inv(T_7_SITE)


def dh_pose_to_site_pose(T_dh):
    """Convert an analytical DH frame-7 pose into a MuJoCo site pose."""

    T_dh = np.asarray(T_dh, dtype=float)

    if T_dh.shape != (4, 4):
        raise ValueError(
            f"Expected a pose with shape (4, 4), got {T_dh.shape}."
        )

    return T_dh @ T_7_SITE


# ---------------------------------------------------------------------------
# Angle and pose utilities
# ---------------------------------------------------------------------------

def wrap_to_pi(q):
    """Wrap scalar or array angles to [-pi, pi]."""

    q = np.asarray(q, dtype=float)
    return (q + np.pi) % (2.0 * np.pi) - np.pi


def torus_delta(q1, q2):
    """Shortest componentwise angular difference."""

    return np.angle(
        np.exp(1j * (np.asarray(q1) - np.asarray(q2)))
    )


def torus_distance(q1, q2):
    """Euclidean distance using shortest angular differences."""

    return float(np.linalg.norm(torus_delta(q1, q2)))


def pose_errors(T1, T2):
    """Return Frobenius position and rotation errors."""

    position_error = np.linalg.norm(
        T1[:3, 3] - T2[:3, 3]
    )

    rotation_error = np.linalg.norm(
        T1[:3, :3] - T2[:3, :3]
    )

    return float(position_error), float(rotation_error)


# ---------------------------------------------------------------------------
# Analytical IK sampling
# ---------------------------------------------------------------------------

def get_all_ik_samples(
    ik,
    T_target,
    psi_samples,
    verbose=False,
    acos_tolerance=1e-8,
    pose_tolerance=1e-5,
    enforce_joint_limits=False,
):
    """Sample all analytical IK branches.

    Parameters
    ----------
    ik : Analytical_IK_7DoF
        Analytical solver.
    T_target : ndarray, shape (4, 4)
        Target pose expressed in the analytical DH frame-7 convention.
    psi_samples : iterable
        Samples of the arm angle.

    Returns
    -------
    list of dict
        A flat list of accepted solutions.  Each item retains its `GC` and
        `psi`, so the solutions can be grouped again without losing branch
        identity.
    """

    T_target = np.asarray(T_target, dtype=float)

    if T_target.shape != (4, 4):
        raise ValueError(
            f"Expected T_target with shape (4, 4), got {T_target.shape}."
        )

    psi_samples = np.asarray(list(psi_samples), dtype=float)

    if psi_samples.ndim != 1:
        raise ValueError("psi_samples must be one-dimensional.")

    solutions = []

    counters = defaultdict(lambda: {
        "attempted": 0,
        "accepted": 0,
        "infeasible": 0,
        "pose_rejected": 0,
        "exceptions": 0,
    })

    for gc_tuple in product([-1, 1], repeat=3):
        GC = np.asarray(gc_tuple, dtype=int)
        stats = counters[gc_tuple]

        for psi in psi_samples:
            stats["attempted"] += 1

            try:
                # The v2 solver returns four cosine arguments:
                # phi, theta_4, theta_2 and theta_6.
                unclipped = np.asarray(
                    ik.IK(
                        rigid_transform=T_target,
                        GC=GC,
                        psi=float(psi),
                        return_unclipped_vals=True,
                    ),
                    dtype=float,
                )

                if unclipped.shape != (4,):
                    raise ValueError(
                        "Expected four unclipped inverse-cosine values."
                    )

                if np.any(
                    np.abs(unclipped) > 1.0 + acos_tolerance
                ):
                    stats["infeasible"] += 1
                    continue

                q = np.asarray(
                    ik.IK(
                        rigid_transform=T_target,
                        GC=GC,
                        psi=float(psi),
                    ),
                    dtype=float,
                )

                if q.shape != (7,) or not np.all(np.isfinite(q)):
                    stats["pose_rejected"] += 1
                    continue

                if enforce_joint_limits:
                    inside_limits = (
                        np.all(q >= ik.limits_lower - 1e-8)
                        and np.all(q <= ik.limits_upper + 1e-8)
                    )

                    if not inside_limits:
                        stats["pose_rejected"] += 1
                        continue

                T_check = np.asarray(ik.FK(q), dtype=float)
                position_error, rotation_error = pose_errors(
                    T_check,
                    T_target,
                )

                if (
                    position_error > pose_tolerance
                    or rotation_error > pose_tolerance
                ):
                    stats["pose_rejected"] += 1
                    continue

                solutions.append({
                    "q": q.copy(),
                    "GC": GC.copy(),
                    "psi": float(psi),
                    "position_error": position_error,
                    "rotation_error": rotation_error,
                })
                stats["accepted"] += 1

            except Exception as error:
                stats["exceptions"] += 1
                if verbose:
                    print(
                        f"Rejected GC={GC}, psi={psi:.6f}: {error}"
                    )

    if verbose:
        for GC, stats in counters.items():
            print(f"GC={GC}: {dict(stats)}")

    return solutions


def group_analytic_solutions(ik_solutions):
    """Group flat analytical samples by their discrete GC branch."""

    branches = defaultdict(list)

    for solution in ik_solutions:
        GC = tuple(np.asarray(solution["GC"], dtype=int))
        branches[GC].append(solution)

    for GC in branches:
        branches[GC].sort(key=lambda item: item["psi"])

    return dict(branches)


def print_analytic_closure_diagnostics(ik_solutions):
    """Print branch-wise closure diagnostics on the joint torus."""

    branches = group_analytic_solutions(ik_solutions)

    print("Analytical branches:")

    for GC, branch in branches.items():
        q = np.asarray([item["q"] for item in branch])

        if len(q) < 2:
            print(f"  GC={GC}: only {len(q)} sample(s)")
            continue

        print(
            f"  GC={GC}: samples={len(q)}, "
            f"psi=[{branch[0]['psi']:.4f}, {branch[-1]['psi']:.4f}], "
            f"torus endpoint distance="
            f"{torus_distance(q[0], q[-1]):.6e}"
        )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_smm(
    ode_components,
    ik_solutions,
    save_path="model/analytic_ik/tests/figures/ode_vs_analytic.png",
):
    """Plot ODE curves and analytical samples in joint-angle coordinates.

    ODE curves are plotted after the existing branch-cut-aware wrapper.
    Analytical samples are grouped by GC before plotting; no unwrapping is
    performed across different GC branches.
    """

    if len(ik_solutions) == 0:
        raise ValueError("ik_solutions is empty.")

    analytic_branches = group_analytic_solutions(ik_solutions)

    ode_qs = []

    for component in ode_components:
        q = np.asarray(component.q, dtype=float)

        if q.ndim != 2 or q.shape[1] != 7:
            raise ValueError(
                "Each ODE component.q must have shape (N, 7)."
            )

        ode_qs.append(wrapped_curve_for_plot(q))

    fig, axes = plt.subplots(
        6,
        7,
        figsize=(18, 14),
        squeeze=False,
    )

    joint_labels = [
        r"$\theta_1$",
        r"$\theta_2$",
        r"$\theta_3$",
        r"$\theta_4$",
        r"$\theta_5$",
        r"$\theta_6$",
        r"$\theta_7$",
    ]

    for row in range(6):
        joint_y = row + 1

        for col in range(7):
            joint_x = col
            ax = axes[row, col]

            if joint_y <= joint_x:
                ax.axis("off")
                continue

            # ODE self-motion curves.
            for q_ode in ode_qs:
                ax.plot(
                    q_ode[:, joint_x],
                    q_ode[:, joint_y],
                    color="tab:orange",
                    linewidth=2.0,
                    alpha=0.75,
                )

            # Analytical samples.  Each GC branch remains separate.
            for branch in analytic_branches.values():
                q_analytic = np.asarray(
                    [item["q"] for item in branch],
                    dtype=float,
                )

                if len(q_analytic) == 0:
                    continue

                ax.scatter(
                    q_analytic[:, joint_x],
                    q_analytic[:, joint_y],
                    color="tab:blue",
                    s=8,
                    alpha=0.60,
                    edgecolors="none",
                )

            ax.set_xlim(-np.pi, np.pi)
            ax.set_ylim(-np.pi, np.pi)
            ax.grid(True, alpha=0.25)

            if row == 5:
                ax.set_xlabel(
                    joint_labels[joint_x],
                    fontsize=14,
                )
            else:
                ax.tick_params(
                    axis="x",
                    labelbottom=False,
                )

            if col == 0:
                ax.set_ylabel(
                    joint_labels[joint_y],
                    fontsize=14,
                )
            else:
                ax.tick_params(
                    axis="y",
                    labelleft=False,
                )

    axes[0, 0].plot(
        [],
        [],
        color="tab:orange",
        linewidth=2.0,
        label="ODE SMM",
    )

    axes[0, 0].scatter(
        [],
        [],
        color="tab:blue",
        s=25,
        label="Analytical IK",
    )

    fig.legend(
        loc="upper center",
        ncol=2,
        fontsize=14,
        frameon=True,
    )

    fig.subplots_adjust(
        top=0.90,
        wspace=0.08,
        hspace=0.08,
    )

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

    return fig, axes


# ---------------------------------------------------------------------------
# ODE configuration
# ---------------------------------------------------------------------------

@dataclass
class ODEConfig:
    RK5_step_size: float = 0.05
    minimum_steps: int = 30
    maximum_steps: int = 20_000
    singularity_tol: float = 1e-9
    closure_tol: float = 0.05


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    robot_cfg = get_robot_config(robot_name="kuka_iiwa_14")
    ode_cfg = ODEConfig()

    # This is only used to find an initial ODE IK solution.  The exact
    # MuJoCo pose reached by that solution is used afterward, avoiding
    # errors caused by the rounded hard-coded matrix.
    T_site_initial = np.array(
        [
            [0.73573196, 0.67552483, 0.04865491, 0.03263181],
            [-0.62280482, 0.64658636, 0.44050249, 0.39944908],
            [0.26611077, -0.35439428, 0.89643437, 0.48280907],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    print("initial site target:\n", T_site_initial)

    rng = np.random.default_rng(42)
    q_guess = random_joint_configuration(
        robot_cfg=robot_cfg,
        rng=rng,
    )

    sol = IKSolution(
        q=q_guess,
        success=False,
    )

    max_ik_attempts = 100

    for _ in range(max_ik_attempts):
        sol = solve_ik_from_seed(
            robot_cfg=robot_cfg,
            q0=sol.q,
            x=T_site_initial,
            ilimit=100,
            slimit=1,
            tol=1e-10,
            joint_limits=False,
        )

        if sol.success:
            break
    else:
        raise RuntimeError("Could not obtain an ODE IK solution.")

    q0 = np.asarray(sol.q, dtype=float)

    # Use the exact MuJoCo pose of q0 as the common target.
    T_site_target = np.asarray(
        target(robot_cfg, q0),
        dtype=float,
    )

    print("exact MuJoCo site target:\n", T_site_target)

    # ODE: generate seeds and trace all connected components.
    seeds = generate_ik_seeds(
        robot_cfg=robot_cfg,
        x=T_site_target,
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

    print(f"ODE seeds: {len(seeds)}")
    print(f"ODE components: {len(components)}")

    for index, component in enumerate(components):
        print(
            f"ODE component {index}: "
            f"samples={len(component.q)}, "
            f"closed={component.closed}, "
            f"stop='{component.stop_reason}', "
            f"torus endpoint distance="
            f"{torus_distance(component.q[0], component.q[-1]):.6e}"
        )

    # Analytical IK expects the DH frame-7 target, not the MuJoCo site
    # target.
    T_target_dh = site_pose_to_dh_pose(T_site_target)

    print("analytical DH target:\n", T_target_dh)

    ik_solver = Analytical_IK_7DoF()

    # Avoid duplicating psi=-pi and psi=+pi, which represent the same
    # point on the periodic arm-angle domain.
    psi_samples = np.linspace(
        -np.pi,
        np.pi,
        4000,
        endpoint=False,
    )

    solutions = get_all_ik_samples(
        ik_solver,
        T_target_dh,
        psi_samples,
        verbose=True,
        acos_tolerance=1e-8,
        pose_tolerance=1e-5,
        enforce_joint_limits=False,
    )

    print(f"Analytical accepted samples: {len(solutions)}")
    print_analytic_closure_diagnostics(solutions)

    # Verify the frame conversion directly at the ODE initial solution.
    T_dh_from_q0 = ik_solver.FK(q0)
    T_site_from_dh = dh_pose_to_site_pose(T_dh_from_q0)

    print(
        "frame conversion error:",
        np.max(np.abs(T_site_from_dh - T_site_target)),
    )

    plot_smm(
        ode_components=components,
        ik_solutions=solutions,
        save_path=(
            "model/analytic_ik/tests/figures/"
            "ode_vs_analytic.png"
        ),
    )