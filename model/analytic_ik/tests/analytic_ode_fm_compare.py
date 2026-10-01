"""Compare ODE, analytical IK, and Flow Matching SMM samples.

The pair plot uses a complete 7-by-7 joint-pair matrix:

    lower triangle: ODE SMM + unlimited analytical IK samples
    upper triangle: ODE SMM + joint-limited analytical IK +
                    full-pose-corrected FM samples
    diagonal:       hidden

The ODE continuation remains unchanged and unconstrained.  The FM model is
loaded with ``FMConfig(robot_name="kuka_iiwa_14")`` and is conditioned on
the same end-effector pose used by the analytical IK and ODE computations.
FM configurations are corrected with the full-pose MuJoCo DLS correction,
including both translational and rotational errors.
"""

from collections import defaultdict
from dataclasses import dataclass
from itertools import product

import matplotlib.pyplot as plt
import numpy as np

from evaluation.eval_7r_4d import wrap_to_pi
from evaluation.eval_correct import mujoco_pose_correct
from evaluation.eval_3r_ode import wrapped_curve_for_plot
from model.analytic_ik.analytic_ik_7dof import Analytical_IK_7DoF
from model.flow_matching import FMConfig, FlowMatching, load_data
from model.ode import *  # noqa: F401,F403


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


ROBOT_NAME = "kuka_iiwa_14"
FM_SAMPLE_COUNT = 50_000
FM_INTEGRATION_STEPS = 100
FM_CORRECTION_ITERS = 3
FM_CORRECTION_LAMBDA = 1e-3
ANALYTICAL_PSI_COUNT = 4000
PLOT_FM_MAX_POINTS = 30_000


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
    """Convert a MuJoCo site pose to analytical DH frame 7."""

    T_site = np.asarray(T_site, dtype=float)
    if T_site.shape != (4, 4):
        raise ValueError(f"Expected shape (4, 4), got {T_site.shape}.")
    return T_site @ np.linalg.inv(T_7_SITE)


def dh_pose_to_site_pose(T_dh):
    """Convert analytical DH frame-7 pose to a MuJoCo site pose."""

    T_dh = np.asarray(T_dh, dtype=float)
    if T_dh.shape != (4, 4):
        raise ValueError(f"Expected shape (4, 4), got {T_dh.shape}.")
    return T_dh @ T_7_SITE


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------


def to_numpy(value):
    """Convert NumPy or torch values to a NumPy float array."""

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def torus_delta(q1, q2):
    return np.angle(
        np.exp(1j * (np.asarray(q1) - np.asarray(q2)))
    )


def torus_distance(q1, q2):
    return float(np.linalg.norm(torus_delta(q1, q2)))


def pose_errors(T1, T2):
    T1 = np.asarray(T1, dtype=float)
    T2 = np.asarray(T2, dtype=float)
    position_error = np.linalg.norm(T1[:3, 3] - T2[:3, 3])
    rotation_error = np.linalg.norm(T1[:3, :3] - T2[:3, :3])
    return float(position_error), float(rotation_error)


def fm_condition_from_pose(position, rotation, x_dim):
    """Construct the FM condition used by the first comparison script."""

    position = np.asarray(position, dtype=float)
    rotation = np.asarray(rotation, dtype=float)

    if x_dim == 3:
        return position

    if x_dim == 9:
        return np.concatenate([position, rotation[:2].reshape(6)])

    raise ValueError(
        "This comparison expects an FM conditioning dimension of 3 or 9; "
        f"received x_dim={x_dim}."
    )


def subsample_rows(qs, max_points, seed=0):
    """Deterministically reduce a large point cloud for plotting."""

    qs = np.asarray(qs, dtype=float)
    if len(qs) <= max_points:
        return qs

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(qs), size=max_points, replace=False)
    indices.sort()
    return qs[indices]


# ---------------------------------------------------------------------------
# Joint limits and analytical IK
# ---------------------------------------------------------------------------


def sample_feasible_configuration(ik, rng, interior_fraction=0.10):
    """Sample a configuration strictly inside the analytical limits."""

    lower = np.asarray(ik.limits_lower, dtype=float).reshape(-1)
    upper = np.asarray(ik.limits_upper, dtype=float).reshape(-1)

    if lower.shape != (7,) or upper.shape != (7,):
        raise ValueError("Expected seven joint limits.")
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("All joint limits must be finite.")
    if np.any(lower >= upper):
        raise ValueError("Every lower limit must be below its upper limit.")
    if not 0.0 <= interior_fraction < 0.5:
        raise ValueError("interior_fraction must be in [0, 0.5).")

    span = upper - lower
    return rng.uniform(
        lower + interior_fraction * span,
        upper - interior_fraction * span,
    )


def equivalent_configuration_inside_limits(
    q,
    lower,
    upper,
    tolerance=1e-8,
):
    """Return a 2*pi-equivalent q inside limits, or None."""

    q = np.asarray(q, dtype=float).reshape(-1)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)

    if q.shape != lower.shape or q.shape != upper.shape:
        raise ValueError("q and limits must have equal shapes.")
    if not np.all(np.isfinite(q)):
        return None

    q_inside = q.copy()
    two_pi = 2.0 * np.pi

    for index, (angle, lo, hi) in enumerate(zip(q, lower, upper)):
        k_min = int(np.ceil((lo - tolerance - angle) / two_pi))
        k_max = int(np.floor((hi + tolerance - angle) / two_pi))

        if k_min > k_max:
            return None

        if k_min <= 0 <= k_max:
            k = 0
        elif k_min > 0:
            k = k_min
        else:
            k = k_max

        candidate = angle + two_pi * k

        if candidate < lo:
            if lo - candidate > tolerance:
                return None
            candidate = lo
        elif candidate > hi:
            if candidate - hi > tolerance:
                return None
            candidate = hi

        q_inside[index] = candidate

    return q_inside


def get_all_ik_samples(
    ik,
    T_target,
    psi_samples,
    *,
    enable_joint_limits=False,
    verbose=False,
    acos_tolerance=1e-8,
    pose_tolerance=1e-5,
    joint_limit_tolerance=1e-8,
):
    """Sample all analytical GC branches for one fixed target pose."""

    T_target = np.asarray(T_target, dtype=float)
    if T_target.shape != (4, 4):
        raise ValueError(f"Expected shape (4, 4), got {T_target.shape}.")

    psi_samples = np.asarray(list(psi_samples), dtype=float)
    if psi_samples.ndim != 1:
        raise ValueError("psi_samples must be one-dimensional.")

    if enable_joint_limits:
        lower = np.asarray(ik.limits_lower, dtype=float).reshape(-1)
        upper = np.asarray(ik.limits_upper, dtype=float).reshape(-1)
        if lower.shape != (7,) or upper.shape != (7,):
            raise ValueError("Analytical limits must have shape (7,).")
    else:
        lower = upper = None

    solutions = []
    counters = defaultdict(
        lambda: {
            "attempted": 0,
            "accepted": 0,
            "infeasible": 0,
            "joint_limit_rejected": 0,
            "pose_rejected": 0,
            "exceptions": 0,
        }
    )

    for gc_tuple in product([-1, 1], repeat=3):
        GC = np.asarray(gc_tuple, dtype=int)
        stats = counters[gc_tuple]

        for psi in psi_samples:
            stats["attempted"] += 1

            try:
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

                if np.any(np.abs(unclipped) > 1.0 + acos_tolerance):
                    stats["infeasible"] += 1
                    continue

                q_raw = np.asarray(
                    ik.IK(
                        rigid_transform=T_target,
                        GC=GC,
                        psi=float(psi),
                    ),
                    dtype=float,
                )

                if q_raw.shape != (7,) or not np.all(np.isfinite(q_raw)):
                    stats["pose_rejected"] += 1
                    continue

                q = q_raw
                if enable_joint_limits:
                    q = equivalent_configuration_inside_limits(
                        q_raw,
                        lower,
                        upper,
                        tolerance=joint_limit_tolerance,
                    )
                    if q is None:
                        stats["joint_limit_rejected"] += 1
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

                solutions.append(
                    {
                        "q": q.copy(),
                        "GC": GC.copy(),
                        "psi": float(psi),
                        "position_error": position_error,
                        "rotation_error": rotation_error,
                    }
                )
                stats["accepted"] += 1

            except Exception as error:
                stats["exceptions"] += 1
                if verbose:
                    print(f"Rejected GC={GC}, psi={psi:.6f}: {error}")

    if verbose:
        for GC, stats in counters.items():
            print(f"GC={GC}: {dict(stats)}")

    return solutions


def group_analytic_solutions(solutions):
    branches = defaultdict(list)
    for solution in solutions:
        GC = tuple(np.asarray(solution["GC"], dtype=int))
        branches[GC].append(solution)

    for branch in branches.values():
        branch.sort(key=lambda item: item["psi"])

    return dict(branches)


def print_branch_comparison(solutions_unlimited, solutions_limited):
    unlimited = group_analytic_solutions(solutions_unlimited)
    limited = group_analytic_solutions(solutions_limited)
    all_gc = sorted(set(unlimited) | set(limited))

    print("Analytical branch comparison:")
    for GC in all_gc:
        n_unlimited = len(unlimited.get(GC, []))
        n_limited = len(limited.get(GC, []))
        print(
            f"  GC={GC}: unlimited={n_unlimited}, "
            f"limited={n_limited}, "
            f"removed={n_unlimited - n_limited}"
        )


# ---------------------------------------------------------------------------
# Flow Matching sampling
# ---------------------------------------------------------------------------


def sample_fm_configurations(
    fm_cfg,
    fm_robot_cfg,
    ik_solver,
    T_site_target,
    *,
    n_samples=FM_SAMPLE_COUNT,
    n_steps=FM_INTEGRATION_STEPS,
    correction_iters=FM_CORRECTION_ITERS,
    enforce_joint_limits=True,
):
    """Generate full-pose-corrected FM samples for the same target pose."""

    train, test, norm = load_data(fm_cfg)
    del train, test

    fm = FlowMatching(fm_cfg, norm)
    fm.load()

    if fm_robot_cfg.x_dim != 9:
        raise ValueError(
            "For a fixed end-effector-pose comparison, the FM model must "
            f"use x_dim=9, but robot '{ROBOT_NAME}' uses "
            f"x_dim={fm_robot_cfg.x_dim}."
        )

    target_position = np.asarray(T_site_target[:3, 3], dtype=float)
    target_rotation = np.asarray(T_site_target[:3, :3], dtype=float)
    condition = fm_condition_from_pose(
        target_position,
        target_rotation,
        fm_robot_cfg.x_dim,
    )

    raw_q = to_numpy(
        fm.sample(
            condition,
            n_samples=n_samples,
            n_steps=n_steps,
        )
    )

    if raw_q.ndim == 1:
        raw_q = raw_q[None, :]
    if raw_q.ndim != 2 or raw_q.shape[1] != 7:
        raise ValueError(f"FM returned an invalid shape: {raw_q.shape}.")

    raw_q = raw_q[np.all(np.isfinite(raw_q), axis=1)]

    if len(raw_q) == 0:
        return np.empty((0, 7), dtype=float)

    # The condition is the 9-D pose representation expected by
    # mujoco_pose_correct: position followed by the first two rows of R.
    # This corrects translation and orientation simultaneously.
    corrected_q = mujoco_pose_correct(
        robot_cfg=fm_robot_cfg,
        qs=raw_q,
        target_pose=condition,
        iters=correction_iters,
        lam=FM_CORRECTION_LAMBDA,
    )
    corrected_q = to_numpy(corrected_q)
    corrected_q = corrected_q[
        np.all(np.isfinite(corrected_q), axis=1)
    ]

    if enforce_joint_limits:
        lower = np.asarray(ik_solver.limits_lower, dtype=float)
        upper = np.asarray(ik_solver.limits_upper, dtype=float)
        mask = (
            np.all(corrected_q >= lower[None, :] - 1e-8, axis=1)
            & np.all(corrected_q <= upper[None, :] + 1e-8, axis=1)
        )
        corrected_q = corrected_q[mask]

    return corrected_q


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_ode_analytic_fm(
    ode_components,
    analytic_unlimited_solutions,
    analytic_limited_solutions,
    fm_q,
    save_path=(
        "model/analytic_ik/tests/figures/"
        "ode_analytic_fm_full_pose_analytic_upper_split_triangles.png"
    ),
):
    """Plot unlimited analytical IK below and limited IK+FM above."""

    if len(analytic_unlimited_solutions) == 0:
        raise ValueError("analytic_unlimited_solutions is empty.")
    if len(analytic_limited_solutions) == 0:
        raise ValueError("analytic_limited_solutions is empty.")
    if len(fm_q) == 0:
        raise ValueError("fm_q is empty.")

    unlimited_branches = group_analytic_solutions(
        analytic_unlimited_solutions
    )
    limited_branches = group_analytic_solutions(
        analytic_limited_solutions
    )
    fm_q = subsample_rows(fm_q, PLOT_FM_MAX_POINTS)
    fm_q = wrap_to_pi(fm_q)

    ode_qs = []
    for component in ode_components:
        q = np.asarray(component.q, dtype=float)
        if q.ndim != 2 or q.shape[1] != 7:
            raise ValueError("Each ODE component.q must have shape (N, 7).")
        ode_qs.append(wrapped_curve_for_plot(q))

    # A full 7-by-7 grid is necessary for 21 pairs in each triangle.
    fig, axes = plt.subplots(
        7,
        7,
        figsize=(18, 18),
        squeeze=False,
    )

    labels = [
        r"$\theta_1$",
        r"$\theta_2$",
        r"$\theta_3$",
        r"$\theta_4$",
        r"$\theta_5$",
        r"$\theta_6$",
        r"$\theta_7$",
    ]

    for row in range(7):
        joint_y = row

        for col in range(7):
            joint_x = col
            ax = axes[row, col]

            if joint_y == joint_x:
                ax.axis("off")
                continue

            # ODE is shown in both triangles and is always the full
            # unconstrained continuous SMM.
            for q_ode in ode_qs:
                ax.plot(
                    q_ode[:, joint_x],
                    q_ode[:, joint_y],
                    color="tab:orange",
                    linewidth=2.0,
                    alpha=0.75,
                    zorder=1,
                )

            if joint_y > joint_x:
                # Lower triangle: unlimited analytical IK.
                for branch in unlimited_branches.values():
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
                        zorder=2,
                    )
            else:
                # Upper triangle: joint-limited analytical IK and FM.
                for branch in limited_branches.values():
                    q_analytic = np.asarray(
                        [item["q"] for item in branch],
                        dtype=float,
                    )
                    if len(q_analytic) == 0:
                        continue

                    ax.scatter(
                        q_analytic[:, joint_x],
                        q_analytic[:, joint_y],
                        color="tab:red",
                        s=8,
                        alpha=0.60,
                        edgecolors="none",
                        zorder=2,
                    )

                ax.scatter(
                    fm_q[:, joint_x],
                    fm_q[:, joint_y],
                    color="tab:green",
                    s=2,
                    alpha=0.18,
                    edgecolors="none",
                    zorder=2,
                )

            ax.set_xlim(-np.pi, np.pi)
            ax.set_ylim(-np.pi, np.pi)
            ax.grid(True, alpha=0.25)

            if row == 6:
                ax.set_xlabel(labels[joint_x], fontsize=14)
            else:
                ax.tick_params(axis="x", labelbottom=False)

            if col == 0:
                ax.set_ylabel(labels[joint_y], fontsize=14)
            else:
                ax.tick_params(axis="y", labelleft=False)

    legend_ax = axes[0, 1]
    legend_ax.plot([], [], color="tab:orange", linewidth=2.0, label="ODE SMM")
    legend_ax.scatter(
        [], [], color="tab:blue", s=25, label="Unlimited analytical IK"
    )
    legend_ax.scatter(
        [], [], color="tab:red", s=25, label="Joint-limited analytical IK"
    )
    legend_ax.scatter(
        [], [], color="tab:green", s=25, label="FM samples"
    )

    fig.text(
        0.04,
        0.925,
        "Lower triangle: ODE + unlimited analytical IK",
        color="tab:blue",
        fontsize=13,
        ha="left",
        va="center",
    )
    fig.text(
        0.96,
        0.925,
        "Upper triangle: ODE + limited analytical IK + FM",
        color="tab:red",
        fontsize=13,
        ha="right",
        va="center",
    )

    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.50, 0.965),
        ncol=4,
        fontsize=13,
        frameon=True,
    )
    fig.subplots_adjust(top=0.89, wspace=0.08, hspace=0.08)

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

    return fig, axes


@dataclass
class ODEConfig:
    RK5_step_size: float = 0.05
    minimum_steps: int = 30
    maximum_steps: int = 20_000
    singularity_tol: float = 1e-9
    closure_tol: float = 0.05


if __name__ == "__main__":
    robot_cfg = get_robot_config(robot_name=ROBOT_NAME)
    fm_cfg = FMConfig(robot_name=ROBOT_NAME)
    fm_robot_cfg = fm_cfg.load_robot
    ode_cfg = ODEConfig()
    ik_solver = Analytical_IK_7DoF()

    # Use the same deterministic feasible target as the analytical/ODE
    # comparison.  The target is generated from q0, not from an unconstrained
    # IK solution of a separately specified pose.
    rng = np.random.default_rng(42)
    q0 = sample_feasible_configuration(
        ik=ik_solver,
        rng=rng,
        interior_fraction=0.10,
    )

    print("robot name:", ROBOT_NAME)
    print("feasible desired q:\n", q0)
    print("analytical lower limits:\n", ik_solver.limits_lower)
    print("analytical upper limits:\n", ik_solver.limits_upper)

    T_site_target = np.asarray(target(robot_cfg, q0), dtype=float)
    T_target_dh = site_pose_to_dh_pose(T_site_target)

    T_dh_from_q0 = np.asarray(ik_solver.FK(q0), dtype=float)
    position_error, rotation_error = pose_errors(
        T_dh_from_q0,
        T_target_dh,
    )

    print("exact MuJoCo site target:\n", T_site_target)
    print("analytical DH target:\n", T_target_dh)
    print(
        "target FK consistency: "
        f"position={position_error:.6e} m, "
        f"rotation={rotation_error:.6e}"
    )

    if position_error > 1e-5 or rotation_error > 1e-5:
        raise RuntimeError(
            "ODE and analytical FK models are inconsistent. "
            "Check T_7_SITE and the DH parameters."
        )

    # ODE remains unchanged and unconstrained.
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
            f"ODE component {index}: samples={len(component.q)}, "
            f"closed={component.closed}, stop='{component.stop_reason}', "
            f"torus endpoint distance="
            f"{torus_distance(component.q[0], component.q[-1]):.6e}"
        )

    psi_samples = np.linspace(
        -np.pi,
        np.pi,
        ANALYTICAL_PSI_COUNT,
        endpoint=False,
    )

    print("\nSampling unlimited analytical IK...")
    solutions_unlimited = get_all_ik_samples(
        ik_solver,
        T_target_dh,
        psi_samples,
        enable_joint_limits=False,
        verbose=True,
    )

    print("\nSampling joint-limited analytical IK for diagnostics...")
    solutions_limited = get_all_ik_samples(
        ik_solver,
        T_target_dh,
        psi_samples,
        enable_joint_limits=True,
        verbose=True,
    )

    print(
        "\nAnalytical samples: "
        f"unlimited={len(solutions_unlimited)}, "
        f"joint-limited={len(solutions_limited)}, "
        f"removed={len(solutions_unlimited) - len(solutions_limited)}"
    )

    print_branch_comparison(solutions_unlimited, solutions_limited)

    print("\nLoading and sampling Flow Matching...")
    fm_q = sample_fm_configurations(
        fm_cfg=fm_cfg,
        fm_robot_cfg=fm_robot_cfg,
        ik_solver=ik_solver,
        T_site_target=T_site_target,
        n_samples=FM_SAMPLE_COUNT,
        n_steps=FM_INTEGRATION_STEPS,
        correction_iters=FM_CORRECTION_ITERS,
        enforce_joint_limits=True,
    )

    print(
        "Full-pose-corrected FM samples retained: "
        f"{len(fm_q)}"
    )

    if len(solutions_unlimited) == 0:
        raise RuntimeError("No unlimited analytical samples were generated.")
    if len(solutions_limited) == 0:
        raise RuntimeError("No joint-limited analytical samples were generated.")
    if len(fm_q) == 0:
        raise RuntimeError("No FM samples were retained.")

    output_path = (
        "model/analytic_ik/tests/figures/"
        "ode_analytic_fm.png"
    )

    plot_ode_analytic_fm(
        ode_components=components,
        analytic_unlimited_solutions=solutions_unlimited,
        analytic_limited_solutions=solutions_limited,
        fm_q=fm_q,
        save_path=output_path,
    )

    print(f"Saved figure to {output_path}")