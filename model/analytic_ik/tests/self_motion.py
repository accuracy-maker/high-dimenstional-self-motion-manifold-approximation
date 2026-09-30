"""
test for compute all IK solution given target pose
All ik solutions lie on the 1-D self motion manifold (1-D curve)
"""

from model.analytic_ik.analytic_ik_7dof import Analytical_IK_7DoF
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

# helper functions
def get_all_ik_samples(ik, T_target, psi_samples, verbose=False):
    """
    Compute sampled IK solutions using only NumPy arrays.

    Parameters
    ----------
    T_target : np.ndarray
        Target homogeneous transform with shape (4, 4).

    psi_samples : iterable
        Samples of the self-motion parameter.
    """

    T_target = np.asarray(T_target, dtype=float)

    if T_target.shape != (4, 4):
        raise ValueError(
            f"Expected T_target with shape (4, 4), "
            f"got {T_target.shape}"
        )

    solutions = []

    for gc_tuple in product([-1, 1], repeat=3):
        GC = np.array(gc_tuple, dtype=int)

        for psi in psi_samples:
            try:
                q = np.asarray(
                    ik.IK(
                        rigid_transform=T_target,
                        GC=GC,
                        psi=psi,
                    ),
                    dtype=float,
                )

                if q.shape != (7,):
                    continue

                # Reject joint-limit violations.
                # inside_limits = np.all(
                #     q >= ik.limits_lower - 1e-8
                # ) and np.all(
                #     q <= ik.limits_upper + 1e-8
                # )

                # if not inside_limits:
                #     continue

                # Forward kinematics of recovered q.
                T_check = np.asarray(
                    ik.FK(q),
                    dtype=float,
                )

                position_error = np.linalg.norm(
                    T_check[:3, 3]
                    - T_target[:3, 3]
                )

                rotation_error = np.linalg.norm(
                    T_check[:3, :3]
                    - T_target[:3, :3]
                )

                if (
                    position_error < 1e-6
                    and rotation_error < 1e-6
                ):
                    solutions.append({
                        "q": q.copy(),
                        "GC": GC.copy(),
                        "psi": float(psi),
                        "position_error": position_error,
                        "rotation_error": rotation_error,
                    })

            except Exception as error:
                if verbose:
                    print(
                        f"Rejected GC={GC}, "
                        f"psi={psi:.4f}: {error}"
                    )

    return solutions

def plot_smm(ik_solutions):
    """
    Plot pairwise projections of the sampled self-motion manifold.

    Rows:
        theta_2, ..., theta_7

    Columns:
        theta_1, ..., theta_7

    Only the lower-triangular subplots are displayed.
    """

    if len(ik_solutions) == 0:
        raise ValueError("ik_solutions is empty.")

    # Extract joint configurations.
    q_values = np.asarray([
        solution["q"] if isinstance(solution, dict) else solution
        for solution in ik_solutions
    ])

    if q_values.ndim != 2 or q_values.shape[1] != 7:
        raise ValueError(
            "Expected q values with shape (number_of_solutions, 7)."
        )

    # Extract psi values if available.
    psi_values = np.asarray([
        solution.get("psi", 0.0)
        if isinstance(solution, dict)
        else 0.0
        for solution in ik_solutions
    ])

    fig, axes = plt.subplots(
        6,
        7,
        figsize=(18, 14),
        squeeze=False,
        constrained_layout=True,
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

    # Use a common colour scale for psi.
    norm = plt.Normalize(
        vmin=np.min(psi_values),
        vmax=np.max(psi_values),
    )
    cmap = plt.get_cmap("viridis")

    for row in range(6):
        theta_row = row + 1  # theta_2,...,theta_7

        for col in range(7):
            ax = axes[row, col]

            # Keep only the lower triangle: theta_i versus theta_j, i > j.
            if col >= theta_row:
                ax.axis("off")
                continue

            scatter = ax.scatter(
                q_values[:, col],
                q_values[:, theta_row],
                c=psi_values,
                cmap=cmap,
                norm=norm,
                s=8,
                alpha=0.75,
                edgecolors="none",
            )

            ax.grid(True, alpha=0.3)

            # Column labels on top row.
            if row == 0:
                ax.set_title(joint_labels[col], fontsize=12)

            # Row labels on the left.
            if col == 0:
                ax.set_ylabel(joint_labels[theta_row], fontsize=12)

            # Only show x-axis labels on bottom visible plots.
            if row == 5:
                ax.set_xlabel(joint_labels[col], fontsize=11)

    # Shared colour bar for psi.
    cbar = fig.colorbar(
        scatter,
        ax=axes,
        shrink=0.75,
        pad=0.02,
    )
    cbar.set_label(r"self-motion parameter $\psi$")

    fig.suptitle(
        "Pairwise Projections of the 7-DoF Self-Motion Manifold",
        fontsize=16,
    )

    plt.savefig("model/analytic_ik/tests/self-motion-manifold.png")
    return fig, axes

ik_solver = Analytical_IK_7DoF()

q_true = np.array([
    0.30,
    -0.40,
    0.50,
    0.70,
    -0.60,
    0.40,
    0.20,
])

T_true = ik_solver.FK(q_true)

psi_samples = np.linspace(-np.pi, np.pi, 401)

solutions = get_all_ik_samples(
	    ik_solver,
	    T_true,
	    psi_samples,
	)

print(f"Number of valid sampled solutions: {len(solutions)}")

branch_solutions = [
    solution
    for solution in solutions
    if np.array_equal(solution["GC"], np.array([1, -1, 1]))
]

plot_smm(branch_solutions)