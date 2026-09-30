"""
generate 4-D SMM in the situation where the target pose is just position
r = 7 - 3 = 4

The algorithm is:
    - given a position p in R^3
    - generate a orientation direction in SO(3)
    - solve the IK solutions by this given SE(3)

sampling diverse orientation direction in SO(3) repeat this algorithm
"""

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


from model.analytic_ik.analytic_ik_7dof import Analytical_IK_7DoF


T_7_SITE = np.array(
    [
        [-1.0, 0.0, 0.0, 0.000],
        [0.0, -1.0, 0.0, 0.000],
        [0.0, 0.0, 1.0, 0.045],
        [0.0, 0.0, 0.0, 1.000],
    ],
    dtype=float,
)

TARGET_POSITION_SITE = np.array(
    [0.03263181, 0.39944908, 0.48280907],
    dtype=float,
)

NUM_ORIENTATIONS = 128
NUM_PSI = 256
RANDOM_SEED = 42
ENFORCE_JOINT_LIMITS = False
POSE_TOLERANCE = 1e-5
ACOS_TOLERANCE = 1e-8

DATA_PATH = "model/analytic_ik/tests/data/analytic_4d_smm.npz"
FIGURE_PATH = "model/analytic_ik/tests/figures/analytic_4d_smm.png"


def sample_so3(number, rng):
    """Generate Haar-uniform rotations and their quaternions."""

    if number <= 0:
        raise ValueError("number must be positive")

    u = rng.random((number, 3))
    s = np.sqrt(u[:, 0])
    t = np.sqrt(1.0 - u[:, 0])
    a = 2.0 * np.pi * u[:, 1]
    b = 2.0 * np.pi * u[:, 2]

    x = t * np.sin(a)
    y = t * np.cos(a)
    z = s * np.sin(b)
    w = s * np.cos(b)

    rotations = np.empty((number, 3, 3), dtype=float)
    rotations[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rotations[:, 0, 1] = 2.0 * (x * y - z * w)
    rotations[:, 0, 2] = 2.0 * (x * z + y * w)
    rotations[:, 1, 0] = 2.0 * (x * y + z * w)
    rotations[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rotations[:, 1, 2] = 2.0 * (y * z - x * w)
    rotations[:, 2, 0] = 2.0 * (x * z - y * w)
    rotations[:, 2, 1] = 2.0 * (y * z + x * w)
    rotations[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)

    quaternions = np.column_stack((x, y, z, w))
    return rotations, quaternions


def make_site_pose(position, rotation):
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(rotation, dtype=float)
    T[:3, 3] = np.asarray(position, dtype=float)
    return T


def site_pose_to_dh_pose(T_site):
    return np.asarray(T_site, dtype=float) @ np.linalg.inv(T_7_SITE)


def pose_errors(T_a, T_b):
    position_error = np.linalg.norm(T_a[:3, 3] - T_b[:3, 3])
    rotation_error = np.linalg.norm(T_a[:3, :3] - T_b[:3, :3])
    return float(position_error), float(rotation_error)


def compute_4d_smm(
    ik,
    target_position_site,
    rotations,
    psi_samples,
    *,
    pose_tolerance=POSE_TOLERANCE,
    acos_tolerance=ACOS_TOLERANCE,
    enforce_joint_limits=False,
    verbose=True,
):
    target_position_site = np.asarray(target_position_site, dtype=float)
    rotations = np.asarray(rotations, dtype=float)
    psi_samples = np.asarray(psi_samples, dtype=float)

    if target_position_site.shape != (3,):
        raise ValueError("target_position_site must have shape (3,).")
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError("rotations must have shape (N, 3, 3).")
    if psi_samples.ndim != 1:
        raise ValueError("psi_samples must be one-dimensional.")

    q_values = []
    orientation_indices = []
    gc_values = []
    psi_values = []
    position_errors = []
    rotation_errors = []

    counts = {
        "attempted": 0,
        "infeasible": 0,
        "joint_limit_rejected": 0,
        "pose_rejected": 0,
        "exceptions": 0,
    }

    for orientation_index, rotation in enumerate(rotations):
        T_site = make_site_pose(target_position_site, rotation)
        T_dh = site_pose_to_dh_pose(T_site)

        for gc_tuple in product((-1, 1), repeat=3):
            GC = np.asarray(gc_tuple, dtype=int)

            for psi in psi_samples:
                counts["attempted"] += 1

                try:
                    unclipped = np.asarray(
                        ik.IK(
                            rigid_transform=T_dh,
                            GC=GC,
                            psi=float(psi),
                            return_unclipped_vals=True,
                            clip_stepback=0.0,
                        ),
                        dtype=float,
                    )

                    if unclipped.shape != (4,):
                        counts["exceptions"] += 1
                        continue

                    if np.any(np.abs(unclipped) > 1.0 + acos_tolerance):
                        counts["infeasible"] += 1
                        continue

                    q = np.asarray(
                        ik.IK(
                            rigid_transform=T_dh,
                            GC=GC,
                            psi=float(psi),
                            clip_stepback=0.0,
                        ),
                        dtype=float,
                    )

                    if q.shape != (7,) or not np.all(np.isfinite(q)):
                        counts["pose_rejected"] += 1
                        continue

                    if enforce_joint_limits:
                        inside_limits = (
                            np.all(q >= ik.limits_lower - 1e-8)
                            and np.all(q <= ik.limits_upper + 1e-8)
                        )
                        if not inside_limits:
                            counts["joint_limit_rejected"] += 1
                            continue

                    T_check = np.asarray(ik.FK(q), dtype=float)
                    position_error, rotation_error = pose_errors(
                        T_check,
                        T_dh,
                    )

                    if (
                        position_error > pose_tolerance
                        or rotation_error > pose_tolerance
                    ):
                        counts["pose_rejected"] += 1
                        continue

                    q_values.append(q)
                    orientation_indices.append(orientation_index)
                    gc_values.append(GC)
                    psi_values.append(float(psi))
                    position_errors.append(position_error)
                    rotation_errors.append(rotation_error)

                except Exception as error:
                    counts["exceptions"] += 1
                    if verbose and counts["exceptions"] <= 5:
                        print(
                            f"Rejected orientation={orientation_index}, "
                            f"GC={GC}, psi={psi:.6f}: {error}"
                        )

    if q_values:
        q_values = np.asarray(q_values, dtype=float)
        orientation_indices = np.asarray(orientation_indices, dtype=int)
        gc_values = np.asarray(gc_values, dtype=int)
        psi_values = np.asarray(psi_values, dtype=float)
        position_errors = np.asarray(position_errors, dtype=float)
        rotation_errors = np.asarray(rotation_errors, dtype=float)
    else:
        q_values = np.empty((0, 7), dtype=float)
        orientation_indices = np.empty((0,), dtype=int)
        gc_values = np.empty((0, 3), dtype=int)
        psi_values = np.empty((0,), dtype=float)
        position_errors = np.empty((0,), dtype=float)
        rotation_errors = np.empty((0,), dtype=float)

    if verbose:
        print(f"Analytical 4-D samples: {len(q_values)}")
        print(f"Sampling counts: {counts}")

    return {
        "q": q_values,
        "orientation_index": orientation_indices,
        "GC": gc_values,
        "psi": psi_values,
        "position_error": position_errors,
        "rotation_error": rotation_errors,
        "rotations": rotations,
        "target_position_site": target_position_site,
        "counts": counts,
    }


def save_smm(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "q": data["q"],
        "orientation_index": data["orientation_index"],
        "GC": data["GC"],
        "psi": data["psi"],
        "position_error": data["position_error"],
        "rotation_error": data["rotation_error"],
        "rotations": data["rotations"],
        "target_position_site": data["target_position_site"],
    }

    if "quaternions" in data:
        payload["quaternions"] = data["quaternions"]

    np.savez_compressed(path, **payload)


def plot_joint_projections(data, path):
    q = data["q"]
    if len(q) == 0:
        return None

    q = (q + np.pi) % (2.0 * np.pi) - np.pi
    colour = data["orientation_index"]

    fig, axes = plt.subplots(6, 7, figsize=(18, 14), squeeze=False)
    labels = [rf"$\theta_{i}$" for i in range(1, 8)]
    scatter = None

    for row in range(6):
        joint_y = row + 1

        for col in range(7):
            joint_x = col
            ax = axes[row, col]

            if joint_y <= joint_x:
                ax.axis("off")
                continue

            scatter = ax.scatter(
                q[:, joint_x],
                q[:, joint_y],
                c=colour,
                cmap="viridis",
                s=2,
                alpha=0.35,
                edgecolors="none",
            )

            ax.set_xlim(-np.pi, np.pi)
            ax.set_ylim(-np.pi, np.pi)
            ax.grid(True, alpha=0.25)

            if row == 5:
                ax.set_xlabel(labels[joint_x])
            else:
                ax.tick_params(axis="x", labelbottom=False)

            if col == 0:
                ax.set_ylabel(labels[joint_y])
            else:
                ax.tick_params(axis="y", labelleft=False)

    if scatter is not None:
        fig.colorbar(
            scatter,
            ax=axes.ravel().tolist(),
            shrink=0.65,
            label="orientation sample index",
        )

    fig.subplots_adjust(wspace=0.08, hspace=0.08)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    return fig


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    rotations, quaternions = sample_so3(NUM_ORIENTATIONS, rng)
    psi_samples = np.linspace(-np.pi, np.pi, NUM_PSI, endpoint=False)

    ik_solver = Analytical_IK_7DoF()
    data = compute_4d_smm(
        ik=ik_solver,
        target_position_site=TARGET_POSITION_SITE,
        rotations=rotations,
        psi_samples=psi_samples,
        enforce_joint_limits=ENFORCE_JOINT_LIMITS,
        pose_tolerance=POSE_TOLERANCE,
        acos_tolerance=ACOS_TOLERANCE,
        verbose=True,
    )
    data["quaternions"] = quaternions

    save_smm(data, DATA_PATH)
    plot_joint_projections(data, FIGURE_PATH)

    print(f"Saved samples to: {DATA_PATH}")
    print(f"Saved projections to: {FIGURE_PATH}")


if __name__ == "__main__":
    main()