"""
ODE for 1-D SMM

reference: Dominic Guri and George Kantor "ODE Methods for Computing One-Dimensional Self-Motion  Manifolds"
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import roboticstoolbox as rtb

class NearSingularityError(RuntimeError):
    """Raised when the planar task Jacobian no longer has rank two."""

@dataclass
class SMMTrace:
    q: np.ndarray
    closed: bool # if the curve is closed
    stop_reason: str
    max_position_error: float
    minimum_task_singular_value: float

def wrap_to_pi(angle: np.ndarray | float) -> np.ndarray:
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi

def torus_delta(q_a: np.ndarray, q_b: np.ndarray) -> np.ndarray:
    return wrap_to_pi(np.asarray(q_a) - np.asarray(q_b))

def torus_distance(q_a: np.ndarray, q_b: np.ndarray) -> float:
    """Euclidean distance after wrap angles"""
    return float(np.linalg.norm(torus_delta(q_a, q_b)))

def build_planar_3r(lengths: Sequence[float]) -> Any:
    """A standard-DH planar RRR robot"""

    lengths = np.asarray(lengths, dtype=float)

    if lengths.shape != (3,) or np.any(lengths <= 0.0):
        raise ValueError("lengths must contain three positive values")

    links = [
        rtb.RevoluteDH(a=float(length), qlim=[-np.pi, np.pi])
        for length in lengths
    ]

    return rtb.DHRobot(links, name="Planar 3R")

def target(robot: Any, q: np.ndarray) -> np.ndarray:
    """target is (2,) if robot is 3R; (6,) if robot is 7R"""
    return np.asarray(robot.fkine(np.asarray(q, dtype=float)).t[:2], dtype=float)

def task_jacobian(robot: Any, q: np.ndarray) -> np.ndarray:
    jacobian = np.asarray(robot.jacob0(np.asarray(q, dtype=float)), dtype=float)
    if jacobian.shape[0] < 2 or jacobian.shape[1] != 3:
        raise ValueError(f"expected a Jacobian compatible with shape (6, 3), got {jacobian.shape}")
    return jacobian[:2, :]

def unit_kernel_vector(
    robot: Any,
    q: np.ndarray,
    singularity_tolerance: float
):
    """compute the normalised 1-D kernel of the task Jacobian (Eq (4) in the papaer)"""
    jacobian = task_jacobian(robot, q)

    # return: U, Sigma, V
    _, singular_values, vh = np.linalg.svd(jacobian, full_matrices=True)

    # A regular 2x3 task Jacobian must have rank two.
    scale = max(1.0, float(singular_values[0]))
    sigma_min = float(singular_values[-1])
    if sigma_min <= singularity_tolerance * scale:
        raise NearSingularityError(
            f"task Jacobian is singular or ill-conditioned: sigma_min={sigma_min:.3e}"
        )

    # null vector: v3 = vh[-1, :] and Jv3 = 0
    null_vector = np.asarray(vh[-1, :], dtype=float)
    norm = float(np.linalg.norm(null_vector))

    if not np.isfinite(norm) or norm <= np.finfo(float).eps:
        raise NearSingularityError("failed to obtain a finite one-dimensional null vector")

    return null_vector / norm, sigma_min


def regularised_tangent(
    robot: Any,
    q: np.ndarray,
    reference: np.ndarray,
    singularity_tolerance: float
) -> tuple[np.ndarray, float]:
    """resolve the successive SVD sign ambiguity using a reference heading Eq(5)"""
    tangent, sigma_min = unit_kernel_vector(robot, q, singularity_tolerance)
    if float(np.dot(tangent, reference)) <= 0.0:
        tangent = -tangent
    return tangent, sigma_min

def RK5_step(
    rhs: Any, 
    q: np.ndarray,
    step_size: float
) -> np.ndarray:
    """
    Solve an intial value problem (IVP)
    One step of Runge-Kutta step
    Eq(7)

    Cash-Karp RK5 Butcher tableau:

    0
    1/5     | 1/5
    3/10    | 3/40       9/40
    3/5     | 3/10      -9/10       6/5
    1       | -11/54     5/2       -70/27      35/27
    7/8     | 1631/55296 175/512    575/13824  44275/110592  253/4096
    --------|----------------------------------------------------------
    5th     | 37/378      0          250/621    125/594       0  512/1771
    
    """
    h = float(step_size)

    k1 = rhs(q)
    k2 = rhs(q + h * (1.0 / 5.0) * k1)
    k3 = rhs(q + h * ((3.0 / 40.0) * k1 + (9.0 / 40.0) * k2))
    k4 = rhs(
        q
        + h
        * (
            (3.0 / 10.0) * k1
            - (9.0 / 10.0) * k2
            + (6.0 / 5.0) * k3
        )
    )
    k5 = rhs(
        q
        + h
        * (
            -(11.0 / 54.0) * k1
            + (5.0 / 2.0) * k2
            - (70.0 / 27.0) * k3
            + (35.0 / 27.0) * k4
        )
    )
    k6 = rhs(
        q
        + h
        * (
            (1631.0 / 55296.0) * k1
            + (175.0 / 512.0) * k2
            + (575.0 / 13824.0) * k3
            + (44275.0 / 110592.0) * k4
            + (253.0 / 4096.0) * k5
        )
    )

    return q + h * (
        (37.0 / 378.0) * k1
        + (250.0 / 621.0) * k3
        + (125.0 / 594.0) * k4
        + (512.0 / 1771.0) * k6
    )

def trace_smm_component(
    robot: Any,
    q0: Sequence[float],
    step_size: float = 0.05,
    closure_tolerance: float | None = None,
    minimum_steps: int = 30,
    maximum_steps: int = 20_000,
    singularity_tolerance: float = 1e-9,
) -> SMMTrace:
    """Integrate one connected SMM component from an initial IK solution"""
    q0_array = np.asarray(q0, dtype=float).reshape(3)

    if step_size <= 0.0:
        raise ValueError("step_size must be positive")

    if minimum_steps < 1 or maximum_steps <= minimum_steps:
        raise ValueError("require 1 <= minimum_steps < maximum_steps")

    closure_tolerance = (
        float(step_size) if closure_tolerance is None else float(closure_tolerance)
    )
    if closure_tolerance <= 0.0:
        raise ValueError("closure_tolerance must be positive")

    x = target(robot, q0_array)

    initial_tangent, sigma_initial = unit_kernel_vector(
        robot, q0_array, singularity_tolerance
    )

    reference = initial_tangent.copy()
    minimum_sigma = sigma_initial

    samples = [q0_array.copy()]
    closed = False
    stop_reason = "maximum number of integration steps reached"

    for step_idx in range(maximum_steps):
        q_current = samples[-1]

        try:
            # Eq(8)
            tangent_reference, sigma = regularised_tangent(
                robot, q_current, reference, singularity_tolerance
            )
            minimum_sigma = min(minimum_sigma, sigma)

            # Eq(7)
            def rhs(q_stage: np.ndarray) -> np.ndarray:
                tangent, _ = regularised_tangent(
                    robot, q_stage, tangent_reference, singularity_tolerance
                )
                return tangent

            q_next= RK5_step(rhs, q_current, step_size)

        except NearSingularityError as exc:
            stop_reason = str(exc)
            break

        if not np.all(np.isfinite(q_next)):
            stop_reason = "integration produced a non-finite joint configuration"
            break

        samples.append(q_next)
        reference = tangent_reference

        if (
            step_idx + 1 >= minimum_steps
            and torus_distance(q_next, q0_array) <= closure_tolerance
        ):
            closed = True
            stop_reason = "returned to the initial configuration"
            break

    q_trace = np.asarray(samples, dtype=float)
    position_errors = np.array(
        [np.linalg.norm(target(robot, q) - x) for q in q_trace],
        dtype=float,
    )

    return SMMTrace(
        q=q_trace,
        closed=closed,
        stop_reason=stop_reason,
        max_position_error=float(np.max(position_errors)),
        minimum_task_singular_value=float(minimum_sigma),
    )


# help functions for plotting
def joint_points(
    robot: Any,
    q: Sequence[float]
) -> np.ndarray:
    """store each joint position for plotting: a matrix of 4x2 array"""
    transforms = robot.fkine_all(np.asarray(q, dtype=float), old=False) # old = False means output the base frame as well
    points = np.asarray([np.asarray(transform.t[:2], dtype=float) for transform in transforms])
    if points.shape != (4, 2):
        raise ValueError(f"expected four planar frame origins, got {points.shape}")
    return points

def reflect_point_across_line(
    point: np.ndarray,
    point_a: np.ndarray,
    point_b: np.ndarray
) -> np.ndarray:
    direction = np.asarray(point_b, dtype=float) - np.asarray(point_a, dtype=float)
    denominator = float(np.dot(direction, direction))
    if denominator <= 1e-14:
        raise NearSingularityError("cannot toggle an elbow whose neighboring joints coincide")
    projection = point_a + direction * float(np.dot(point - point_a, direction)) / denominator
    return 2.0 * projection - point

def config_from_joint_points(
    joint_points: np.ndarray
) -> np.ndarray:
    """recover configration of 3R manipulator from four plannar points"""
    points = np.asarray(joint_points, dtype=float)

    if points.shape != (4,2):
        raise ValueError("points must have shape (4,2)")

    absolute_angles = np.array(
        [
            np.arctan2(
                points[index + 1, 1] - points[index, 1],
                points[index + 1, 0] - points[index, 0],
            )
            for index in range(3)
        ],
        dtype=float,
    )
    q = np.array(
        [
            absolute_angles[0],
            absolute_angles[1] - absolute_angles[0],
            absolute_angles[2] - absolute_angles[1],
        ],
        dtype=float,
    )
    return wrap_to_pi(q)

def toggle_elbow(
    robot: Any,
    q: Sequence[float],
    elbow: int        
) -> np.ndarray:
    """ Toggle eblow 1 or elbow 2 while preserving the end-effector positions"""
    points = joint_points(robot, q)
    toggled = points.copy()

    if elbow == 1:
        # Keep p0 and p2 fixed, and reflect p1 across the chord p0--p2.
        toggled[1] = reflect_point_across_line(points[1], points[0], points[2])
    elif elbow == 2:
        # Keep p1 and p3 fixed, and reflect p2 across the chord p1--p3.
        toggled[2] = reflect_point_across_line(points[2], points[1], points[3])
    else:
        raise ValueError("elbow must be 1 or 2")

    return config_from_joint_points(toggled)

# unique configuration: two candidates' distance > tolerance
def unique_configs(
    configs: Iterable[np.ndarray],
    tolerance: float = 1e-8
) -> list[np.ndarray]:

    unique: list[np.ndarray] = []
    for config in configs:
        q = np.asarray(config, dtype=float).reshape(3)
        if all(torus_distance(q, previous) > tolerance for previous in unique):
            unique.append(q)
    return unique

def rrr_component_seed(
    robot: Any,
    q0: Sequence[float]
) -> list[np.ndarray]:
    """Construct Q* = {q0, toggle elbow 1, toogle elbow 2}"""

    q0_array = np.asarray(q0, dtype=float).reshape(3)
    x = target(robot, q0_array)
    candidates = [q0_array]

    for elbow in (1, 2):
        try:
            candidate = toggle_elbow(robot, q0_array, elbow)
        except NearSingularityError:
            continue
        if np.linalg.norm(target(robot, candidate) - x) > 1e-9:
            raise RuntimeError("elbow toggle failed to preserve the target position")
        candidates.append(candidate)

    return unique_configs(candidates)

def distance_to_component(
    q: np.ndarray,
    component: SMMTrace
) -> float:
    """Distance from one config to a sampled SMM component"""

    differences = wrap_to_pi(component.q - np.asarray(q, dtype=float).reshape(1, 3))
    return float(np.min(np.linalg.norm(differences, axis=1)))

# search all SMM components
def search_smm_components(
    robot: Any,
    seeds = Sequence[np.ndarray],
    step_size: float = 0.05,
    closure_tolerance: float | None = None,
    minimum_steps: int = 20,
    maximum_steps: int = 20_000,
    singularity_tolerance: float = 1e-9
) -> list[SMMTrace]:
    """Algorithm 1: trace each seed not represented by a previous component"""

    components: list[SMMTrace] = []
    # representation tolerance
    repre_tol = float(step_size)

    for seed in seeds:
        represented = any(
            distance_to_component(np.asarray(seed, dtype=float), component)
            <= repre_tol
            for component in components
        )
        if represented:
            continue

        components.append(
            trace_smm_component(
                robot,
                seed,
                step_size=step_size,
                closure_tolerance=closure_tolerance,
                minimum_steps=minimum_steps,
                maximum_steps=maximum_steps,
                singularity_tolerance=singularity_tolerance,
            )
        )

    return components