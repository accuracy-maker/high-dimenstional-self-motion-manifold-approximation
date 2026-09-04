"""
ODE for 1-D SMM

reference: Dominic Guri and George Kantor "ODE Methods for Computing One-Dimensional Self-Motion  Manifolds"
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

# robot config
from assets.data_generation import RobotConfig, get_robot_config

import mujoco
import roboticstoolbox as rtb

class NearSingularityError(RuntimeError):
    """Raised when the planar task Jacobian no longer has rank two."""

@dataclass
class SMMTrace:
    q: np.ndarray
    closed: bool # if the curve is closed
    stop_reason: str
    max_position_error: float
    mean_position_error: float
    max_orientation_error: float | None
    mean_orientation_error: float | None
    minimum_task_singular_value: float

def wrap_to_pi(angle: np.ndarray | float) -> np.ndarray:
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi

def torus_delta(q_a: np.ndarray, q_b: np.ndarray) -> np.ndarray:
    return wrap_to_pi(np.asarray(q_a) - np.asarray(q_b))

def torus_distance(q_a: np.ndarray, q_b: np.ndarray) -> float:
    """Euclidean distance after wrap angles"""
    return float(np.linalg.norm(torus_delta(q_a, q_b)))

# def build_planar_3r(lengths: Sequence[float]) -> Any:
#     """A standard-DH planar RRR robot"""

#     lengths = np.asarray(lengths, dtype=float)

#     if lengths.shape != (3,) or np.any(lengths <= 0.0):
#         raise ValueError("lengths must contain three positive values")

#     links = [
#         rtb.RevoluteDH(a=float(length), qlim=[-np.pi, np.pi])
#         for length in lengths
#     ]

#     return rtb.DHRobot(links, name="Planar 3R")

# def target(robot: Any, q: np.ndarray) -> np.ndarray:
#     """target is (2,) if robot is 3R; (6,) if robot is 7R"""
#     return np.asarray(robot.fkine(np.asarray(q, dtype=float)).t[:2], dtype=float)

def target(robot_cfg: RobotConfig, q: np.ndarray) -> np.ndarray:
    if robot_cfg.name == "3R":
        return np.asarray(robot_cfg.robot.fkine(np.asarray(q, dtype=float)).t[:2], dtype=float)

    elif robot_cfg.name == "franka_emika_panda" or robot_cfg.name == "kuka_iiwa_14":
        model = robot_cfg.robot
        data = mujoco.MjData(model)
        data.qpos[:7] = q
        mujoco.mj_kinematics(model, data)

        # read position and rotation
        site_id =  model.site(robot_cfg.ee_name).id
        p = data.site_xpos[site_id]
        R =  data.site_xmat[site_id].reshape(3, 3)
        # homogeneous transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p

        return T

    else:
        raise NameError("invalid robot name")
        

# def task_jacobian(robot: Any, q: np.ndarray) -> np.ndarray:
#     jacobian = np.asarray(robot.jacob0(np.asarray(q, dtype=float)), dtype=float)
#     if jacobian.shape[0] < 2 or jacobian.shape[1] != 3:
#         raise ValueError(f"expected a Jacobian compatible with shape (6, 3), got {jacobian.shape}")
#     return jacobian[:2, :]

def task_jacobian(robot_cfg: RobotConfig, q: np.ndarray) -> np.ndarray:
    if robot_cfg.name == "3R":
        jacobian = np.asarray(robot_cfg.robot.jacob0(np.asarray(q, dtype=float)), dtype=float)
        if jacobian.shape[0] < 2 or jacobian.shape[1] != 3:
            raise ValueError(f"expected a Jacobian compatible with shape (6, 3), got {jacobian.shape}")
        return jacobian[:2, :]

    elif robot_cfg.name == "franka_emika_panda" or robot_cfg.name == "kuka_iiwa_14":
        model = robot_cfg.robot
        data = mujoco.MjData(model)

        data.qpos[:7] = q
        # here should use forward to compute jacobian
        mujoco.mj_forward(model, data)

        site_id =  model.site(robot_cfg.ee_name).id

        # pos and rot Jac
        jac_pos = np.zeros((3, model.nv))
        jac_rot = np.zeros((3, model.nv))

        mujoco.mj_jacSite(
            model,
            data,
            jac_pos,
            jac_rot,
            site_id,
        )

        jacobian = np.vstack([
                jac_pos[:, :7],
                jac_rot[:, :7],
            ])

        if jacobian.shape != (6,7):
            raise ValueError(f"expected a Jacobian compatible with shape (6, 7), got {jacobian.shape}")

        return jacobian

    else:
        raise NameError("invalid robot name")


def unit_kernel_vector(
    robot_cfg: RobotConfig,
    q: np.ndarray,
    singularity_tolerance: float
):
    """compute the normalised 1-D kernel of the task Jacobian (Eq (4) in the papaer)"""
    jacobian = task_jacobian(robot_cfg, q)

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
    robot_cfg: RobotConfig,
    q: np.ndarray,
    reference: np.ndarray,
    singularity_tolerance: float
) -> tuple[np.ndarray, float]:
    """resolve the successive SVD sign ambiguity using a reference heading Eq(5)"""
    tangent, sigma_min = unit_kernel_vector(robot_cfg, q, singularity_tolerance)
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

def pose_error(target_pose, curr_pose):
    """inputs are all 4x4 homogeneous matrix"""
    pos_err= np.linalg.norm(target_pose[:3,3] - curr_pose[:3, 3])

    R_err = target_pose[:3, :3] @ curr_pose[:3, :3].T
    cos_angle = (np.trace(R_err) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
            
    ori_err = np.arccos(cos_angle)

    return pos_err, ori_err

def trace_smm_component(
    robot_cfg: RobotConfig,
    q0: Sequence[float],
    step_size: float = 0.05,
    closure_tolerance: float | None = None,
    minimum_steps: int = 30,
    maximum_steps: int = 20_000,
    singularity_tolerance: float = 1e-9,
) -> SMMTrace:
    """Integrate one connected SMM component from an initial IK solution"""
    q0_array = np.asarray(q0, dtype=float).reshape(-1)

    if step_size <= 0.0:
        raise ValueError("step_size must be positive")

    if minimum_steps < 1 or maximum_steps <= minimum_steps:
        raise ValueError("require 1 <= minimum_steps < maximum_steps")

    closure_tolerance = (
        float(step_size) if closure_tolerance is None else float(closure_tolerance)
    )
    if closure_tolerance <= 0.0:
        raise ValueError("closure_tolerance must be positive")

    T = target(robot_cfg, q0_array)

    initial_tangent, sigma_initial = unit_kernel_vector(
        robot_cfg, q0_array, singularity_tolerance
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
                robot_cfg, q_current, reference, singularity_tolerance
            )
            minimum_sigma = min(minimum_sigma, sigma)

            # Eq(7)
            def rhs(q_stage: np.ndarray) -> np.ndarray:
                tangent, _ = regularised_tangent(
                    robot_cfg, q_stage, tangent_reference, singularity_tolerance
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


    if robot_cfg.name == "3R":
        position_errors = np.array(
                [np.linalg.norm(target(robot_cfg, q) - T) for q in q_trace],
                dtype=float
            )

        return SMMTrace(
            q=q_trace,
            closed=closed,
            stop_reason=stop_reason,
            max_position_error=float(np.max(position_errors)),
            mean_position_error=float(np.mean(position_errors)),
            max_orientation_error = None,
            mean_orientation_error= None,
            minimum_task_singular_value=float(minimum_sigma),
        )

    elif robot_cfg.name == "franka_emika_panda" or robot_cfg.name == "kuka_iiwa_14":
        position_errors = np.array(
            [np.linalg.norm(target(robot_cfg, q)[:3, 3] - T[:3, 3]) for q in q_trace],
            dtype=float
        )

        # T is target, target(robot_cfg, q) is current pose
        #  R_err = R_d @ R_c^T
        R_err = np.array(
            [T[:3, :3] @ target(robot_cfg, q)[:3, :3].T for q in q_trace],
            dtype=float
        )

        cos_angle = (
                np.trace(R_err, axis1=1, axis2=2) - 1.0
            ) / 2.0

        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        ori_errors = np.arccos(cos_angle)

        return SMMTrace(
            q=q_trace,
            closed=closed,
            stop_reason=stop_reason,
            max_position_error=float(np.max(position_errors)),
            mean_position_error=float(np.mean(position_errors)),
            max_orientation_error=float(np.max(ori_errors)),
            mean_orientation_error=float(np.mean(ori_errors)),
            minimum_task_singular_value=float(minimum_sigma),
        )

    else:
        raise NameError("invalid robot name")


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
    robot_cfg: RobotConfig,
    q0: Sequence[float]
) -> list[np.ndarray]:

    robot = robot_cfg.robot

    q0_array = np.asarray(q0, dtype=float).reshape(3)
    x = target(robot_cfg, q0_array)
    candidates = [q0_array]

    for elbow in (1, 2):
        try:
            candidate = toggle_elbow(robot, q0_array, elbow)
        except NearSingularityError:
            continue

        if np.linalg.norm(target(robot_cfg, candidate) - x) > 1e-9:
            raise RuntimeError(
                "elbow toggle failed to preserve the target position"
            )

        candidates.append(candidate)

    return unique_configs(candidates)

# def distance_to_component(
#     q: np.ndarray,
#     component: SMMTrace
# ) -> float:
#     """Distance from one config to a sampled SMM component"""

#     differences = wrap_to_pi(component.q - np.asarray(q, dtype=float).reshape(1, 3))
#     return float(np.min(np.linalg.norm(differences, axis=1)))

def distance_to_component(
    q: np.ndarray,
    component: SMMTrace
) -> float:
    """Distance from one config to a sampled SMM component"""

    q = np.asarray(q, dtype=float).reshape(1, -1)
    differences = wrap_to_pi(component.q - q)

    return float(
        np.min(np.linalg.norm(differences, axis=1))
    )

def random_joint_configuration(robot_cfg: RobotConfig, rng: np.random.Generator) -> np.ndarray:
    """sample a 7R configs uniformly from its joint limits"""
    q_min = robot_cfg.q_min
    q_max = robot_cfg.q_max

    return rng.uniform(q_min, q_max)

@dataclass
class IKSolution:
    q: np.ndarray
    success: bool

def solve_ik_from_seed(
    robot_cfg: RobotConfig,
    x: np.ndarray,
    q0: np.ndarray,
    ilimit: int, # iteration limit
    slimit: int, # search attempt limit
    tol: float,
    joint_limits: bool
) -> IKSolution:
    """One search attempt LM-IK solver"""

    q0 = np.asarray(q0, dtype=float)

    for search_idx in range(slimit):

        # first attempt starts from q0
        if search_idx == 0:
            q = q0.copy()

        # later attempts use random joint configurations
        else:
            q = np.random.uniform(
                robot_cfg.q_min,
                robot_cfg.q_max,
            )

        for _ in range(ilimit):

            # current EE pose
            T = target(
                robot_cfg=robot_cfg,
                q=q,
            )

            # target pose
            T_des = x

            # position error
            e_pos = T_des[:3, 3] - T[:3, 3]

            # orientation error in base/world frame
            R = T[:3, :3]
            R_des = T_des[:3, :3]

            R_err = R_des @ R.T

            e_rot = 0.5 * np.array([
                R_err[2, 1] - R_err[1, 2],
                R_err[0, 2] - R_err[2, 0],
                R_err[1, 0] - R_err[0, 1],
            ])

            # task-space error
            e = np.concatenate([
                e_pos,
                e_rot,
            ])

            # convergence
            if np.linalg.norm(e) < tol:
                return IKSolution(
                    q=q,
                    success=True,
                )

            # task Jacobian
            J = task_jacobian(
                robot_cfg=robot_cfg,
                q=q,
            )

            # LM / damped least-squares
            damping = 1e-3

            dq = J.T @ np.linalg.solve(
                J @ J.T
                + damping**2 * np.eye(J.shape[0]),
                e,
            )

            # update
            q = q + dq

            # enforce joint limits
            if joint_limits:
                q = np.clip(
                    q,
                    robot_cfg.q_min,
                    robot_cfg.q_max,
                )

    return IKSolution(
        q=q,
        success=False,
    )


def configuration_distance(q_a: np.ndarray, q_b: np.ndarray) -> float:
    """Euclidean norm of the wrapped joint displacement."""

    return float(np.linalg.norm(wrap_to_pi(np.asarray(q_a, dtype=float) - np.asarray(q_b, dtype=float))))

def distance_to_seed_set(q: np.ndarray, seeds: Sequence[np.ndarray]) -> float:
    """Minimum wrapped distance from q to a set of configurations."""

    if not seeds:
        return np.inf
    return min(configuration_distance(q, previous) for previous in seeds)

def generate_ik_seeds(
    robot_cfg: RobotConfig,
    x: np.ndarray,
    q0: Sequence[float],
    *,
    number_of_attempts: int = 150,
    random_seed: int = 0,
    ik_tolerance: float = 1e-10,
    ik_iteration_limit: int = 100,
    uniqueness_tolerance: float = 1e-4,
    respect_joint_limits: bool = False,
    validation_position_tolerance: float = 1e-5,
    validation_orientation_tolerance: float = 1e-5
) -> list[np.ndarray]:
    """Generate Q* by repeatedly solving IK from random initial guess"""
    if number_of_attempts < 0:
        raise ValueError("number_of_attempts must be non-negative")

    q0_array = np.asarray(q0, dtype=float).reshape(-1)

    seeds: list[np.ndarray] = [q0_array.copy()]
    rng = np.random.default_rng(random_seed)

    for _ in range(number_of_attempts):
        q_guess = random_joint_configuration(robot_cfg, rng)
        try:
            q_solution = solve_ik_from_seed(
                robot_cfg=robot_cfg,
                x = x,
                q0=q_guess,
                ilimit=ik_iteration_limit,
                slimit=1,
                tol=ik_tolerance,
                joint_limits=respect_joint_limits
            )
        except (ValueError, np.linalg.LinAlgError):
            continue

        if q_solution.success is False:
            continue

        curr = target(robot_cfg, q_solution.q)

        pos_err, ori_err = pose_error(
            target_pose=x,
            curr_pose=curr
        )

        if (
            pos_err > validation_position_tolerance
            or ori_err > validation_orientation_tolerance
        ):
            continue

        if distance_to_seed_set(q_solution.q, seeds) > uniqueness_tolerance:
            seeds.append(q_solution.q)

    return seeds


# search all SMM components
def search_smm_components(
    robot_cfg: RobotConfig,
    seeds: Sequence[np.ndarray],
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

    for seed_index, seed in enumerate(seeds, start=1):
        represented = any(
            distance_to_component(np.asarray(seed, dtype=float), component)
            <= repre_tol
            for component in components
        )
        if represented:
            continue

        try:
            component = trace_smm_component(
                robot_cfg,
                seed,
                step_size=step_size,
                closure_tolerance=closure_tolerance,
                minimum_steps=minimum_steps,
                maximum_steps=maximum_steps,
                singularity_tolerance=singularity_tolerance,
            )
        
        except NearSingularityError as exc:
            print(f"Skipping singular IK seed {seed_index}: {exc}")
            continue

        components.append(component)

    return components

if __name__ == "__main__":
    robot_name = "kuka_iiwa_14"
    robot_cfg = get_robot_config(robot_name=robot_name)

    x = target(
        robot_cfg=robot_cfg,
        q = [1, 1, 1, 1, 1, 1, 1]
    )

    print(f"x = {x}")

    jacobian = task_jacobian(
        robot_cfg=robot_cfg,
        q = [1, 1, 1, 1, 1, 1, 1]
    )

    print(f'task jacobian: {jacobian}')

    unit_vec, sigma_min = unit_kernel_vector(
        robot_cfg=robot_cfg,
        q = [1, 1, 1, 1, 1, 1, 1],
        singularity_tolerance=1e-9
    )

    norm = np.linalg.norm(unit_vec)

    print(f"unit null vector: {unit_vec} | norm = {norm}")

    print(f"sigma min: {sigma_min}")

    smm = trace_smm_component(
        robot_cfg=robot_cfg,
        q0 = [1, 1, 1, 1, 1, 1, 1],
        closure_tolerance=0.05
    )

    print(f"smm q trace:\n {smm.q}")
    print(f"smm is close:\n {smm.closed}")
    print(f"smm stop reason:\n {smm.stop_reason}")
    print(f"max position error\n: {smm.max_position_error}")
    print(f"max orientation error\n: {smm.max_orientation_error}")


    seeds = generate_ik_seeds(
        robot_cfg=robot_cfg,
        x = x,
        q0=[1, 1, 1, 1, 1, 1, 1],
    )


    # compute all smm
    components = search_smm_components(
        robot_cfg=robot_cfg,
        seeds=seeds,
        closure_tolerance=0.05
    )

    print("*" * 10)
    print(f"len of components: {len(components)}")

    for idx, comp in enumerate(components):
        print(f"component: {idx + 1}")
        # print(f"smm q trace:\n {comp.q}")
        print(f"smm is close:\n {comp.closed}")
        print(f"smm stop reason:\n {comp.stop_reason}")
        print(f"max position error\n: {comp.max_position_error}")
        print(f"max orientation error\n: {comp.max_orientation_error}")
