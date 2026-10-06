"""
Generate a (q, x) dataset for the self-motion-manifold project.

Robots:
    1. planar 3R    : loaded from robotics-toolbox (rtb)       -> assets/3R/planar3r.npz
    2. franka panda : loaded from franka_emika_panda/panda.xml -> assets/franka_emika_panda/7r_pose.npz
    3. kuka iiwa14  : loaded from kuka_iiwa_14/iiwa14.xml      -> assets/7R/kuka_iiwa_14/7r_pose.npz

Dataset convention:
    qs : (N, q_dim) joint configurations, Sobol-sampled over the joint limits
    xs : (N, x_dim) end-effector poses,
         x_dim = 2 (planar) or 9 (pose = position + first two rotation-matrix columns)

Only pose datasets are generated for the 7R arms; for position tasks simply use
xs[:, :3] of the pose dataset (see RobotConfig.x_dim, task="position").
"""

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import roboticstoolbox as rtb
from scipy.stats import qmc
from tqdm import tqdm

# for franka tdcr
import re

ROOT_PATH = Path(__file__).resolve().parents[0]

# for franka tdcr
TDCR_JOINT_PATTERN = re.compile(
    r"^joint_\d+_.+$"
)

@dataclass(frozen=True)
class RobotConfig:
    name: str
    backend: str
    robot: object
    save_path: Path
    task: str
    x_max: float

    xml_path: Path | None = None

    # Used for ordinary joint-coordinate robots.
    joint_names: tuple[str, ...] = ()

    ee_type: str = "site"
    ee_name: str = "attachment_site"

    # "joint"       : ordinary MuJoCo joint representation
    # "panda_clark" : 7 Panda joints + 2 Clark coordinates per TDCR segment
    q_representation: str = "joint"

    # Used by the reduced Panda + TDCR representation.
    panda_joint_names: tuple[str, ...] = ()
    clark_num_segments: int = 3
    clark_tendon_distance_mm: float = 4.0

    # For ordinary TDCR joints this is the artificial hinge limit.
    # For Clark coordinates this is the maximum bending angle.
    tdcr_limit_deg: float = 45.0

    def _ids_from_names(
        self,
        names: tuple[str, ...],
    ) -> list[int]:
        """Convert MuJoCo joint names to joint IDs."""

        joint_ids = []

        for name in names:
            joint_id = mujoco.mj_name2id(
                self.robot,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )

            if joint_id < 0:
                raise ValueError(
                    f"Joint '{name}' was not found in the MuJoCo model."
                )

            joint_ids.append(int(joint_id))

        return joint_ids

    @property
    def _panda_jnt_ids(self) -> list[int]:
        """Return the seven Panda joint IDs."""

        if self.panda_joint_names:
            names = self.panda_joint_names
        else:
            names = tuple(
                f"panda_joint{i}"
                for i in range(1, 8)
            )

        joint_ids = self._ids_from_names(names)

        if len(joint_ids) != 7:
            raise ValueError(
                "The reduced Panda-Clark representation requires "
                "exactly seven Panda joints."
            )

        return joint_ids

    @property
    def _jnt_ids(self) -> list[int]:
        """
        Return physical MuJoCo joint IDs.

        For panda_clark, this returns only the Panda joint IDs.
        Clark coordinates are not MuJoCo joints, so they do not have
        MuJoCo joint IDs.
        """

        if self.backend == "rtb":
            return []

        if self.backend != "mujoco":
            raise ValueError(
                f"Invalid backend: {self.backend}"
            )

        if self.q_representation == "panda_clark":
            return self._panda_jnt_ids

        if self.q_representation != "joint":
            raise ValueError(
                f"Unknown q representation: "
                f"{self.q_representation}"
            )

        # Explicitly specified joint representation.
        if self.joint_names:
            joint_ids = self._ids_from_names(
                self.joint_names
            )

        else:
            # Automatic raw Panda + TDCR hinge representation.
            joint_ids = []

            for joint_id in range(self.robot.njnt):
                joint_name = mujoco.mj_id2name(
                    self.robot,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    joint_id,
                )

                if joint_name is None:
                    continue

                is_hinge = (
                    self.robot.jnt_type[joint_id]
                    == mujoco.mjtJoint.mjJNT_HINGE
                )

                is_panda_joint = joint_name.startswith(
                    "panda"
                )

                is_tdcr_joint = (
                    TDCR_JOINT_PATTERN.match(
                        joint_name
                    )
                    is not None
                )

                if (
                    is_hinge
                    and (
                        is_panda_joint
                        or is_tdcr_joint
                    )
                ):
                    joint_ids.append(joint_id)

        # Match MuJoCo qpos ordering.
        joint_ids.sort(
            key=lambda jid: self.robot.jnt_qposadr[jid]
        )

        return joint_ids

    @property
    def clark_radius_mm(self) -> float:
        """Maximum radius of each Clark-coordinate disk."""

        if self.q_representation != "panda_clark":
            raise ValueError(
                "Clark radius is only defined for panda_clark."
            )

        return (
            self.clark_tendon_distance_mm
            * np.deg2rad(self.tdcr_limit_deg)
        )

    @property
    def _joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return bounds for the stored q representation.

        For panda_clark, the Clark bounds are componentwise box bounds.
        The actual valid Clark domain is a disk per segment.
        """

        if self.q_representation == "panda_clark":
            lower = []
            upper = []

            # Panda joint limits.
            for joint_id in self._panda_jnt_ids:
                if not self.robot.jnt_limited[joint_id]:
                    joint_name = mujoco.mj_id2name(
                        self.robot,
                        mujoco.mjtObj.mjOBJ_JOINT,
                        joint_id,
                    )

                    raise ValueError(
                        f"Panda joint '{joint_name}' has no limits."
                    )

                lo, hi = self.robot.jnt_range[joint_id]
                lower.append(float(lo))
                upper.append(float(hi))

            # Clark-coordinate component bounds.
            #
            # These are only the enclosing square:
            #
            #     -radius <= c_x,c_y <= radius
            #
            # The true valid region is:
            #
            #     c_x^2 + c_y^2 <= radius^2
            radius = self.clark_radius_mm

            for _ in range(self.clark_num_segments):
                lower.extend([-radius, -radius])
                upper.extend([radius, radius])

            return (
                np.asarray(lower, dtype=float),
                np.asarray(upper, dtype=float),
            )

        lower = []
        upper = []

        tdcr_limit = np.deg2rad(
            self.tdcr_limit_deg
        )

        for joint_id in self._jnt_ids:
            joint_name = mujoco.mj_id2name(
                self.robot,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_id,
            )

            if self.robot.jnt_limited[joint_id]:
                lo, hi = self.robot.jnt_range[joint_id]

            elif (
                joint_name is not None
                and TDCR_JOINT_PATTERN.match(
                    joint_name
                ) is not None
            ):
                lo = -tdcr_limit
                hi = tdcr_limit

            else:
                raise ValueError(
                    f"No joint limits available for "
                    f"'{joint_name}'."
                )

            lower.append(float(lo))
            upper.append(float(hi))

        return (
            np.asarray(lower, dtype=float),
            np.asarray(upper, dtype=float),
        )

    @property
    def q_min(self) -> np.ndarray:
        if self.backend == "rtb":
            return self.robot.qlim[0]

        q_min, _ = self._joint_limits
        return q_min

    @property
    def q_max(self) -> np.ndarray:
        if self.backend == "rtb":
            return self.robot.qlim[1]

        _, q_max = self._joint_limits
        return q_max

    @property
    def q_dim(self) -> int:
        if self.backend == "rtb":
            return self.robot.n

        if self.backend == "mujoco":
            if self.q_representation == "panda_clark":
                return 7 + 2 * self.clark_num_segments

            return len(self._jnt_ids)

        raise ValueError(
            f"Invalid backend: {self.backend}"
        )

    @property
    def x_dim(self) -> int:
        if self.task == "planar":
            return 2

        if self.task == "position":
            return 3

        if self.task == "pose":
            return 9

        raise ValueError(
            f"Invalid task: {self.task}"
        )


ROBOT_CONFIGS = {
    "3R": RobotConfig(
        name="3R",
        backend="rtb",
        robot=rtb.models.DH.Planar3(),
        save_path=ROOT_PATH / "3R" / "planar3r.npz",
        task="planar",
        x_max=3.0
    ),
    "franka_emika_panda": RobotConfig(
        name="franka_emika_panda",
        backend="mujoco",
        robot=mujoco.MjModel.from_xml_path(str(ROOT_PATH / "franka_emika_panda" / "panda_nohand.xml")),
        xml_path=ROOT_PATH / "franka_emika_panda" / "panda.xml",
        save_path=ROOT_PATH / "7R" / "franka_emika_panda" / "7r_pose.npz",
        task="pose",
        x_max=1.2,
        joint_names=tuple(f"joint{i}" for i in range(1, 8)),
        ee_type="site",      # panda.xml has no attachment_site, use the hand body frame
        ee_name="attachment_site",
    ),
    "franka_emika_panda_position": RobotConfig(
        name="franka_emika_panda",
        backend="mujoco",
        robot=mujoco.MjModel.from_xml_path(str(ROOT_PATH / "franka_emika_panda" / "panda_nohand.xml")),
        xml_path=ROOT_PATH / "franka_emika_panda" / "panda.xml",
        save_path=ROOT_PATH / "7R" / "franka_emika_panda" / "7r_pose.npz",
        task="position",
        x_max=1.2,
        joint_names=tuple(f"joint{i}" for i in range(1, 8)),
        ee_type="site",      # panda.xml has no attachment_site, use the hand body frame
        ee_name="attachment_site",
    ),
    "kuka_iiwa_14": RobotConfig(
        name="kuka_iiwa_14",
        backend="mujoco",
        robot=mujoco.MjModel.from_xml_path(str(ROOT_PATH / "kuka_iiwa_14" / "iiwa14.xml")),
        xml_path=ROOT_PATH / "kuka_iiwa_14" / "iiwa14.xml",
        save_path=ROOT_PATH / "7R" / "kuka_iiwa_14" / "7r_pose.npz",
        task="pose",
        x_max=1.2,
        joint_names=tuple(f"joint{i}" for i in range(1, 8)),
        ee_type="site",
        ee_name="attachment_site",
    ),
    "kuka_iiwa_14_position": RobotConfig(
        name="kuka_iiwa_14",
        backend="mujoco",
        robot=mujoco.MjModel.from_xml_path(str(ROOT_PATH / "kuka_iiwa_14" / "iiwa14.xml")),
        xml_path=ROOT_PATH / "kuka_iiwa_14" / "iiwa14.xml",
        save_path=ROOT_PATH / "7R" / "kuka_iiwa_14" / "7r_pose.npz",
        task="position",
        x_max=1.2,
        joint_names=tuple(f"joint{i}" for i in range(1, 8)),
        ee_type="site",
        ee_name="attachment_site",
    ),
    "franka_tdcr": RobotConfig(
        name="franka_tdcr",
        backend="mujoco",
        robot=mujoco.MjModel.from_xml_path(
            str(
                ROOT_PATH
                / "opencr-mujoco"
                / "assets"
                / "example_three_segment_franka_franka_scene.xml"
            )
        ),
        xml_path=(
            ROOT_PATH
            / "opencr-mujoco"
            / "assets"
            / "example_three_segment_franka_franka_scene.xml"
        ),
        save_path=(
            ROOT_PATH
            / "franka_tdcr"
            / "franka_tdcr_dataset.npz"
        ),
        task="pose",
        x_max=1.6,

        # Correct TDCR tip definition
        ee_type="body",
        ee_name="EE_pos",

        q_representation="panda_clark",
        panda_joint_names=tuple(
            f"panda_joint{i}"
            for i in range(1, 8)
        ),
        clark_num_segments=3,
        clark_tendon_distance_mm=4.0,
        tdcr_limit_deg=45.0,
    )
}


def get_robot_config(robot_name: str) -> RobotConfig:
    try:
        return ROBOT_CONFIGS[robot_name]
    except KeyError:
        raise NameError("The model name is invalid.")


def print_robot_config(config: RobotConfig):
    print(f"{config.name} manipulator is loaded ({config.backend})")
    if config.backend == "rtb":
        print(config.robot)
    else:
        print(f"xml path: {config.xml_path}")
    print(f'minimum joint limits in rad:')
    print(config.q_min)
    print(f'maximum joint limits in rad:')
    print(config.q_max)
    print(f'dataset will be saved at: {config.save_path}')


def sobel_joint_samples(
    n_samples: int,
    config: RobotConfig,
    seed=None,
):
    # scramble: introduce randomness
    sampler = qmc.Sobol(
        d=config.q_dim,
        scramble=True,
        seed=seed,
    )

    # sobel generates samples 2^m
    m = math.ceil(math.log2(max(n_samples, 1)))

    # extract first n_samples
    u = sampler.random_base2(m=m)[:n_samples]

    return qmc.scale(
        u,
        config.q_min,
        config.q_max,
    )


def mujoco_fk(qs: np.ndarray, config: RobotConfig) -> np.ndarray:
    """Forward kinematics of the mujoco robot at the end-effector frame.

    Returns (N, 9) poses [x, y, z, R_col0, R_col1]; task="position" slices [:3].
    """
    model = config.robot
    data = mujoco.MjData(model)

    jnt_adrs = [model.jnt_qposadr[jid] for jid in config._jnt_ids]

    if config.ee_type == "site":
        ee_id = model.site(config.ee_name).id
    elif config.ee_type == "body":
        ee_id = model.body(config.ee_name).id
    else:
        raise ValueError(f"Invalid ee_type: {config.ee_type}")

    poses = np.empty((len(qs), 9))

    time_start = time.time()
    for i, q in enumerate(tqdm(qs, desc=f"{config.name} FK")):
        data.qpos[:] = 0.0  # un-sampled dofs (e.g. panda fingers) fixed at 0
        for adr, qi in zip(jnt_adrs, q):
            data.qpos[adr] = qi

        mujoco.mj_kinematics(model, data)

        if config.ee_type == "site":
            pos = data.site_xpos[ee_id]
            xmat = data.site_xmat[ee_id]
        else:
            pos = data.xpos[ee_id]
            xmat = data.xmat[ee_id]

        poses[i, :3] = pos
        poses[i, 3:] = xmat[:6]  # first two rotation-matrix columns
    print(f"time = {(time.time() - time_start) * 1000:.3f} ms")

    if config.task == "position":
        return poses[:, :3]  # position task is just the pose position columns
    return poses


def generate_dataset(
    n_samples: int,
    config: RobotConfig,
    seed=None,
):
    qs = sobel_joint_samples(
        n_samples=n_samples,
        config=config,
        seed=seed,
    )

    if config.backend == "rtb":
        # planar motion
        T = np.array(config.robot.fkine(qs).A)
        xs = T[:, :2, 3]

    elif config.backend == "mujoco":
        xs = mujoco_fk(qs, config)

    else:
        raise ValueError(f"Invalid backend: {config.backend}")

    config.save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(config.save_path, qs=qs, xs=xs)
    print(f"Saved {n_samples} samples to {config.save_path}")
    return qs, xs


def plot_space(
    config: RobotConfig,
    q: np.ndarray,
    p: np.ndarray,
):
    if config.task == "planar":
        # plot configuration space theta2-theta3 plane and workspace
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # theta2-theta3 plane
        ax1.scatter(q[:, 1], q[:, 2], color = 'blue')
        ax1.set_xlabel(r'$\theta_2$ (rad)')
        ax1.set_ylabel(r'$\theta_3$ (rad)')
        ax1.set_xlim(config.q_min[0] - 1, config.q_max[0] + 1)
        ax1.set_ylim(config.q_min[0] - 1, config.q_max[0] + 1)

        ax1.set_aspect('equal', adjustable='box')
        ax1.grid(True, alpha=0.3)

        # draw the workspace boundary
        # Draw theoretical workspace boundary
        theta = np.linspace(0, 2 * np.pi, 150)
        r = 3.0

        x_boundary = r * np.cos(theta)
        y_boundary = r * np.sin(theta)

        ax2.plot(
            x_boundary,
            y_boundary,
            color='red',
            linewidth=2,
            label='Workspace boundary'
        )

        # Plot FK positions
        ax2.scatter(
            p[:, 0],
            p[:, 1],
            color='blue',
            label='Samples'
        )

        ax2.set_xlabel('x')
        ax2.set_ylabel('y')

        ax2.set_xlim(-r-1, r+1)
        ax2.set_ylim(-r-1, r+1)

        ax2.set_aspect('equal', adjustable='box')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.show()

    else:
        # High-level layout:
        # left  = 3x3 joint-space histograms
        # right = 3D workspace
        fig = plt.figure(figsize=(16, 7))

        outer = fig.add_gridspec(
            1, 2,
            width_ratios=[1.2, 1.0],
            wspace=0.25,
        )

        # ---------------------------------------------------------
        # Left: 3x3 grid containing histograms for the 7 joints
        # ---------------------------------------------------------
        joint_grid = outer[0].subgridspec(
            3, 3,
            hspace=0.5,
            wspace=0.35,
        )

        for i in range(config.q_dim):
            ax = fig.add_subplot(joint_grid[i // 3, i % 3])

            ax.hist(q[:, i], bins=30)

            ax.set_title(rf"$q_{i+1}$")
            ax.set_xlabel("Joint angle (rad)")
            ax.set_ylabel("Count")
            ax.set_xlim(config.q_min[i], config.q_max[i])
            ax.grid(True, alpha=0.3)

        # Turn the two unused cells off
        for i in range(config.q_dim, 9):
            ax = fig.add_subplot(joint_grid[i // 3, i % 3])
            ax.axis("off")

        # ---------------------------------------------------------
        # Right: 3D workspace
        # ---------------------------------------------------------
        ax_workspace = fig.add_subplot(outer[1], projection="3d")

        ax_workspace.scatter(
            p[:, 0],
            p[:, 1],
            p[:, 2],
            s=15,
            # alpha=0.5,
        )

        ax_workspace.set_xlabel("x (m)")
        ax_workspace.set_ylabel("y (m)")
        ax_workspace.set_zlabel("z (m)")
        ax_workspace.set_title("Workspace")

        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a (q, x) dataset for the SMM project.")
    parser.add_argument(
        "--robot",
        type=str,
        default="3R",
        choices=list(ROBOT_CONFIGS),
        help="robot name in ROBOT_CONFIGS",
    )
    parser.add_argument("--samples", type=int, default=2000000, help="number of joint samples")
    parser.add_argument("--seed", type=int, default=42, help="random seed for the Sobol sampler")
    parser.add_argument("--plot", action="store_true", help="plot the sampled dataset")
    args = parser.parse_args()

    config = get_robot_config(args.robot)
    print_robot_config(config)

    qs, xs = generate_dataset(
        n_samples=args.samples,
        config=config,
        seed=args.seed,
    )

    if args.plot:
        plot_space(config, qs, xs)
