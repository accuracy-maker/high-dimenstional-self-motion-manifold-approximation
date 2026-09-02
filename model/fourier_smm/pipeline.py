"""
Trace all distinct SMMs of Planar 3R manipulator with unit link length

3R manipulator has 3 DoF but task (planar motion) just needs 2-D (x, y).
There would be 1-D SMMs which are curves
"""


import time
import pickle
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass

# realtime_smm functions
from realtime_smm import TaskSpace, TrainingConfig
from realtime_smm.helpers.types import AxisParams, NodeStage, SMMStatus
from realtime_smm.grid import Grid
from realtime_smm.learning import SMMNetworkBundle, train_cluster_networks
from realtime_smm.postprocess_grid import postprocess_grid

from robots.planar3r import planar3r
from robots.panda import panda, canonical_roll_frame, fibonacci_sphere, roll_offset, rotz
from robots.iiwa import iiwa

ROOT_PATH = Path(__file__).resolve().parents[0]
print(f"root path: {ROOT_PATH}")

@dataclass
class TASKConfig:
    robot_name: str
    task: str # planar, pose

    @property
    def get_robot(self):
        if self.robot_name == "3R":
            robot = planar3r()

        elif self.robot_name == "panda":
            robot = panda()

        elif self.robot_name == "iiwa":
            robot = iiwa()

        else:
            raise NameError("invalid robot name")

        return robot

    @property
    def get_taskspace(self):
        if self.robot_name == "3R":
            if self.task == "planar":
                TASKSPACE = TaskSpace.X | TaskSpace.Y
                return TASKSPACE
            else:
                raise NameError("for 3R, only support planar task")

        elif self.robot_name == "panda":
            if self.task == "pose":
                TASKSPACE = TaskSpace.X | TaskSpace.Y | TaskSpace.Z | TaskSpace.SO3
                return TASKSPACE
            else:
                raise NameError("task is not valid")

        elif self.robot_name == "iiwa":
            if self.task == "pose":
                TASKSPACE = TaskSpace.X | TaskSpace.Y | TaskSpace.Z | TaskSpace.SO3
                return TASKSPACE
            else:
                raise NameError("task is not valid")
        else:
            raise NameError("current robot name is invalid")

    @property
    def get_saved_path(self) -> Path:
        path = ROOT_PATH / "results" / f"{self.robot_name}_{self.task}"
        path.mkdir(parents=True, exist_ok=True)
        return path

# help functions
def section(t):
    print(f"\n--- {t} " + "-" * max(0, 60 - len(t)))

"""
Clark & Xie mentioned two approches to reduce the dimention when doing workspace
clustering:
    world-z rotation: Rz(a) @ T <=> q1 += a (handled by the package's XY half-plane option)
    tool-z rotation: T @ Rz(b)  <=> q7 += b 

The package samples the SO3 axis uniformly over all of SO(3), which ignores the
second symmetry.  Replacing that sampler with canonical-roll frames over a
Fibonacci sphere turns the orientation axis from a 3-D into a 2-D grid
"""
def install_roll_reduced_SO3(n_dirs: int):
    """Replace original uniform SO3 sampling"""
    def sampler(count, seed=0):
        return [canonical_roll_frame(d) for d in fibonacci_sphere(int(count))]

    Grid._sample_uniform_so3 = staticmethod(sampler)

def build_grid(
    task_config,
    pos_res=0.1,
    x_range=(-3.0, 3.0),
    z_range=None,
    n_dirs=None,
    SO3_k=6,
):
    if task_config.robot_name == "3R":
        axis_params = [
            AxisParams(
                axis=TaskSpace.X,
                lower=x_range[0],
                upper=x_range[1],
                resolution=pos_res,
            ),
            AxisParams(
                axis=TaskSpace.Y,
                lower=0.0,
                upper=0.0,
                resolution=pos_res,
            ),
        ]

        grid = Grid(task_config.get_taskspace, axis_params)

        grid.use_xy_halfplane = True

        return grid

    elif task_config.robot_name in ("panda", "iiwa"):

        if n_dirs is None:
            raise ValueError("n_dirs must be provided for Panda/iiwa")

        if z_range is None:
            raise ValueError("z_range must be provided for Panda/iiwa")

        install_roll_reduced_SO3(n_dirs)

        orn_res = float(np.sqrt(4.0 * np.pi / n_dirs)) + 1e-9

        axis_params = [
            AxisParams(
                axis=TaskSpace.X,
                lower=x_range[0],
                upper=x_range[1],
                resolution=pos_res,
            ),
            AxisParams(
                axis=TaskSpace.Y,
                lower=0.0,
                upper=0.0,
                resolution=pos_res,
            ),
            AxisParams(
                axis=TaskSpace.Z,
                lower=z_range[0],
                upper=z_range[1],
                resolution=pos_res,
            ),
            AxisParams(
                axis=TaskSpace.SO3,
                resolution=orn_res,
            ),
        ]

        grid = Grid(
            task_config.get_taskspace,
            axis_params,
            so3_k=SO3_k,
        )

        grid.use_xy_halfplane = True

        return grid

class RollReducedBundle:
    def __init__(self, bundle: SMMNetworkBundle):
        self.bundle = bundle

    def __call__(self, T, samples=None):
        T = np.asarray(T, dtype=float)
        beta = roll_offset(T[:3, :3])
        Tc = T.copy()
        Tc[:3, :3] = T[:3, :3] @ rotz(-beta)
        ws = self.bundle(Tc, samples=samples)
        for smm in ws.data:
            if smm.data is not None:
                smm.data[:, 6] = np.angle(np.exp(1j * (smm.data[:, 6] + beta)))
        return ws


if __name__ == "__main__":
    robot_name = "3R"
    task = "planar"

    TaskConfig = TASKConfig(
        robot_name=robot_name,
        task = task
    )

    folder_path = TaskConfig.get_saved_path
    print(f"path is: {TaskConfig.get_saved_path}")

    robot = TaskConfig.get_robot

    t_all = time.time()

    section("1. grid and SMM data")
    grid_pkl = folder_path / f"{robot_name}_{task}_grid.pkl"
    grid = build_grid(
        task_config=TaskConfig,
        pos_res=0.005,
        x_range=[-3.0, 3.0],
    )

    nodes = list(grid.graph.nodes)

    Ts = np.array([grid.graph.nodes[n]["T"] for n in nodes])
    print(f"[grid] {len(nodes)} nodes  ({grid.sizes})  "
              f"{grid.graph.number_of_edges()} edges", flush=True)

    t0 = time.time()

    def prog(done, total):
            el = time.time() - t0
            print(f"[smm] {done}/{total}  {el:.0f}s elapsed  "
                  f"{el / max(done,1):.3f} s/node  eta {el/max(done,1)*(total-done):.0f}s",
                  flush=True)

    res = robot.workspace_smms_many(
            Ts, samples=128, step=0.05, smm_iters=1200,
            seeds=40, chunk=1000, progress=prog)

    for n, ws in zip(nodes, res):
        grid.set_node_smms(n, ws)
        grid.graph.nodes[n]["status"] = NodeStage.COMPUTED_SMMS

    stat = {}
    for ws in res:
        k = ws.status.name if ws.status != SMMStatus.OK else f"OK-{len(ws.data)}"
        stat[k] = stat.get(k, 0) + 1
    print(f"[smm] done in {time.time()-t0:.0f}s  node status: {stat}", flush=True)
    with open(grid_pkl, "wb") as f:
        pickle.dump(grid, f)

    section("2. clustering and homotopy grouping")
    clusters_pkl = folder_path / f"{robot_name}_{task}_clusters.pkl"
    with open(grid_pkl, "rb") as f:
        grid = pickle.load(f)

    t0 = time.time()
    clusters = postprocess_grid(grid)
    clusters = [c for c in clusters if c.number_of_nodes() > 0]
    sizes = sorted((c.number_of_nodes() for c in clusters), reverse=True)
    print(f"[cluster] {len(clusters)} clusters in {time.time()-t0:.0f}s  "
              f"sizes {sizes[:12]}{'...' if len(sizes)>12 else ''}", flush=True)
    with open(clusters_pkl, "wb") as f:
        pickle.dump((grid, clusters), f)

    section("3. training")
    with open(clusters_pkl, "rb") as f:
        grid, clusters = pickle.load(f)

    keep = [c for c in clusters if c.number_of_nodes() >= 8]
    print(f"[train] training on {len(keep)}/{len(clusters)} clusters "
            f"(>=8 nodes)", flush=True)
    cfg = TrainingConfig(
        epochs=1500, batch_size=64, learning_rate=1e-3,
        weight_decay=1e-4, hidden_dims=(150,) * 6, activation="leaky_relu",
        fft_cutoff=24, device="cuda")
    t0 = time.time()
    bundle = train_cluster_networks(grid, keep, name=robot_name, config=cfg,
                                    output_root=folder_path)
    print(f"[train] done in {time.time()-t0:.0f}s -> {folder_path}/{robot_name}", flush=True)

    print(f"[all] total {time.time()-t_all:.0f}s", flush=True)