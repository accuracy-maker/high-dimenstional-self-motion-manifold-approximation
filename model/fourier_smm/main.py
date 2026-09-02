"""
This is the main file that assembles:
1. run pipeline: workspace clustering, homotopy classes grouping, training neural network
2. run evaluation: FK_error before and after correct
3. plot SMMs: null-space projection and learned ones
"""

import time
import argparse
import pickle
from pathlib import Path
import numpy as np

from pipeline import TASKConfig, section, build_grid
from plot import plot_overlay
from robots.panda import canonical_roll_frame, rotz, SE3

from realtime_smm import TaskSpace, TrainingConfig
from realtime_smm.helpers.types import AxisParams, NodeStage, SMMStatus
from realtime_smm.postprocess_grid import postprocess_grid
from realtime_smm.learning import SMMNetworkBundle, train_cluster_networks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--robot_name", help="name of the robot", default="3R")
    ap.add_argument("-t", "--task", help="name of task: planar or pose", default='planar')

    # workspace args
    ap.add_argument("-x", "--x_range", help="range of x-axis", type=float, nargs=2, default=[-3.0, 3.0])
    ap.add_argument("-y", "--y_range", help="range of y-axis", type=float, nargs=2, default=[-3.0, 3.0])
    ap.add_argument("-z", "--z_range", help="range of z-axis", type=float, nargs=2, default=[0.0, 0.8])
    ap.add_argument("--dirs", help="number of directions of SO(2)", type=int, default=42)
    ap.add_argument("--pos_res", help="resolution of the grid", type=float, default=0.1)


    # fourier smm
    ap.add_argument("--samples", help="number of samples at each curve", type=int, default=128)
    ap.add_argument("--step", help="step size of smm trace", type=float, default=0.05)
    ap.add_argument("--smm_iters", help="maximum number of tracing iterations", type=int, default=1200)
    ap.add_argument("--seeds", help="number of seeds searched for IK solve", type=int, default=40)
    ap.add_argument("--chunk", help="batch of pose processed at once", type=int, default=500)

    # training
    ap.add_argument("-e", "--epoch", help="number of epochs for training", type=int, default=1500)
    ap.add_argument("-fc", "--fft_cutoff", help="cut off high frequnce fourier coefficients", type=int, default=24)

    # evalaute
    ap.add_argument("--n_evals", help="number of targets testing in evaluation", type=int, default=10000)
    ap.add_argument("--n_correct", help="number of corrections steps of dls ik", type=int, default=3)

    # global set
    ap.add_argument("--stage", help="method stages: grid, cluster, train, evaluate, plot", default="all")
    ap.add_argument("--seed", help="seed for random generator", type=int, default=42)
    args = ap.parse_args()

    # global variables
    TaskConfig = TASKConfig(
        robot_name = args.robot_name,
        task = args.task
    )

    robot = TaskConfig.get_robot
    print("Fast SMM Robot created")

    folder_path = TaskConfig.get_saved_path
    print(f"folder path is: {folder_path}")

    t_all = time.time()

    #---------------------------------------------------------
    if args.stage in ("all", "grid"):
        section("1. grid and SMM data")
        grid_pkl = folder_path / f"{args.robot_name}_{args.task}_grid.pkl"
    
        grid = build_grid(
            task_config=TaskConfig,
            pos_res=args.pos_res,
            x_range=args.x_range,
            z_range=args.z_range,
            n_dirs=args.dirs
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
                Ts,
                samples=args.samples,
                step=args.step,
                smm_iters=args.smm_iters,
                seeds=args.seeds, 
                chunk=args.chunk, 
                progress=prog)

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

    #---------------------------------------------------------
    if args.stage in ("all", "cluster"):
        section("2. clustering and homotopy grouping")
        clusters_pkl = folder_path / f"{args.robot_name}_{args.task}_clusters.pkl"
    
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


    #---------------------------------------------------------
    if args.stage in ("all", "train"):    
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

        bundle = train_cluster_networks(grid, keep, name=args.robot_name, config=cfg,
                                        output_root=folder_path)
        print(f"[train] done in {time.time()-t0:.0f}s -> {folder_path}/{args.robot_name}", flush=True)

    #---------------------------------------------------------
    if args.stage in ("all", "evaluate"):
        section("4. evaluation")
        rng = np.random.default_rng(args.seed)

        if args.robot_name == "3R":
            xs, ys = [], []
            while len(xs) < args.n_evals:
                n_needed = args.n_evals - len(xs)
                cx = rng.uniform(*args.x_range, int(n_needed * 1.4) + 16)
                cy = rng.uniform(*args.y_range, int(n_needed * 1.4) + 16)
                keep = cx ** 2 + cy ** 2 <= 3.0 ** 2
                xs.extend(cx[keep][:n_needed])
                ys.extend(cy[keep][:n_needed])
            xs, ys = np.array(xs), np.array(ys)

            x_rep_chunks, q_raw_chunks = [], []
            n_ok = 0
            for i, (x, y) in enumerate(zip(xs, ys)):
                T = np.eye(4)
                T[0, 3], T[1, 3] = x, y
                ws = bundle(T, samples=args.samples)
                if ws.status.name != "OK":
                    continue
                n_ok += 1
                q = np.concatenate([b.data.astype(float) for b in ws.data], axis=0)
                x_rep_chunks.append(np.tile([x, y], (q.shape[0], 1)))
                q_raw_chunks.append(q)
                if (i + 1) % 2000 == 0:
                    print(f"  ...{i + 1}/{args.n_evals} targets")
            x_rep = np.concatenate(x_rep_chunks)
            q_raw = np.concatenate(q_raw_chunks)

            T_target = np.tile(np.eye(4), (q_raw.shape[0], 1, 1))
            T_target[:, 0, 3] = x_rep[:, 0]
            T_target[:, 1, 3] = x_rep[:, 1]

            ep, _ = robot.bk.fk_error_pct(q_raw, T_target)
            # print(f"ep mean: {ep.mean()} | ep min: {ep.min()}")
            err_pct = np.maximum(ep * 100.0, 1e-16)
            print(f"Fourier-series SMM raw output ({n_ok}/{args.n_evals} reachable targets, "
                    f"{q_raw.shape[0]} configs): mean {err_pct.mean():.3g}% | "
                    f"median {np.median(err_pct):.3g}% | max {err_pct.max():.3g}%")

            q_corrected = robot.bk.ik_correct(q_raw, T_target, iters=args.n_correct)
            ep_c, _ = robot.bk.fk_error_pct(q_corrected, T_target)
            err_corrected_pct = np.maximum(ep_c * 100.0, 1e-16)
            print(f"  + {args.n_correct} Jacobian IK-correction steps: "
                    f"mean {err_corrected_pct.mean():.3g}% | median {np.median(err_corrected_pct):.3g}% | "
                    f"max {err_corrected_pct.max():.3g}%")

        elif args.robot_name in ("panda", "iiwa"):
            # generate target samples
            x = rng.uniform(*args.x_range, size=args.n_evals)
            z = rng.uniform(*args.z_range, size=args.n_evals)
            alpha = rng.uniform(0, 2 * np.pi, size=args.n_evals)
            v = rng.normal(size=(args.n_evals, 3))
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            roll = rng.uniform(-np.pi, np.pi, size=args.n_evals)
            Ts = np.empty((args.n_evals, 4, 4))
            for i in range(args.n_evals):
                R = canonical_roll_frame(v[i]) @ rotz(roll[i])
                p = np.array([x[i] * np.cos(alpha[i]), x[i] * np.sin(alpha[i]), z[i]])
                Ts[i] = SE3(R, p)

            # qs = np.empty((args.n_evals,  7))
            # generate qs
            ep, eo = [], []

            for i, T in enumerate(Ts):
                # print(f"T shape: {T.shape}")
                ws = bundle(T, samples=args.samples)
                q = np.concatenate([b.angle.astype(float) for b in ws.data], axis=0)

                # repeat T_rep
                T_rep = np.tile(np.eye(4), (q.shape[0], 1, 1))
                # print(f"q shape: {q.shape}")
                # print(f"T_rep shape: {T_rep.shape}")
                # qs[i,:] = q
                ep_i, eo_i = robot.bk.fk_error_pct(q, T_rep)
                # print(f"target {i}, ep = {ep.mean() :.2f}, eo = {eo.mean() :.2f}")
                ep.append(ep_i.mean())
                eo.append(eo_i.mean())

            err = 0.5 * (np.mean(ep) + np.mean(eo))
            print(f"pos error mean {np.mean(ep) :.2f} | ori error mean: {np.mean(eo) :.2f} | overall error: {err :.2f}")



    #---------------------------------------------------------
    if args.stage in ("all", "plot"):
        section("5. plotting")
        fig_path = folder_path / "figures" / f"{args.robot_name}_{args.task}.png"
        fig_path.mkdir(parents=True, exist_ok=True)

        T = np.eye(4)
        T[0, 3] = 0.7
        T[1, 3] = 0.3
        T[2, 3] = 0.3

        # Make up an orientation
        rx = np.deg2rad(30.0)
        ry = np.deg2rad(20.0)
        rz = np.deg2rad(45.0)

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx),  np.cos(rx)],
        ])

        Ry = np.array([
            [ np.cos(ry), 0, np.sin(ry)],
            [0,           1, 0],
            [-np.sin(ry), 0, np.cos(ry)],
        ])

        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz),  np.cos(rz), 0],
            [0,           0,          1],
        ])

        R = Rz @ Ry @ Rx

        T[:3, :3] = R


        plot_overlay(
            robot = robot,
            bundle = bundle,
            T=T,
            save_path=fig_path
        )

    print(f"[all] total {time.time()-t_all:.0f}s", flush=True)

if __name__ == "__main__":
    main()
                
