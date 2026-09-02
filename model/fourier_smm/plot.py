"""
Plot SMMs
compare null-space project vs learned fourier approximator
"""
import numpy as np
from pipeline import TASKConfig
from realtime_smm.learning import SMMNetworkBundle
import matplotlib.pyplot as plt

def plot_overlay(robot, bundle, T, save_path):
    exact = robot.workspace_smms(
        T,
        samples=1000
    )

    predicts = bundle(T, samples = 1000)

    def _branches(ws):
        return [branch.angle.astype(np.float32, copy=False) for branch in ws.data if branch.angle is not None]

    exact_branches = _branches(exact)
    predicted_branches = _branches(predicts)

    # print(f"exact branches: {exact_branches}")
    # print(f"predicted branches: {predicted_branches}")
    for pred_branch in predicted_branches:
        diff = np.abs(pred_branch[1:] - pred_branch[:-1])
        mask = np.amax(diff, axis=1) > np.pi
        pred_branch[:-1][mask] = np.nan

    for exact_branch in exact_branches:
        diff = np.abs(exact_branch[1:] - exact_branch[:-1])
        mask = np.amax(diff, axis=1) > np.pi
        exact_branch[:-1][mask] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)
    pairs = ((0, 1, r"$\theta_1$ vs $\theta_2$"), (1, 2, r"$\theta_2$ vs $\theta_3$"))

    for idx, branch in enumerate(exact_branches):
        for ax, (i, j, label) in zip(axes, pairs):
            ax.plot(
                branch[:, i],
                branch[:, j],
                color="dodgerblue",
                linewidth=6.0,
                zorder=1,
                alpha=0.2,
                label="Exact" if idx == 0 else None,
            )

    for idx, branch in enumerate(predicted_branches):
        for ax, (i, j, label) in zip(axes, pairs):
            ax.plot(
                branch[:, i],
                branch[:, j],
                color="dodgerblue",
                linewidth=1.5,
                zorder=2,
                label="Bundle" if idx == 0 else None,
            )

    for ax, (_, _, label) in zip(axes, pairs):
        ax.set_xlabel(label.split(" vs ")[0])
        ax.set_ylabel(label.split(" vs ")[1])
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="best")
    plt.suptitle("SMM Overlay")
    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    robot_name = "3R"
    task = "planar"

    TaskConfig = TASKConfig(
        robot_name=robot_name,
        task = task
    )

    robot = TaskConfig.get_robot

    folder_path = TaskConfig.get_saved_path
    print(f"path is: {TaskConfig.get_saved_path}")
    BASE_DIR = folder_path
    bundle = SMMNetworkBundle.load(name="3R", base_dir=BASE_DIR, device="cpu")

    T = np.eye(4)
    T[0, 3] = 0.7
    T[1, 3] = 0.1

    save_path = folder_path / "figures" / f"{robot_name}_{task}.png"

    plot_overlay(
        robot = robot,
        bundle = bundle,
        T=T,
        save_path=save_path
    )