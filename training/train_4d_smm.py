"""train flow-matching model based on the dataset"""

import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from model.flow_matching import FMConfig, FlowMatching, load_data, DataLoader

import torch
import torch.nn as nn



def train(cfg: FMConfig) -> FlowMatching:
    # reproducible
    torch.manual_seed(cfg.seed)
    print(f"training on the device: {cfg.device}")

    # read data
    train_set, test_set, norm = load_data(cfg)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=4096,
    )

    # flow-matching model
    fm = FlowMatching(cfg, norm)

    opt = torch.optim.AdamW(
        fm.model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=cfg.n_epochs,
    )

    best_val = float("inf")

    # training
    for epoch in tqdm(range(cfg.n_epochs)):
        fm.model.train()
        train_loss = 0.0
        n_train = 0

        for q1, x in train_loader:
            q1, x = q1.to(cfg.device), x.to(cfg.device)
            # print(f"x shape:{x.shape}")
            x = x[:, :3]
            # print(f"x shape: {x.shape}")
            loss = fm.loss(q1, x)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            batch_size = q1.shape[0]
            train_loss += loss.item() * batch_size
            n_train += batch_size

        sched.step()

        fm.model.eval()
        val_loss = 0.0
        n_val = 0

        with torch.no_grad():
            for q1, x in test_loader:
                q1, x = q1.to(cfg.device), x.to(cfg.device)
                x = x[:, :3]
                loss = fm.loss(q1, x)

                batch_size = q1.shape[0]
                val_loss += loss.item() * batch_size
                n_val += batch_size

        train_loss /= n_train
        val_loss /= n_val

        if val_loss < best_val:
            best_val = val_loss
            fm.save()

        # if epoch % 10 == 0 or epoch == cfg.n_epochs - 1:
        #     print(
        #         f"epoch {epoch:4d} | train {train_loss:.4f} "
        #         f"| val {val_loss:.4f} | best {best_val:.4f}"
        #     )

    fm.load()
    return fm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training flow-matching model")
    parser.add_argument(
        "--robot_name",
        type=str,
        default="3R",
        help="robot name in ROBOT_CONFIGS",
    )
    args = parser.parse_args()


    cfg = FMConfig(robot_name=args.robot_name)
    robot = cfg.load_robot
    print(f"robot: {robot.name} | task: {robot.task}")
    fm = train(cfg)

