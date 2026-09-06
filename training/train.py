"""train flow-matching model based on the dataset"""

import numpy as np
import argparse
from sklearn import metrics
from tqdm import tqdm
import matplotlib.pyplot as plt
from model.flow_matching import FMConfig, FlowMatching, load_data, DataLoader
import time
import json

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

    train_metrics = {
        "train_time": 0.0,
        "train_loss": [],
        "valid_loss": [],
    }

    t_s = time.time()

    for epoch in tqdm(range(cfg.n_epochs)):
        fm.model.train()

        train_loss = 0.0
        n_train = 0

        for q1, x in train_loader:
            q1, x = q1.to(cfg.device), x.to(cfg.device)

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

                loss = fm.loss(q1, x)

                batch_size = q1.shape[0]
                val_loss += loss.item() * batch_size
                n_val += batch_size

        train_loss /= n_train
        val_loss /= n_val

        train_metrics["train_loss"].append(train_loss)
        train_metrics["valid_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            fm.save()

    t_train = time.time() - t_s
    train_metrics["train_time"] = t_train

    # save dict as json
    json_path = f"training/{cfg.robot_name}.train_metrics.json"
    with open(json_path, "w") as f:
        json.dump(train_metrics, f, indent=4)
    print(f"training metrics saved to {json_path}")


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
    fm = train(cfg)

