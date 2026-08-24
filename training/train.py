"""train flow-matching model based on the dataset"""

import numpy as np
import math
import time
import matplotlib.pyplot as plt
from model.flow_matching import FMConfig, FlowMatching, load_data, DataLoader

import torch
import torch.nn as nn



def train(cfg: FMConfig) -> FlowMatching:
    # reproducible
    torch.manual_seed(cfg.seed)
    print(f"training on the device: {cfg.device}")
    # read data
    train_set, test_set = load_data(cfg)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=4096)

    # flow-matching model
    fm = FlowMatching(cfg)
    opt = torch.optim.AdamW(fm.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.n_epochs)

    best_val = float("inf")

    # training
    for epoch in range(cfg.n_epochs):
        fm.model.train()
        train_loss, n_batches = 0.0, 0

        for q1, x in train_loader:
            q1, x = q1.to(cfg.device), x.to(cfg.device)
            loss = fm.loss(q1, x)
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item()
            n_batches += 1
        sched.step()

        fm.model.eval()
        with torch.no_grad():
            val_loss = sum(
                fm.loss(q1.to(cfg.device), x.to(cfg.device)).item() for q1, x in test_loader
            ) / len(test_loader)

        if val_loss < best_val:
            best_val = val_loss
            fm.save()

        if epoch % 10 == 0 or epoch == cfg.n_epochs - 1:
            print(
                f"epoch {epoch:4d} | train {train_loss / n_batches:.4f} "
                f"| val {val_loss:.4f} | best {best_val:.4f}"
            )

    fm.load()
    return fm


if __name__ == "__main__":
    cfg = FMConfig()
    fm = train(cfg)

