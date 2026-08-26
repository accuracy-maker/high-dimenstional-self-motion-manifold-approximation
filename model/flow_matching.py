"""
A flow matching model approximates the conditional distribution P(q|x)
    q: configurations in C-space
    x: target in W-space
"""

import math
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import matplotlib.pyplot as plt

# torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# robot config
from assets.data_generation import RobotConfig, get_robot_config

ROOT_PATH = Path(__file__).resolve().parents[1]
# print(f"root path: {ROOT_PATH}")

@dataclass
class FMConfig:
    # robot info
    robot_name: str = "7R_pose"
    
    @property
    def load_robot(self) -> RobotConfig:
        return get_robot_config(self.robot_name)

    # neural network config
    hidden_dim: int = 256
    n_layers: int = 4
    time_emb_dim: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 512
    n_epochs: int = 1000
    n_ode_steps: int = 100
    test_size: float = 0.1

    # ckpt
    ckpt_path: Path = ROOT_PATH / "training/checkpoints" / f"{robot_name}.pt"
    
    # device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    seed: int = 42

# data preprocessing
def load_data(cfg: FMConfig):
    
    robot = cfg.load_robot
    data = np.load(robot.save_path)
    print(f"data is loaded from {robot.save_path}")

    # normalise angles
    raw_qs = torch.from_numpy(data['qs']).float()
    # print(f"raw qs shape: {raw_qs.shape}")


    qs_mid = torch.as_tensor((robot.q_min + robot.q_max) / 2, dtype=torch.float32) # (N,)
    qs_half = torch.as_tensor((robot.q_max - robot.q_min) / 2, dtype=torch.float32) # (N,)

    norm_qs = (raw_qs - qs_mid) / qs_half
    # print(f"normalised qs shape: {norm_qs.shape}")
    # print(f"minimum values of norm qs: {torch.min(norm_qs, dim=0).values}")
    # print(f"maximum values of norm qs: {torch.max(norm_qs, dim=0).values}")

    
    # normalise position 
    pos = torch.from_numpy(data['xs']).float()

    pos = pos.clone()
    pos[:, :3] = pos[:, :3] / robot.x_max

    # reproducible seed
    n = norm_qs.shape[0] # how many rows
    g = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(n, generator=g)

    # split dataset
    n_test = int(n * cfg.test_size)
    train_idx, test_idx = perm[n_test:], perm[:n_test]

    train_set = TensorDataset(norm_qs[train_idx], pos[train_idx])
    test_set = TensorDataset(norm_qs[test_idx], pos[test_idx])

    return train_set, test_set

# Flow matching model
class TimeEmbedding(nn.Module):
    """Sinusoidal embedding of t in [0,1]"""

    def __init__(self, dim: int):
        super().__init__()

        half = dim // 2
        # linespace is uniform in log-space
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), half))
        self.register_buffer('freqs', freqs)

    def forward(self, t):
        # t: (B, 1) -> t: (b, dim)
        arg = t * self.freqs[None, :]
        return torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)

class VelocityField(nn.Module):
    """MLP v_theta(q_t, t, x): concat conditioning, SiLU activation"""

    def __init__(self, cfg: FMConfig):
        super().__init__()

        robot = cfg.load_robot
        self.time_emb = TimeEmbedding(cfg.time_emb_dim)

        in_dim = robot.q_dim + cfg.time_emb_dim + robot.x_dim

        layers = [nn.Linear(in_dim, cfg.hidden_dim), nn.SiLU()]
        for _ in range(cfg.n_layers - 1):
            layers += [nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.SiLU()]
        layers += [nn.Linear(cfg.hidden_dim, robot.q_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, q_t, t, x):
        h = torch.cat([q_t, self.time_emb(t), x], dim=-1)
        return self.net(h)

class FlowMatching:
    """CFM loss + Euler ODE sampling + exact log-likelihood"""

    def __init__(self, cfg: FMConfig):
        self.cfg = cfg
        self.model = VelocityField(cfg).to(cfg.device)

    def loss(self, q1, x):
        """ q0 ~ N(0, I) -> P(q1|x)"""
        b = q1.shape[0] # batch
        t = torch.rand(b, 1, device=q1.device)
        q0 = torch.randn_like(q1)
        q_t = (1.0 - t) * q0 + t * q1
        # print(f'x dim: {x.shape}')
        v = self.model(q_t, t, x)
        return((v - (q1 - q0)) ** 2).mean()

    @torch.no_grad()
    def sample(self, x, n_samples: int, n_steps: int | None = None):
        """sample q ~ P(q|x). x: (xdim,) or (B, xdim). Return unnormalised q"""
        n_steps = n_steps or self.cfg.n_ode_steps

        robot = self.cfg.load_robot


        self.model.eval()

        x = torch.as_tensor(x, dtype=torch.float32, device=self.cfg.device)
        

        if x.dim() == 1:
            # add batch dim for neural network forward
            x = x[None, :]

        # just scale the position
        x = x.clone()
        x[:, :3] = x[:, :3] / robot.x_max
        x = x.repeat_interleave(n_samples, dim=0) # keep it same for each q sample

        q = torch.randn(x.shape[0], robot.q_dim, device=self.cfg.device)
        dt = 1.0 / n_steps

        # ODE
        for k in range(n_steps):
            t = torch.full((q.shape[0], 1), k * dt, device=self.cfg.device)
            q = q + dt * self.model(q, t, x)
        
        q_norm = q.clamp(-1.0, 1.0)
        # unnormalised
        q_mid = torch.as_tensor((robot.q_min + robot.q_max) / 2, dtype=torch.float32, device = self.cfg.device) # (N,)
        q_half = torch.as_tensor((robot.q_max - robot.q_min) / 2, dtype=torch.float32, device = self.cfg.device) # (N,)     
        raw_q = q_mid + q_half * q_norm

        return raw_q.cpu().numpy()

    def _velocity_divergence(self, q_t, t, x, retain: bool):
        q_t = q_t.detach().requires_grad_(True)
        v = self.model(q_t, t, x)
        div = torch.zeros(q_t.shape[0], device=q_t.device)

        for i in range(self.cfg.q_dim):
            keep = retain or (i < self.cfg.q_dim - 1)
            grad = torch.autograd.grad(v[:, i].sum(), q_t, retain_graph=keep)[0]
            div = div + grad[:, i]
        
        return v.detach(), div.detach()


    def save(self):
        self.cfg.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.cfg.ckpt_path)

    def load(self):
        state = torch.load(self.cfg.ckpt_path, map_location=self.cfg.device)
        self.model.load_state_dict(state)

if __name__ == "__main__":
    cfg = FMConfig()
    train_set, test_set = load_data(cfg)
    print(f"train size: {len(train_set)} | test size: {len(test_set)}")
