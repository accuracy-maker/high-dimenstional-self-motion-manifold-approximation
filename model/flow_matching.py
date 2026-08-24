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

ROOT_PATH = Path(__file__).resolve().parents[1]
# print(f"root path: {ROOT_PATH}")

@dataclass
class FMConfig:
    # 3 joint angles directly represented in normalized coordinates
    q_dim: int = 3

    # planar workspace position (x, y)
    x_dim: int = 2

    Q_MIN: float = -math.pi / 3
    Q_MAX: float = math.pi / 3

    # maximum reach of 3R planar manipulator
    X_MAX: float = 3.0

    hidden_dim: int = 256
    n_layers: int = 4
    time_emb_dim: int = 64

    lr: float = 1e-3
    weight_decay: float = 1e-5

    batch_size: int = 512
    n_epochs: int = 500

    n_ode_steps: int = 100

    test_size: float = 0.1

    dataset_path: Path = ROOT_PATH / "assets/3Rplanar/planar3r.npz"
    ckpt_path: Path = ROOT_PATH / "training/checkpoints/fm_3r.pt"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    seed: int = 42

# data preprocessing

def load_data(cfg: FMConfig):
    data = np.load(cfg.dataset_path)

    # normalise angles
    raw_qs = torch.from_numpy(data['qs']).float()

    qs_mid = (cfg.Q_MIN + cfg.Q_MAX) / 2
    qs_half = (cfg.Q_MAX - cfg.Q_MIN) / 2
    norm_qs = (raw_qs - qs_mid) / qs_half # norm_qs \in [-1, 1]
    
    # normalise position 
    ps = torch.from_numpy(data['ps']).float() / cfg.X_MAX

    # reproducible seed
    n = norm_qs.shape[0] # how many rows
    g = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(n, generator=g)

    # split dataset
    n_test = int(n * cfg.test_size)
    train_idx, test_idx = perm[n_test:], perm[:n_test]

    train_set = TensorDataset(norm_qs[train_idx], ps[train_idx])
    test_set = TensorDataset(norm_qs[test_idx], ps[test_idx])

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

        self.time_emb = TimeEmbedding(cfg.time_emb_dim)

        in_dim = cfg.q_dim + cfg.time_emb_dim + cfg.x_dim

        layers = [nn.Linear(in_dim, cfg.hidden_dim), nn.SiLU()]
        for _ in range(cfg.n_layers - 1):
            layers += [nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.SiLU()]
        layers += [nn.Linear(cfg.hidden_dim, cfg.q_dim)]
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
        v = self.model(q_t, t, x)
        return((v - (q1 - q0)) ** 2).mean()

    @torch.no_grad()
    def sample(self, x, n_samples: int, n_steps: int | None = None):
        """sample q ~ P(q|x). x: (xdim,) or (B, xdim). Return unnormalised q"""
        n_steps = n_steps or self.cfg.n_ode_steps

        q_mid = (self.cfg.Q_MIN + self.cfg.Q_MAX) / 2
        q_half = (self.cfg.Q_MAX - self.cfg.Q_MIN) / 2       

        self.model.eval()

        x = torch.as_tensor(x, dtype=torch.float32, device=self.cfg.device)
        if x.dim() == 1:
            # add batch dim for neural network forward
            x = x[None, :]
        x = (x / self.cfg.X_MAX).repeat_interleave(n_samples, dim=0) # keep it same for each q sample

        q = torch.randn(x.shape[0], self.cfg.q_dim, device=self.cfg.device)
        dt = 1.0 / n_steps

        # ODE
        for k in range(n_steps):
            t = torch.full((q.shape[0], 1), k * dt, device=self.cfg.device)
            q = q + dt * self.model(q, t, x)
        
        q_norm = q.clamp(-1.0, 1.0)
        # unnormalised
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
