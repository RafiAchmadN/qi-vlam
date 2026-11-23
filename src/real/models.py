"""Vision-language-conditioned (camera + joints + task) policies for real-robot data.

All models predict an ACTION CHUNK: the next K joint-position targets
(normalized per dataset), as in ACT / SmolVLA-style low-level control.

  state_mlp    joints + task only (no camera)                 — shows the camera is needed
  cnn_mlp      conv encoder + spatial-softmax keypoints + MLP  — the main vision policy
  act_tf       conv feature tokens + joint/task token -> 2-layer Transformer encoder (ACT-lite)
  *_ttns/_mps  cnn_mlp with the MLP layers TTNS/MPS-compressed + healed (quantum-inspired)
  *_int8       cnn_mlp with post-training INT8 linear layers
  head_*       frozen cnn_mlp trunk -> 6-unit tanh | cosine-surrogate | PennyLane PQC bottleneck
"""

from __future__ import annotations

import copy
import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialSoftmax(nn.Module):
    """Expected (x, y) image coordinates of each feature map — standard for visuomotor policies."""

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        b, c, h, w = f.shape
        p = F.softmax(f.reshape(b, c, h * w), dim=-1).reshape(b, c, h, w)
        ys = torch.linspace(-1, 1, h, device=f.device)
        xs = torch.linspace(-1, 1, w, device=f.device)
        return torch.cat([(p.sum(3) * ys).sum(2), (p.sum(2) * xs).sum(2)], 1)


class ConvEncoder(nn.Module):
    def __init__(self, ch: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2), nn.GELU(),
            nn.Conv2d(32, ch, 3, 2, 1), nn.GELU(),
            nn.Conv2d(ch, ch, 3, 1, 1), nn.GELU(),
            nn.Conv2d(ch, 128, 3, 2, 1), nn.GELU())
        self.ss = SpatialSoftmax()

    def forward(self, img: torch.Tensor, tokens: bool = False):
        f = self.net(img)
        if tokens:
            return f
        return torch.cat([self.ss(f), f.mean((2, 3))], 1)   # 256 keypoint coords + 128 pooled = 384


def mlp(din: int, dout: int, hidden: int = 256, depth: int = 3) -> nn.Sequential:
    layers = [nn.Linear(din, hidden), nn.GELU(approximate="tanh")]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), nn.GELU(approximate="tanh")]
    layers += [nn.Linear(hidden, dout)]
    return nn.Sequential(*layers)


class Policy(nn.Module):
    """kind in {state_mlp, cnn_mlp, act_tf}. Inputs: img (B,3,S,S) in [0,1], state (B,6) normalized,
    task (B,T) one-hot. Output: (B, K, 6) normalized action chunk."""

    def __init__(self, kind: str, n_tasks: int, k: int = 10, hidden: int = 256) -> None:
        super().__init__()
        self.kind, self.k, self.n_tasks = kind, k, n_tasks
        if kind == "state_mlp":
            self.head = mlp(6 + n_tasks, k * 6, hidden)
        elif kind == "cnn_mlp":
            self.enc = ConvEncoder()
            self.head = mlp(384 + 6 + n_tasks, k * 6, hidden)
        elif kind == "act_tf":
            self.enc = ConvEncoder()
            d = 128
            self.pos = nn.Parameter(torch.randn(1, 64 + 1, d) * 0.02)
            self.inp = nn.Linear(6 + n_tasks, d)
            layer = nn.TransformerEncoderLayer(d, 4, 256, dropout=0.0, batch_first=True, norm_first=True)
            self.tf = nn.TransformerEncoder(layer, 2)
            self.head = mlp(d, k * 6, hidden, depth=2)
        else:
            raise ValueError(kind)

    def features(self, img, state, task):
        if self.kind == "state_mlp":
            return torch.cat([state, task], 1)
        if self.kind == "cnn_mlp":
            return torch.cat([self.enc(img), state, task], 1)
        f = self.enc(img, tokens=True)                   # (B,128,8,8)
        tok = f.flatten(2).transpose(1, 2)               # (B,64,128)
        x = torch.cat([self.inp(torch.cat([state, task], 1))[:, None], tok], 1) + self.pos[:, :tok.shape[1] + 1]
        return self.tf(x)[:, 0]

    def forward(self, img, state, task):
        return self.head(self.features(img, state, task)).view(-1, self.k, 6)


class HeadPolicy(nn.Module):
    """Frozen trained cnn_mlp up to its last hidden layer + bottleneck head (tanh/surrogate/PQC)."""

    def __init__(self, base: Policy, kind: str, q: int = 6) -> None:
        super().__init__()
        from src.training.rl_alignment import PQCPolicyHead
        self.base = copy.deepcopy(base)
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.trunk = nn.Sequential(*list(self.base.head)[:-1])
        if kind == "pqc":
            from src.training.rl_alignment import PennyLanePQCHead
            self.f = PennyLanePQCHead(self.trunk[-2].out_features, q, 2)
        elif kind == "surrogate":
            self.f = PQCPolicyHead(self.trunk[-2].out_features, q, 2)
        else:
            self.f = nn.Sequential(nn.Linear(self.trunk[-2].out_features, q), nn.Tanh())
        self.out = nn.Linear(q, base.k * 6)
        self.k = base.k

    def forward(self, img, state, task):
        with torch.no_grad():
            h = self.trunk(self.base.features(img, state, task))
        return self.out(self.f(h)).view(-1, self.k, 6)


def compress_head(policy: Policy, method: str, chi: int) -> Policy:
    from src.compression.tensor_network import compress_model
    p = copy.deepcopy(policy)
    p.head, _ = compress_model(p.head, bond_dimension=chi, method=method, min_features=64)
    return p


def int8_head(policy: Policy) -> Policy:
    from src.compression.quantization import quantize_model_int8
    p = copy.deepcopy(policy)
    p.head = quantize_model_int8(p.head, min_features=64)
    return p


def count(m: nn.Module) -> Dict[str, float]:
    params = sum(p.numel() for p in m.parameters())
    int8 = sum(b.numel() for n, b in m.named_buffers() if "weight_int8" in n)
    fp_bytes = sum(p.numel() * 4 for p in m.parameters())
    scale = sum(b.numel() * 4 for n, b in m.named_buffers() if n.endswith("scale"))
    return {"params": int(params + int8), "storage_kb": (fp_bytes + int8 + scale) / 1024}


def augment(img: torch.Tensor, g: torch.Generator, pad: int = 4) -> torch.Tensor:
    """Random-shift augmentation (DrQ): pad by replication and random crop back."""
    b, c, h, w = img.shape
    x = F.pad(img, (pad, pad, pad, pad), mode="replicate")
    ox = torch.randint(0, 2 * pad + 1, (b,), generator=g)
    oy = torch.randint(0, 2 * pad + 1, (b,), generator=g)
    idx_y = (oy[:, None] + torch.arange(h)[None]).clamp(max=h + 2 * pad - 1)
    idx_x = (ox[:, None] + torch.arange(w)[None]).clamp(max=w + 2 * pad - 1)
    x = x[torch.arange(b)[:, None, None, None], torch.arange(c)[None, :, None, None],
          idx_y[:, None, :, None], idx_x[:, None, None, :]]
    return x


def temporal_ensemble(chunks, w: float = 0.1):
    """ACT temporal ensembling: average the predictions each past chunk makes for 'now'."""
    preds = [c[len(chunks) - 1 - j] for j, c in enumerate(chunks) if len(chunks) - 1 - j < len(c)]
    ws = torch.tensor([math.exp(-w * i) for i in range(len(preds))], dtype=torch.float32)  # oldest weighted most (ACT)
    return (torch.stack(preds) * ws[:, None]).sum(0) / ws.sum()
