"""Model zoo: train and compare policy architectures on the SAME simulated
pick-and-place dataset, all evaluated closed-loop in MuJoCo physics.

    python -m src.sim.model_zoo --seeds 3            # -> results/model_zoo.json

Every model maps the 17 policy features (src/sim/features.py) to 4 joint
velocities + a gripper command; only the architecture / training objective
changes:

  mlp            3x256 GELU MLP (the baseline used by the app)
  mlp_small      3x96 MLP — parameter-matched to the compressed models
  ttns           mlp, TTNS chi=16 compression + healing (quantum-inspired)
  mps            mlp, MPS chi=16 compression + healing (quantum-inspired)
  int8           mlp, post-training per-channel INT8 (classical compression)
  resmlp         residual MLP with LayerNorm (3 blocks x 256)
  act_chunk      MLP predicting a 5-step action chunk + temporal ensembling (ACT-style)
  diffusion      conditional DDPM over the action (16 denoising steps, Diffusion-Policy-style)
  head_tanh      frozen mlp trunk -> 6-unit tanh bottleneck -> linear       (classical control)
  head_surrogate frozen mlp trunk -> 6-unit cosine surrogate of the PQC     (classical surrogate)
  head_pqc       frozen mlp trunk -> 6-qubit PennyLane PQC (2 layers)       (quantum, simulated)

Eval sets as in train_pickplace: training places with randomized dynamics,
held-out factory, out-of-range dynamics. Reported: success, collisions,
params, storage, single-thread CPU latency per control step, train time.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from src.robot import arm
from src.sim.features import PP_DIM, PP_OUT, pickplace_features
from src.sim.train_pickplace import HELDOUT_ENV, TRAIN_ENVS, load_dataset, loss_fn, make_policy

K_CHUNK = 5
T_DIFF = 16
MODELS = ("mlp", "mlp_small", "ttns", "mps", "int8", "resmlp", "act_chunk", "diffusion",
          "head_tanh", "head_surrogate", "head_pqc")


# ---------------------------------------------------------------- architectures
class ResMLP(nn.Module):
    def __init__(self, hidden: int = 256, blocks: int = 3) -> None:
        super().__init__()
        self.inp = nn.Linear(PP_DIM, hidden)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden),
                                                   nn.GELU(approximate="tanh"), nn.Linear(hidden, hidden))
                                     for _ in range(blocks)])
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, PP_OUT), nn.Tanh())

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = h + b(h)
        return self.out(h)


class ChunkMLP(nn.Module):
    def __init__(self, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(PP_DIM, hidden), nn.GELU(approximate="tanh"),
                                 nn.Linear(hidden, hidden), nn.GELU(approximate="tanh"),
                                 nn.Linear(hidden, hidden), nn.GELU(approximate="tanh"),
                                 nn.Linear(hidden, PP_OUT * K_CHUNK), nn.Tanh())

    def forward(self, x):
        return self.net(x).view(-1, K_CHUNK, PP_OUT)


def _betas(T: int = T_DIFF):
    s = 0.008
    t = torch.linspace(0, T, T + 1) / T
    ab = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = ab / ab[0]
    return torch.clamp(1 - ab[1:] / ab[:-1], 1e-4, 0.5)


class Denoiser(nn.Module):
    """eps_theta(a_t, t, obs) — conditional MLP denoiser (cosine schedule DDPM)."""

    def __init__(self, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(PP_OUT + PP_DIM + 16, hidden), nn.GELU(approximate="tanh"),
                                 nn.Linear(hidden, hidden), nn.GELU(approximate="tanh"),
                                 nn.Linear(hidden, hidden), nn.GELU(approximate="tanh"),
                                 nn.Linear(hidden, PP_OUT))
        b = _betas()
        self.register_buffer("betas", b)
        self.register_buffer("abar", torch.cumprod(1 - b, 0))

    @staticmethod
    def temb(t: torch.Tensor) -> torch.Tensor:
        f = torch.exp(torch.arange(8) * -math.log(1000.0) / 8)
        a = t.float()[:, None] * f[None]
        return torch.cat([torch.sin(a), torch.cos(a)], 1)

    def forward(self, a, t, x):
        return self.net(torch.cat([a, self.temb(t), x], 1))

    def loss(self, x, y, g):
        t = torch.randint(0, T_DIFF, (x.shape[0],), generator=g)
        eps = torch.randn(y.shape, generator=g)
        ab = self.abar[t][:, None]
        return ((self.forward(ab.sqrt() * y + (1 - ab).sqrt() * eps, t, x) - eps) ** 2).mean()

    @torch.no_grad()
    def sample(self, x, g):
        a = torch.randn((x.shape[0], PP_OUT), generator=g)
        for t in reversed(range(T_DIFF)):
            tt = torch.full((x.shape[0],), t, dtype=torch.long)
            eps = self.forward(a, tt, x)
            b, ab = self.betas[t], self.abar[t]
            a = (a - b / (1 - ab).sqrt() * eps) / (1 - b).sqrt()
            if t > 0:
                a = a + b.sqrt() * torch.randn(a.shape, generator=g)
        return a.clamp(-1, 1)


class HeadModel(nn.Module):
    """Frozen trunk (all but the last Linear+Tanh of a trained mlp) + bottleneck head."""

    def __init__(self, trunk: nn.Module, kind: str, q: int = 6) -> None:
        super().__init__()
        from src.training.rl_alignment import PQCPolicyHead
        self.trunk = trunk
        for p in self.trunk.parameters():
            p.requires_grad_(False)
        self.kind = kind
        if kind == "pqc":
            from src.training.rl_alignment import PennyLanePQCHead
            self.f = PennyLanePQCHead(256, q, 2)
        elif kind == "surrogate":
            self.f = PQCPolicyHead(256, q, 2)
        else:
            self.f = nn.Sequential(nn.Linear(256, q), nn.Tanh())
        self.out = nn.Sequential(nn.Linear(q, PP_OUT), nn.Tanh())

    def forward(self, x):
        return self.out(self.f(self.trunk(x)))


def trunk_of(mlp: nn.Sequential) -> nn.Sequential:
    return copy.deepcopy(nn.Sequential(*list(mlp)[:-2]))


def build(kind: str, cfg: Dict) -> nn.Module:
    """Rebuild an (untrained) module skeleton; weights come from a state_dict."""
    from src.compression.quantization import quantize_model_int8
    from src.compression.tensor_network import compress_model
    if kind in ("mlp", "int8", "ttns", "mps"):
        m = make_policy(256)
        if kind == "int8":
            return quantize_model_int8(m, min_features=64)
        if kind in ("ttns", "mps"):
            return compress_model(m, bond_dimension=16, method=kind, min_features=64)[0]
        return m
    if kind == "mlp_small":
        return make_policy(96)
    if kind == "resmlp":
        return ResMLP()
    if kind == "act_chunk":
        return ChunkMLP()
    if kind == "diffusion":
        return Denoiser()
    if kind.startswith("head_"):
        return HeadModel(trunk_of(make_policy(256)), kind[5:])
    raise ValueError(kind)


# ---------------------------------------------------------------- training
def chunk_targets(Y: np.ndarray, ep: np.ndarray) -> np.ndarray:
    out = np.repeat(Y[:, None, :], K_CHUNK, axis=1).copy()
    n = len(Y)
    for k in range(1, K_CHUNK):
        idx = np.minimum(np.arange(n) + k, n - 1)
        same = ep[idx] == ep
        out[:, k] = np.where(same[:, None], Y[idx], out[:, k - 1])
    return out


def train_model(kind: str, Xt, Yt, ept, seed: int, steps: int, base: Optional[nn.Module] = None,
                heal_steps: int = 2000, head_steps: int = 800, head_frames: int = 20000) -> nn.Module:
    from src.compression.quantization import quantize_model_int8
    from src.compression.tensor_network import compress_model
    from src.sim.train_pickplace import fit
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    if kind in ("ttns", "mps"):
        m, _ = compress_model(copy.deepcopy(base), bond_dimension=16, method=kind, min_features=64)
        fit(m, Xt, Yt, heal_steps, 5e-4, seed + 100)
        return m
    if kind == "int8":
        return quantize_model_int8(copy.deepcopy(base), min_features=64)
    if kind.startswith("head_"):
        m = HeadModel(trunk_of(base), kind[5:])
        idx = torch.randperm(Xt.shape[0], generator=g)[:head_frames]
        Xs, Ys = Xt[idx], Yt[idx]
        opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=3e-3)
        for _ in range(head_steps):
            i = torch.randint(0, Xs.shape[0], (256,), generator=g)
            loss = loss_fn(m(Xs[i]), Ys[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
        return m
    m = build(kind, {})
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3 if kind != "diffusion" else 1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    Yc = torch.from_numpy(chunk_targets(Yt.numpy(), ept)) if kind == "act_chunk" else None
    for _ in range(steps):
        i = torch.randint(0, Xt.shape[0], (2048,), generator=g)
        if kind == "act_chunk":
            p = m(Xt[i])
            loss = ((p[..., :4] - Yc[i][..., :4]) ** 2).mean() + 0.5 * ((p[..., 4] - Yc[i][..., 4]) ** 2).mean()
        elif kind == "diffusion":
            loss = m.loss(Xt[i], Yt[i], g)
        else:
            loss = loss_fn(m(Xt[i]), Yt[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
    return m


# ---------------------------------------------------------------- inference wrapper
class ZooPolicy:
    """Stateful per-episode policy: feature vector -> 5 outputs in [-1, 1]."""

    def __init__(self, kind: str, model: nn.Module, seed: int = 0) -> None:
        self.kind, self.m = kind, model.eval()
        self.g = torch.Generator().manual_seed(seed)
        self.reset()

    def reset(self) -> None:
        self.buf: List[np.ndarray] = []   # act_chunk: past chunk predictions

    @torch.no_grad()
    def __call__(self, x: np.ndarray) -> np.ndarray:
        xt = torch.from_numpy(np.asarray(x, np.float32))[None]
        if self.kind == "diffusion":
            return self.m.sample(xt, self.g)[0].numpy()
        if self.kind == "act_chunk":
            self.buf.append(self.m(xt)[0].numpy())
            self.buf = self.buf[-K_CHUNK:]
            # temporal ensembling (ACT): prediction for "now" from each past chunk, w_i = exp(-0.1 i)
            preds = [c[len(self.buf) - 1 - j] for j, c in enumerate(self.buf)]
            w = np.exp(-0.1 * np.arange(len(preds)))[::-1]
            return (np.asarray(preds) * w[:, None]).sum(0) / w.sum()
        return self.m(xt)[0].numpy()


def zoo_episode(kind: str, state: bytes, env_id: str, seed: int, randomize: bool, dynamics, max_steps: int = 500):
    from src.sim.envs import sample_episode
    from src.sim.expert import MAX_WINDUP, roll_for_yaw
    from src.sim.physics import DT, ArmSim
    torch.set_num_threads(1)
    m = build(kind, {})
    m.load_state_dict(torch.load(io.BytesIO(state), weights_only=False))
    pol = ZooPolicy(kind, m, seed)
    ep = sample_episode(env_id, seed, randomize=randomize)
    sim = ArmSim(ep, dynamics=dynamics)
    q_cmd, roll, stable = arm.Q_HOME.copy(), 0.0, 0
    for _ in range(max_steps):
        o = sim.observe()
        q = o["q"]
        y = pol(pickplace_features(q, o["obj"], o["goal"], ep.goal_kind == "cube"))
        qd = np.clip(y[:4], -1, 1) * arm.QD_MAX[:4]
        grip = 1.0 if y[4] > 0 else 0.0
        if grip > 0.5 and o["obj"][2] < sim.cube_half + 0.01:
            roll = roll_for_yaw(q[0], o["obj"][3])
        q_cmd[:4] = np.clip(q_cmd[:4] + qd * DT, q[:4] - MAX_WINDUP, q[:4] + MAX_WINDUP)
        q_cmd[4] += np.clip(roll - q_cmd[4], -arm.QD_MAX[4] * DT, arm.QD_MAX[4] * DT)
        q_cmd[5] = grip
        q_cmd = arm.clamp_q(q_cmd)
        sim.step(q_cmd)
        stable = stable + 1 if sim.success() else 0
        if stable >= 10:
            return {"success": True, "t": round(sim.t, 2), "collision": sim.collision_steps > 0}
    return {"success": False, "t": round(sim.t, 2), "collision": sim.collision_steps > 0}


def _job(a):
    return zoo_episode(*a)


def evaluate(kind: str, model: nn.Module, n_train=40, n_heldout=20, n_hard=20, seed0=60_000, workers=0) -> Dict:
    from src.sim.envs import hard_dynamics, load_env
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    st = buf.getvalue()
    jobs, tags = [], []
    for k in range(n_train):
        jobs.append((kind, st, TRAIN_ENVS[k % 4], seed0 + k, True, None)); tags.append("train_envs")
    for k in range(n_heldout):
        jobs.append((kind, st, HELDOUT_ENV, seed0 + 1000 + k, True, None)); tags.append("heldout_env")
    for k in range(n_hard):
        e = TRAIN_ENVS[k % 4]
        jobs.append((kind, st, e, seed0 + 2000 + k, False, hard_dynamics(load_env(e)))); tags.append("hard_dynamics")
    with ProcessPoolExecutor(max_workers=workers or max(1, (os.cpu_count() or 2) - 1)) as pool:
        res = list(pool.map(_job, jobs, chunksize=2))
    out = {}
    for tag in ("train_envs", "heldout_env", "hard_dynamics"):
        r = [x for x, t in zip(res, tags) if t == tag]
        if r:
            out[tag] = {"success_pct": 100.0 * np.mean([x["success"] for x in r]), "n": len(r),
                        "collision_pct": 100.0 * np.mean([x["collision"] for x in r])}
    return out


def model_stats(kind: str, m: nn.Module, x0: np.ndarray) -> Dict:
    n_params = sum(p.numel() for p in m.parameters()) + sum(b.numel() for n, b in m.named_buffers()
                                                            if "weight_int8" in n or n.endswith("scale"))
    storage = sum(p.numel() * 4 for p in m.parameters() if p.requires_grad or kind.startswith("head_"))
    if kind == "int8":
        storage = sum(b.numel() * (1 if b.dtype == torch.int8 else 4) for b in m.buffers()) + \
            sum(p.numel() * 4 for p in m.parameters())
    torch.set_num_threads(1)
    pol = ZooPolicy(kind, m)
    for _ in range(5):
        pol(x0)
    n = 30 if kind in ("diffusion", "head_pqc") else 300
    t0 = time.perf_counter()
    for _ in range(n):
        pol(x0)
    return {"params": int(n_params), "storage_kb": storage / 1024, "latency_ms": 1000 * (time.perf_counter() - t0) / n}


def run(data: str = "data/qivla_sim", seeds: int = 3, steps: int = 8000, models=MODELS,
        n_train=40, n_heldout=20, n_hard=20, out_json: str = "results/model_zoo.json", workers: int = 0) -> Dict:
    from src.training.rl_alignment import _PENNYLANE
    t_all = time.time()
    X, Y, ep = load_dataset(data)
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    models = [k for k in models if k != "head_pqc" or _PENNYLANE]
    runs: Dict[str, List[Dict]] = {}
    for s in range(seeds):
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
        t0 = time.time()
        base = train_model("mlp", Xt, Yt, ep, s, steps)
        t_base = time.time() - t0
        for kind in models:
            t0 = time.time()
            m = base if kind == "mlp" else train_model(kind, Xt, Yt, ep, s, steps, base=base)
            t_train = t_base if kind == "mlp" else time.time() - t0
            stats = model_stats(kind, m, X[0])
            torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
            r = {"seed": s, "train_s": t_train, **stats,
                 **evaluate(kind, m, n_train, n_heldout, n_hard, workers=workers)}
            runs.setdefault(kind, []).append(r)
            print(f"[seed {s}] {kind:15s} params={stats['params']:>7,} lat={stats['latency_ms']:.2f}ms "
                  + " ".join(f"{t}={r[t]['success_pct']:.0f}%" for t in ("train_envs", "heldout_env", "hard_dynamics")),
                  flush=True)
            Path(out_json).parent.mkdir(parents=True, exist_ok=True)
            Path(out_json).write_text(json.dumps({"partial": True, "runs": runs}, indent=1), encoding="utf-8")
    summary = {}
    for kind, rs in runs.items():
        e = {"params": rs[0]["params"], "storage_kb": rs[0]["storage_kb"],
             "latency_ms": float(np.mean([r["latency_ms"] for r in rs])),
             "train_s": float(np.mean([r["train_s"] for r in rs]))}
        for t in ("train_envs", "heldout_env", "hard_dynamics"):
            v = [r[t]["success_pct"] for r in rs]
            e[t] = {"success_mean": float(np.mean(v)), "success_std": float(np.std(v)), "per_seed": v,
                    "collision_mean": float(np.mean([r[t]["collision_pct"] for r in rs])), "n_per_seed": rs[0][t]["n"]}
        summary[kind] = e
    result = {"task": "pick-and-place, MuJoCo friction grasp, 3-5 objects (cubes/cylinders), 5 places",
              "dataset": {"path": data, "frames": int(len(X)), "episodes": int(len(np.unique(ep)))},
              "config": {"seeds": seeds, "steps": steps, "k_chunk": K_CHUNK, "diffusion_steps": T_DIFF,
                         "eval": {"train_envs": n_train, "heldout_env": n_heldout, "hard_dynamics": n_hard}},
              "summary": summary, "runs": runs,
              "notes": ("All models see the same 17 features and are run by the same closed-loop controller "
                        "wrapper. head_* models reuse each seed's trained mlp trunk (frozen). PQC is simulated "
                        "noiselessly with PennyLane (default.qubit); its latency is the simulator's, not a QPU's. "
                        "Simulation only; no physical robot."),
              "wall_s": round(time.time() - t_all, 1)}
    Path(out_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/qivla_sim")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--models", default=",".join(MODELS))
    a = p.parse_args()
    r = run(a.data, a.seeds, a.steps, a.models.split(","))
    print(json.dumps(r["summary"], indent=1))


if __name__ == "__main__":
    main()
