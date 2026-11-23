"""Model comparison on the open LIBERO dataset (real robot demonstrations).

Task: language-conditioned behaviour cloning, offline. Input per frame is
[proprio state, causal state velocity, hashed instruction embedding]
(`src/vla_data/libero.py::build_features`); target is the recorded 7-dim
action (6 continuous delta-pose dims + binary gripper).

Compared, under identical data/splits/seeds:

  dense         FP32 MLP policy (baseline)
  healing_ctrl  dense + the same extra fine-tune steps the compressed models
                get (the fair control; extra steps help dense models too)
  int8          dense after per-channel INT8 PTQ (classical compression baseline)
  mps_chi*      dense -> MPS/tensor-train compression -> healing
  ttns_chi*     dense -> TTNS (truncated SVD, LoRA-style) -> healing

and, on a frozen dense trunk, three bottleneck heads with equal width
(Linear(256->q) -> f -> Linear(q->7)), isolating the quantum layer:

  head_pqc        f = PennyLane PQC (angle embedding + StronglyEntanglingLayers)
  head_surrogate  f = cosine surrogate, same parameter count, no entanglement
  head_tanh       f = tanh (classical)

Metrics on HELD-OUT EPISODES (split per episode, stratified per task):
normalized MSE of the 6 continuous dims (1.0 = predicting the train mean),
gripper accuracy, parameter count, single-sample CPU latency.

Scope, stated plainly: this is offline action prediction without camera
images, not closed-loop task success in the LIBERO simulator (which needs
MuJoCo/robosuite and is the metric published VLA papers report). Low MSE
is necessary, not sufficient, for task success.

    python -m src.evaluation.libero_benchmark --seeds 3
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from src.compression.quantization import quantize_model_int8
from src.compression.tensor_network import compress_model
from src.training.rl_alignment import PQCPolicyHead, _PENNYLANE
from src.vla_data.libero import SUITES, build_features, episode_split, load_libero

if _PENNYLANE:
    from src.training.rl_alignment import PennyLanePQCHead


class PolicyMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512, depth: int = 3, trunk_out: int = 256):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(in_dim, hidden), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, trunk_out), nn.GELU()]
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(trunk_out, 7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))


def bc_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mse = ((pred[:, :6] - y[:, :6]) ** 2).mean()
    bce = nn.functional.binary_cross_entropy_with_logits(pred[:, 6], y[:, 6])
    return mse + 0.5 * bce


def train(model: nn.Module, X: torch.Tensor, Y: torch.Tensor, steps: int, lr: float,
          seed: int, batch: int = 512, params=None) -> List[float]:
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(params if params is not None else model.parameters(),
                            lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    model.train()
    hist = []
    for s in range(steps):
        idx = torch.randint(0, X.shape[0], (batch,), generator=g)
        loss = bc_loss(model(X[idx]), Y[idx])
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if s % 100 == 0 or s == steps - 1:
            hist.append(round(float(loss), 5))
    return hist


@torch.no_grad()
def evaluate(model: nn.Module, X: torch.Tensor, Y: torch.Tensor) -> Dict[str, float]:
    model.eval()
    pred = torch.cat([model(X[i:i + 8192]) for i in range(0, X.shape[0], 8192)])
    nmse = float(((pred[:, :6] - Y[:, :6]) ** 2).mean())  # Y standardized -> 1.0 = mean pred
    grip = float(((pred[:, 6] > 0).float() == Y[:, 6]).float().mean())
    return {"nmse": nmse, "gripper_acc": grip}


def n_params(model: nn.Module) -> int:
    n = sum(p.numel() for p in model.parameters())
    n += sum(b.numel() for name, b in model.named_buffers() if "weight_int8" in name)
    return n


def storage_bytes(model: nn.Module) -> int:
    from src.compression.quantization import Int8Linear
    total = 0
    for m in model.modules():
        if isinstance(m, Int8Linear):
            total += m.effective_param_bytes()
        elif not list(m.children()):
            total += 4 * sum(p.numel() for p in m.parameters(recurse=False))
    return total


@torch.no_grad()
def latency_ms(model: nn.Module, in_dim: int, n: int = 500) -> float:
    """Median single-sample latency, single-threaded (robot-side CPU profile;
    multi-threaded timings at batch 1 were dominated by scheduler noise)."""
    model.eval()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        x = torch.randn(1, in_dim)
        for _ in range(50):
            model(x)
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            model(x)
            ts.append((time.perf_counter() - t0) * 1000)
    finally:
        torch.set_num_threads(threads)
    return float(np.median(ts))


class BottleneckHead(nn.Module):
    def __init__(self, kind: str, in_dim: int, q: int, n_layers: int = 2):
        super().__init__()
        self.kind = kind
        if kind == "pqc":
            self.f = PennyLanePQCHead(in_dim, q, n_layers)
        elif kind == "surrogate":
            self.f = PQCPolicyHead(in_dim, q, n_layers)
        else:
            self.f = nn.Sequential(nn.Linear(in_dim, q), nn.Tanh())
        self.out = nn.Linear(q, 7)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.out(self.f(h))


def _ms(vals: List[float]) -> Dict[str, float]:
    return {"mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0}


# (method, bond_dimension). On 512x512 layers MPS chi=32 and TTNS chi=64 give
# the SAME 3.74x per-layer ratio (equal-compression comparison isolating the
# tensor-network topology); the chi=16 pair is the aggressive ~14-15x regime.
DEFAULT_CONFIGS = (("mps", 32), ("ttns", 64), ("mps", 16), ("ttns", 16))


def run(n_seeds: int = 3, steps: int = 3000, heal_steps: int = 600,
        configs=DEFAULT_CONFIGS, head_qubits: int = 6,
        head_steps: int = 600, head_train_frames: int = 20000,
        out_path: str = "results/libero_benchmark.json") -> Dict:
    data = load_libero()
    feats = build_features(data)
    act = data.action.copy()
    act[:, 6] = (act[:, 6] > 0).astype(np.float32)  # gripper -> {0 open, 1 close}

    per_seed: Dict[str, List[Dict]] = {}
    per_suite: Dict[str, Dict[str, List[float]]] = {}
    curves = {}
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        split = episode_split(data, seed=seed)
        tr, te = split["train"], split["test"]
        mu, sd = feats[tr].mean(0), feats[tr].std(0) + 1e-6
        amu, asd = act[tr, :6].mean(0), act[tr, :6].std(0) + 1e-6

        def prep(mask):
            X = torch.from_numpy((feats[mask] - mu) / sd)
            Y = act[mask].copy()
            Y[:, :6] = (Y[:, :6] - amu) / asd
            return X, torch.from_numpy(Y)

        Xtr, Ytr = prep(tr)
        Xte, Yte = prep(te)
        in_dim = Xtr.shape[1]

        dense = PolicyMLP(in_dim)
        t0 = time.time()
        curves[f"dense_s{seed}"] = train(dense, Xtr, Ytr, steps, 1e-3, seed)
        train_s = time.time() - t0

        models: Dict[str, nn.Module] = {"dense": dense}
        ctrl = copy.deepcopy(dense)
        train(ctrl, Xtr, Ytr, heal_steps, 3e-4, seed + 100)
        models["healing_ctrl"] = ctrl
        models["int8"] = quantize_model_int8(copy.deepcopy(dense))
        ratios = {}
        for method, chi in configs:
            m, rep = compress_model(copy.deepcopy(dense), bond_dimension=chi, method=method)
            name = f"{method}_chi{chi}"
            models[f"{name}_noheal"] = copy.deepcopy(m)
            curves[f"{name}_heal_s{seed}"] = train(m, Xtr, Ytr, heal_steps, 3e-4, seed + 100)
            models[name] = m
            ratios[name] = rep.overall_ratio

        dense_params = n_params(dense)
        for name, m in models.items():
            r = evaluate(m, Xte, Yte)
            r.update({"params": n_params(m), "storage_bytes": storage_bytes(m),
                      "param_ratio_vs_dense": dense_params / n_params(m),
                      "latency_ms_p50": latency_ms(m, in_dim)})
            if name in ratios:
                r["compressed_layer_ratio"] = ratios[name]
            per_seed.setdefault(name, []).append(r)
            if name in ("dense", "healing_ctrl") or name in ratios:
                te_task = data.task[te]
                for suite, rng in SUITES.items():
                    mm = torch.from_numpy(np.isin(te_task, list(rng)))
                    rs = evaluate(m, Xte[mm], Yte[mm])
                    per_suite.setdefault(name, {}).setdefault(suite, []).append(rs["nmse"])

        # --- bottleneck heads on the frozen dense trunk (quantum-layer isolation)
        with torch.no_grad():
            dense.eval()
            g = torch.Generator().manual_seed(seed)
            sub = torch.randperm(Xtr.shape[0], generator=g)[:head_train_frames]
            Htr, Hte = dense.trunk(Xtr[sub]), dense.trunk(Xte)
        kinds = ["tanh", "surrogate"] + (["pqc"] if _PENNYLANE else [])
        for kind in kinds:
            torch.manual_seed(seed)
            head = BottleneckHead(kind, Htr.shape[1], head_qubits)
            t0 = time.time()
            train(head, Htr, Ytr[sub], head_steps, 5e-3, seed, batch=128)
            r = evaluate(head, Hte, Yte)
            r.update({"head_params": sum(p.numel() for p in head.parameters()),
                      "head_train_s": time.time() - t0})
            per_seed.setdefault(f"head_{kind}", []).append(r)
        print(f"seed {seed} done (dense train {train_s:.0f}s)", flush=True)

    summary = {}
    for name, runs in per_seed.items():
        summary[name] = {k: _ms([r[k] for r in runs]) for k in runs[0]}
    out = {
        "dataset": {"name": "lerobot/libero", "episodes": data.n_episodes,
                    "frames": int(len(data.state)), "tasks": len(data.tasks),
                    "split": "per-episode 80/10/10 stratified by task",
                    "features": "state(8) + causal velocity(8) + hashed-ngram language(64)"},
        "config": {"n_seeds": n_seeds, "steps": steps, "heal_steps": heal_steps,
                   "compression_configs": [list(c) for c in configs], "head_qubits": head_qubits,
                   "head_steps": head_steps, "head_train_frames": head_train_frames},
        "summary": summary,
        "per_suite_nmse": {n: {s: _ms(v) for s, v in d.items()} for n, d in per_suite.items()},
        "runs": per_seed,
        "loss_curves": curves,
        "scope_note": ("Offline held-out-episode action prediction on real LIBERO demos, "
                       "no images; not closed-loop simulator success rate."),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--heal-steps", type=int, default=600)
    p.add_argument("--head-steps", type=int, default=600)
    p.add_argument("--out", default="results/libero_benchmark.json")
    a = p.parse_args()
    t0 = time.time()
    res = run(n_seeds=a.seeds, steps=a.steps, heal_steps=a.heal_steps,
              head_steps=a.head_steps, out_path=a.out)
    for k, v in res["summary"].items():
        print(f"{k:22s} nmse={v['nmse']['mean']:.4f}±{v['nmse']['std']:.4f} "
              f"grip={v['gripper_acc']['mean']:.3f}")
    print(f"done in {time.time() - t0:.0f}s -> {a.out}")
