"""End-to-end benchmark: QI tensor-network compression vs INT8 baseline.

Runs on a proxy VLAM (transformer-style MLP tower) so the full pipeline is
verifiable on CPU in a clean environment; the same code path applies to
LLaVA-1.5-7B / pi-zero checkpoints by swapping the model constructor (see
docs/USAGE.md section 4). All results reported as mean +/- std across
`n_seeds` independent runs. Emits results.json consumed by
scripts/generate_report.py.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from pathlib import Path
from typing import Callable, Dict, List

import torch
import torch.nn as nn

from src.compression.tensor_network import compress_model
from src.compression.quantization import quantize_model_int8
from src.inference.model_server import LatencyProfiler
from src.training.vlam_trainer import (
    QuantumInspiredRegularization,
    VLAMTrainer,
)


def make_proxy_vlam(dim: int = 256, depth: int = 4, out_dim: int = 8, seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    layers: List[nn.Module] = []
    for _ in range(depth):
        layers += [nn.Linear(dim, dim), nn.GELU()]
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


def make_proxy_vlam_ttns(dim: int = 256, depth: int = 4, out_dim: int = 8,
                         bond_dimension: int = 24, seed: int = 0) -> nn.Module:
    """Same tower but hidden Linear layers are TTNS-factorized from init."""
    from src.compression.tensor_network import TTNSLinear
    torch.manual_seed(seed)
    layers: List[nn.Module] = []
    for _ in range(depth):
        layers += [TTNSLinear(dim, dim, bond_dimension=bond_dimension), nn.GELU()]
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


def model_linear_flops(model: nn.Module) -> int:
    """Forward multiply-add FLOPs of all linear-type layers (per sample)."""
    from src.compression.tensor_network import MPSLinear, TTNSLinear
    total = 0
    for m in model.modules():
        if isinstance(m, TTNSLinear):
            chi = m.root.shape[0]
            total += 2 * (chi * m.in_features + chi * chi + chi * m.out_features)
        elif isinstance(m, MPSLinear):
            total += 2 * sum(p.numel() for p in m.cores)  # contraction cost bound
        elif isinstance(m, nn.Linear):
            total += 2 * m.in_features * m.out_features
    return total


def make_task(dim: int = 256, out_dim: int = 8, seed: int = 0) -> Callable[[], tuple]:
    g = torch.Generator().manual_seed(seed)
    # low-rank ground-truth mapping -> realistic compressible structure
    A = torch.randn(dim, 16, generator=g)
    B = torch.randn(16, out_dim, generator=g)

    def batch_fn(batch: int = 64):
        x = torch.randn(batch, dim, generator=g)
        y = torch.tanh(x @ A @ B / dim**0.5)
        return x, y

    return batch_fn


def task_accuracy(model: nn.Module, batch_fn: Callable, n_batches: int = 10) -> float:
    """1 - normalized MSE, clipped to [0, 1]."""
    model.eval()
    errs, vars_ = [], []
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = batch_fn()
            errs.append(nn.functional.mse_loss(model(x), y).item())
            vars_.append(y.var().item())
    nmse = sum(errs) / max(sum(vars_), 1e-9)
    return max(0.0, 1.0 - nmse)


def _mean_std(vals: List[float]) -> Dict[str, float]:
    return {
        "mean": statistics.mean(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "runs": vals,
    }


def run_benchmark(n_seeds: int = 3, bond_dimension: int = 24, dim: int = 256,
                  train_steps: int = 300, out_path: str = "results.json") -> Dict:
    results: Dict = {
        "config": {"n_seeds": n_seeds, "bond_dimension": bond_dimension,
                   "hidden_dim": dim, "train_steps": train_steps},
        "compression": {}, "training_efficiency": {}, "latency": {}, "ablation": {},
    }
    acc_base, acc_mps, acc_int8 = [], [], []
    ratio_mps, ratio_int8 = [], []
    steps_base, steps_qi, wall_base, wall_qi = [], [], [], []
    flops_base, flops_qi = [], []

    for seed in range(n_seeds):
        torch.manual_seed(seed)
        batch_fn = make_task(dim=dim, seed=seed)

        # --- train a baseline model to convergence-ish state
        model = make_proxy_vlam(dim=dim, seed=seed)
        trainer = VLAMTrainer(model, max_steps=train_steps, seed=seed)
        r_base = trainer.train(batch_fn)
        steps_base.append(r_base.steps_to_convergence)
        wall_base.append(r_base.wall_clock_seconds)
        acc_base.append(task_accuracy(model, batch_fn))

        # --- QI training efficiency: TTNS-factorized parameterization
        # (forward contracts factors directly -> per-step FLOPs reduced by
        #  1 - [chi(in+out)+chi^2] / (in*out) per compressed layer)
        model_qi = make_proxy_vlam_ttns(dim=dim, bond_dimension=bond_dimension,
                                        seed=seed)
        trainer_qi = VLAMTrainer(model_qi, max_steps=train_steps, seed=seed)
        r_qi = trainer_qi.train(batch_fn)
        steps_qi.append(r_qi.steps_to_convergence)
        wall_qi.append(r_qi.wall_clock_seconds)
        flops_base.append(model_linear_flops(model))
        flops_qi.append(model_linear_flops(model_qi))

        # --- MPS compression of the trained baseline + healing fine-tune
        # (post-compression retraining of tensor cores, per CompactifAI)
        mps_model, report = compress_model(copy.deepcopy(model),
                                           bond_dimension=bond_dimension)
        VLAMTrainer(mps_model, max_steps=train_steps // 2, lr=1e-3,
                    seed=seed).train(batch_fn)
        acc_mps.append(task_accuracy(mps_model, batch_fn))
        ratio_mps.append(report.overall_ratio)

        # --- INT8 baseline
        int8_model = quantize_model_int8(copy.deepcopy(model))
        acc_int8.append(task_accuracy(int8_model, batch_fn))
        # int8: 4x storage reduction on quantized layers (fp32 -> int8)
        ratio_int8.append(4.0)

    results["compression"] = {
        "accuracy_baseline_fp32": _mean_std(acc_base),
        "accuracy_mps": _mean_std(acc_mps),
        "accuracy_int8": _mean_std(acc_int8),
        "param_ratio_mps": _mean_std(ratio_mps),
        "storage_ratio_int8": _mean_std(ratio_int8),
        "accuracy_drop_mps_pct": _mean_std(
            [100 * (b - m) / max(b, 1e-9) for b, m in zip(acc_base, acc_mps)]),
    }
    results["training_efficiency"] = {
        "steps_baseline": _mean_std([float(s) for s in steps_base]),
        "steps_qi": _mean_std([float(s) for s in steps_qi]),
        "steps_reduction_pct": _mean_std(
            [100 * (b - q) / max(b, 1) for b, q in zip(steps_base, steps_qi)]),
        "wall_clock_baseline_s": _mean_std(wall_base),
        "wall_clock_qi_s": _mean_std(wall_qi),
        "flops_per_sample_baseline": _mean_std([float(f) for f in flops_base]),
        "flops_per_sample_qi": _mean_std([float(f) for f in flops_qi]),
        "flops_per_step_reduction_pct": _mean_std(
            [100 * (b - q) / b for b, q in zip(flops_base, flops_qi)]),
        # Honest net metric: total training FLOPs = per-step FLOPs x steps.
        # TTNS may need MORE steps to converge; the net accounts for both.
        "total_training_flops_reduction_pct": _mean_std(
            [100 * (fb * sb - fq * sq) / (fb * sb)
             for fb, sb, fq, sq in zip(flops_base, steps_base, flops_qi, steps_qi)]),
    }

    # --- latency profile (compressed model)
    prof = LatencyProfiler(budget_ms=100.0)
    latency = prof.profile_model(mps_model, (1, dim), n_runs=100)
    results["latency"] = latency

    # --- ablation: isolate QI component (see ablation.py for full grid)
    results["ablation"]["note"] = (
        "MPS vs no-compression vs INT8 above already isolates the QI component; "
        "run src/evaluation/ablation.py for the bond-dimension Pareto sweep.")

    Path(out_path).write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--bond-dimension", type=int, default=24)
    p.add_argument("--out", default="results.json")
    a = p.parse_args()
    t0 = time.time()
    res = run_benchmark(n_seeds=a.seeds, bond_dimension=a.bond_dimension, out_path=a.out)
    print(json.dumps({k: v for k, v in res.items() if k != "config"}, indent=2))
    print(f"Done in {time.time()-t0:.1f}s -> {a.out}")
