#!/usr/bin/env python3
"""Measured wall-clock forward-pass benchmark: dense vs MPS vs TTNS.

`src/evaluation/benchmark.py:model_linear_flops` reports *analytic* FLOPs
(multiply-adds), which is what the technical report's "-79.7%" claim is
based on. Analytic FLOPs are not the same as wall-clock speed: at small
dim, Python/PyTorch dispatch overhead per `einsum`/matmul call can dominate,
so a tensor-network layer with fewer FLOPs can still be *slower* in wall
clock than a single dense `nn.Linear` op. This script measures actual
wall-clock time (CPU) across a range of `dim` to find the crossover point
where contraction genuinely beats dense — CLAUDE.md roadmap item 3.

Run:
    PYTHONPATH=. python scripts/wallclock_bench.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

from src.compression.tensor_network import MPSLinear, TTNSLinear


def time_forward(model: nn.Module, x: torch.Tensor, n_runs: int, warmup: int) -> float:
    """Mean wall-clock forward-pass latency in ms."""
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            model(x)
        elapsed = time.perf_counter() - t0
    return 1000.0 * elapsed / n_runs


def run(dims: List[int], bond_dimension: int, batch: int, n_runs: int,
       warmup: int, out_path: str) -> Dict:
    results = []
    for dim in dims:
        x = torch.randn(batch, dim)
        dense = nn.Linear(dim, dim)
        mps = MPSLinear.from_linear(dense, bond_dimension=bond_dimension)
        ttns = TTNSLinear.from_linear(dense, bond_dimension=bond_dimension)

        dense_ms = time_forward(dense, x, n_runs, warmup)
        mps_ms = time_forward(mps, x, n_runs, warmup)
        ttns_ms = time_forward(ttns, x, n_runs, warmup)

        row = {
            "dim": dim,
            "bond_dimension": bond_dimension,
            "dense_ms": dense_ms,
            "mps_ms": mps_ms,
            "ttns_ms": ttns_ms,
            "mps_speedup_x": dense_ms / mps_ms,
            "ttns_speedup_x": dense_ms / ttns_ms,
            "mps_param_ratio": mps.get_compression_ratio(),
            "ttns_param_ratio": ttns.get_compression_ratio(),
        }
        results.append(row)
        print(f"dim={dim:5d}  dense={dense_ms:7.4f}ms  mps={mps_ms:7.4f}ms "
              f"({row['mps_speedup_x']:.2f}x)  ttns={ttns_ms:7.4f}ms "
              f"({row['ttns_speedup_x']:.2f}x)")

    crossover = next((r["dim"] for r in results if r["ttns_speedup_x"] > 1.0), None)
    summary = {
        "note": ("Measured CPU wall-clock forward latency (mean over n_runs, "
                 "warmup discarded), NOT analytic FLOPs. At small dim, Python/"
                 "PyTorch dispatch overhead per contraction step can dominate, "
                 "so a tensor-network layer can be wall-clock SLOWER than dense "
                 "even though it has fewer FLOPs and fewer parameters — this is "
                 "the gap this script measures, complementing the analytic "
                 "'-79.7%' FLOPs figure in results.json."),
        "ttns_wallclock_crossover_dim": crossover,
        "rows": results,
    }
    Path(out_path).write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")
    if crossover:
        print(f"TTNS beats dense in wall-clock from dim={crossover} onward "
              f"(at bond_dimension={bond_dimension}).")
    else:
        print("TTNS never beat dense in wall-clock across the tested dims "
              "(dispatch overhead dominates) — real speedup requires either "
              "larger dim or a fused/compiled kernel, not eager PyTorch ops.")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dims", type=int, nargs="+", default=[128, 256, 512, 1024, 2048])
    ap.add_argument("--bond-dimension", type=int, default=24)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n-runs", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out", default="wallclock_results.json")
    args = ap.parse_args()
    run(args.dims, args.bond_dimension, args.batch, args.n_runs, args.warmup, args.out)
