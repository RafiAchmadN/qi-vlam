"""Ablation: bond-dimension Pareto sweep isolating the QI component.

Produces pareto.json mapping bond dimension chi -> (compression ratio,
accuracy), plus a no-QI control (raw truncated SVD baseline is the chi-sweep
of TTNS; MPS is the QI structure under test).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, List

import torch

from src.compression.tensor_network import compress_model
from src.evaluation.benchmark import make_proxy_vlam, make_task, task_accuracy
from src.training.vlam_trainer import VLAMTrainer


def run_ablation(bond_dims: List[int] = (4, 8, 16, 24, 32, 48),
                 seed: int = 0, out_path: str = "pareto.json") -> Dict:
    batch_fn = make_task(seed=seed)
    model = make_proxy_vlam(seed=seed)
    VLAMTrainer(model, max_steps=300, seed=seed).train(batch_fn)
    base_acc = task_accuracy(model, batch_fn)

    points = []
    for chi in bond_dims:
        for method in ("mps", "ttns"):
            m, report = compress_model(copy.deepcopy(model), bond_dimension=chi, method=method)
            acc_noheal = task_accuracy(m, batch_fn)
            VLAMTrainer(m, max_steps=150, lr=1e-3, seed=seed).train(batch_fn)
            acc_healed = task_accuracy(m, batch_fn)
            points.append({
                "method": method,
                "bond_dimension": chi,
                "compression_ratio": report.overall_ratio,
                "accuracy_no_heal": acc_noheal,
                "accuracy_healed": acc_healed,
                "accuracy_drop_pct": 100 * (base_acc - acc_healed) / base_acc,
            })

    # control: healing-only (continue training the uncompressed model the same
    # extra 150 steps) — isolates how much gain is from extra optimization vs
    # from surviving the tensor-network truncation
    ctrl = copy.deepcopy(model)
    VLAMTrainer(ctrl, max_steps=150, lr=1e-3, seed=seed).train(batch_fn)
    healing_only_acc = task_accuracy(ctrl, batch_fn)

    result = {
        "baseline_accuracy": base_acc,
        "healing_only_control_accuracy": healing_only_acc,
        "note": ("ttns == truncated-SVD low-rank baseline (LoRA-style); mps is the "
                 "quantum-inspired structure under test. Comparing the two at equal "
                 "compression isolates whether the MPS topology adds value beyond "
                 "plain low-rank factorization."),
        "pareto": points,
    }
    Path(out_path).write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    res = run_ablation()
    print(json.dumps(res, indent=2))
