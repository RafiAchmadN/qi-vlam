"""RL alignment: rollout budget to target reward, PQC vs classical.

Compares three policy heads under identical GRPO settings and seeds:
  - mlp:       classical 2-layer MLP (parameter-matched)
  - pqc:       genuine PennyLane circuit (AngleEmbedding + StronglyEntanglingLayers)
  - surrogate: cosine-feature classical control (isolates entanglement effect)

Metric: rollouts consumed to reach target mean reward,
and reward at fixed budget. Mean ± std across seeds -> rl_results.json.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

from src.training.rl_alignment import GRPOAligner, PQCPolicyHead

try:
    from src.training.rl_alignment import PennyLanePQCHead
    _HAS_PQC = True
except Exception:  # pragma: no cover
    _HAS_PQC = False


def make_reward(action_dim: int = 4):
    def reward(states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        target = torch.tanh(states[:, :action_dim])
        return 1.0 - ((actions - target) ** 2).mean(dim=-1)
    return reward


def make_head(kind: str, state_dim: int, action_dim: int, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    if kind == "mlp":
        return nn.Sequential(nn.Linear(state_dim, 16), nn.Tanh(),
                             nn.Linear(16, action_dim))
    if kind == "surrogate":
        return PQCPolicyHead(state_dim, action_dim, n_layers=2)
    if kind == "pqc":
        return PennyLanePQCHead(state_dim, action_dim, n_layers=2)
    raise ValueError(kind)


def run_rl_experiment(n_seeds: int = 3, state_dim: int = 8, action_dim: int = 4,
                      max_rollouts: int = 2000, target: float = 0.85,
                      out_path: str = "rl_results.json") -> Dict:
    kinds = ["mlp", "surrogate"] + (["pqc"] if _HAS_PQC else [])
    reward_fn = make_reward(action_dim)
    results: Dict[str, Dict] = {}
    for kind in kinds:
        rollouts, finals = [], []
        for seed in range(n_seeds):
            head = make_head(kind, state_dim, action_dim, seed)
            res = GRPOAligner(head, seed=seed, lr=1e-2).align(
                reward_fn, state_dim=state_dim,
                max_rollouts=max_rollouts, target_reward=target)
            rollouts.append(float(res.rollouts_used))
            h = res.reward_history
            finals.append(sum(h[-10:]) / min(10, len(h)))
        results[kind] = {
            "rollouts_to_target": {"mean": statistics.mean(rollouts),
                                   "std": statistics.stdev(rollouts) if n_seeds > 1 else 0.0,
                                   "runs": rollouts},
            "final_reward_ma10": {"mean": statistics.mean(finals),
                                  "std": statistics.stdev(finals) if n_seeds > 1 else 0.0,
                                  "runs": finals},
            "params": sum(p.numel() for p in make_head(kind, state_dim, action_dim, 0).parameters()),
        }
    Path(out_path).write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    print(json.dumps(run_rl_experiment(), indent=2))
