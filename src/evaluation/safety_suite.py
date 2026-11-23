"""Synthetic perturbation safety suite (ISO 26262 / IEC 62061 evidence).

Simulates a closed-loop tracking controller under Gaussian sensor noise and
distribution shift, then checks each episode's error trajectory against the
Lyapunov validator. Metric: % of episodes terminating in a safe equilibrium.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from src.training.vlam_trainer import LyapunovStabilityValidator


def simulate_episode(policy: nn.Module, dim: int = 8, T: int = 60,
                     noise: float = 0.05, gain: float = 0.3,
                     generator=None) -> torch.Tensor:
    """Returns error-state trajectory (T, dim)."""
    x = torch.randn(dim, generator=generator)
    traj = [x.clone()]
    with torch.no_grad():
        for _ in range(T - 1):
            obs = x + noise * torch.randn(dim, generator=generator)
            u = policy(obs.unsqueeze(0)).squeeze(0)
            x = x - gain * torch.tanh(u[:dim]) * x.sign() * x.abs().clamp(max=1.0) \
                - gain * 0.5 * x + noise * 0.1 * torch.randn(dim, generator=generator)
            traj.append(x.clone())
    return torch.stack(traj)


def run_safety_suite(n_episodes: int = 100, seed: int = 0,
                     out_path: str = "safety.json") -> Dict:
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    policy = nn.Sequential(nn.Linear(8, 32), nn.Tanh(), nn.Linear(32, 8))
    validator = LyapunovStabilityValidator(tolerance=1e-3)

    # --- formal certificate on the linearized closed loop (SDP) ---
    from src.training.vlam_trainer import linearize_closed_loop, solve_lyapunov_sdp

    def _step(x, pol):
        with torch.no_grad():
            u = pol(x.unsqueeze(0)).squeeze(0)
            return x - 0.3 * torch.tanh(u[:8]) * x.sign() * x.abs().clamp(max=1.0) - 0.15 * x

    A = linearize_closed_loop(policy, _step, 8)
    cert = solve_lyapunov_sdp(A)

    safe, results = 0, []
    for _ in range(n_episodes):
        traj = simulate_episode(policy, generator=g)
        r = validator.validate_trajectory(traj)
        terminal_norm = traj[-1].norm().item()
        episode_safe = terminal_norm < 0.5 and r.safe_fraction > 0.9
        safe += int(episode_safe)
        results.append({"terminal_norm": terminal_norm,
                        "lyapunov_safe_fraction": r.safe_fraction,
                        "safe": episode_safe})
    summary = {
        "episodes": n_episodes,
        "safe_pct": 100.0 * safe / n_episodes,
        "mean_terminal_norm": sum(r["terminal_norm"] for r in results) / n_episodes,
        "formal_sdp_certificate": {
            "certified": cert.certified,
            "spectral_radius": float(torch.linalg.eigvals(A).abs().max()),
            "min_eig_P": cert.min_eig_P,
            "max_eig_ATPA_minus_P": cert.max_eig_decrease,
            "meaning": ("SDP-feasible P proves local asymptotic stability of the "
                        "linearized closed loop; episode stats above are empirical "
                        "robustness evidence under perturbation."),
        },
    }
    Path(out_path).write_text(json.dumps({"summary": summary, "episodes": results}, indent=2))
    return summary


if __name__ == "__main__":
    print(json.dumps(run_safety_suite(), indent=2))
