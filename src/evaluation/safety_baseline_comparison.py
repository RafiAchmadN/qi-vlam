"""Classical-baseline comparison for safety guarantees: heuristic reward
clipping at the path boundary (the common engineering practice) vs a
verifiable Lyapunov/SDP certificate.

Sweeps a scalar gain controlling closed-loop spectral radius rho(A) from
clearly stable to clearly unstable, comparing two ways of answering "is
this control policy safe?":

  (a) heuristic clipping — hard-clamp the state to a bound every step (the
      textbook "prevent visible blow-up" approach with NO analysis of
      whether the system actually converges), safety judged empirically
      per-episode by whether the trajectory ends small.
  (b) our approach — solve_lyapunov_sdp(A) (src/training/vlam_trainer.py,
      unchanged), computed BEFORE running a single episode.

The point is not "our %-safe beats theirs" (rerun with a big enough clip
bound and short enough horizon and clipping alone can look fine for any
finite trial). The demonstrated gap: heuristic indicators can show false
confidence for a system that is genuinely unstable but hasn't visibly
diverged yet within the sampled horizon, while the SDP certificate answers
correctly and instantly because it is derived from the actual mathematical
condition for Schur stability (rho(A) < 1), not from observing outcomes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from src.training.vlam_trainer import solve_lyapunov_sdp


def heuristic_clip_episode(A: torch.Tensor, dim: int, T: int, clip_bound: float,
                           noise: float, generator: torch.Generator) -> torch.Tensor:
    """The baseline itself: linear dynamics x_{t+1} = A x_t + noise, with the
    state hard-clamped to [-clip_bound, clip_bound] every step — no
    Lyapunov shaping, no stability analysis, just "keep it in bounds"."""
    x = torch.randn(dim, generator=generator) * 0.5
    for _ in range(T - 1):
        x = A @ x + noise * torch.randn(dim, generator=generator)
        x = torch.clamp(x, -clip_bound, clip_bound)
    return x


def run_safety_baseline_comparison(
    gains: Sequence[float] = (0.5, 0.9, 1.0, 1.02, 1.05, 1.2),
    dim: int = 4, n_episodes: int = 30, T: int = 15, clip_bound: float = 5.0,
    noise: float = 0.02, seed: int = 0,
    out_path: str = "safety_baseline_comparison.json",
) -> Dict:
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    results: List[Dict] = []
    for gain in gains:
        base = torch.randn(dim, dim, generator=g)
        base = base / torch.linalg.eigvals(base).abs().max()  # normalize rho(base)=1
        A = gain * base
        rho = float(torch.linalg.eigvals(A).abs().max())
        truly_stable = rho < 1.0

        safe = sum(
            1 for _ in range(n_episodes)
            if heuristic_clip_episode(A, dim, T, clip_bound, noise, g).norm().item() < 2.0
        )
        heuristic_safe_pct = 100.0 * safe / n_episodes
        cert = solve_lyapunov_sdp(A)

        row = {
            "gain": gain,
            "spectral_radius": rho,
            "truly_stable": truly_stable,
            "heuristic_clipping_safe_pct": heuristic_safe_pct,
            "sdp_certified": cert.certified,
            "false_confidence": bool(heuristic_safe_pct >= 90.0 and not truly_stable),
        }
        results.append(row)
        print(f"gain={gain:.2f} rho={rho:.3f} truly_stable={truly_stable}  "
              f"heuristic_safe%={heuristic_safe_pct:.0f}  sdp_certified={cert.certified}"
              f"{'  <-- FALSE CONFIDENCE' if row['false_confidence'] else ''}")

    n_false_confidence = sum(r["false_confidence"] for r in results)
    summary = {
        "sweep": results,
        "n_false_confidence_cases": n_false_confidence,
        "note": ("Classical-baseline comparison for safety guarantees (heuristic "
                "reward clipping at path boundary) vs our verifiable SDP certificate (solve_lyapunov_sdp, unchanged). "
                "false_confidence=True marks gains where heuristic clipping reported "
                ">=90% of episodes 'safe' for a system that is mathematically unstable "
                "(spectral radius >= 1) — the SDP certificate never makes this mistake "
                "by construction (Schur stability is exactly rho(A)<1, the same fact "
                "the certificate solves for), and answers before running a single "
                "episode. This is NOT a claim that heuristic clipping always fails; "
                "near the stability boundary specifically, a short finite-horizon trial "
                "can look fine right up until it doesn't."),
    }
    Path(out_path).write_text(json.dumps(summary, indent=2))
    print(f"\n{n_false_confidence}/{len(results)} gains showed heuristic false confidence. "
          f"Wrote {out_path}")
    return summary


if __name__ == "__main__":
    run_safety_baseline_comparison()
