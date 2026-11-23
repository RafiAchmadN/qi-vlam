"""VLAM training loop with quantum-inspired regularization and Lyapunov safety.

- QuantumInspiredRegularization: nuclear-norm surrogate penalty pushing weight
  matrices toward low effective rank, so post-hoc MPS/TTNS truncation loses
  less accuracy and convergence is measured against an AdamW + cosine baseline.
- LyapunovStabilityValidator: certifies that a control policy's closed-loop
  trajectory satisfies V(x) > 0 and dV/dt <= 0 along sampled rollouts,
  providing empirical safe-equilibrium evidence (in the spirit of
  ISO 26262 / IEC 62061; not a certification).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class QuantumInspiredRegularization:
    """Low-rank (nuclear-norm surrogate) penalty on large Linear weights.

    Uses ||W||_* approximated via a few power-iteration singular values to
    stay cheap; encourages spectra compatible with low bond-dimension MPS.
    """

    def __init__(self, weight: float = 1e-4, top_k: int = 8, min_features: int = 64) -> None:
        self.weight = weight
        self.top_k = top_k
        self.min_features = min_features

    def __call__(self, model: nn.Module) -> torch.Tensor:
        penalty = torch.zeros((), device=next(model.parameters()).device)
        for m in model.modules():
            if isinstance(m, nn.Linear) and min(m.in_features, m.out_features) >= self.min_features:
                # tail energy: total Frobenius^2 minus top-k singular energy
                W = m.weight
                try:
                    S = torch.linalg.svdvals(W)
                except RuntimeError:  # pragma: no cover
                    continue
                k = min(self.top_k, S.shape[0])
                tail = S[k:].sum()
                penalty = penalty + tail
        return self.weight * penalty


@dataclass
class LyapunovResult:
    positive_definite: bool
    monotonically_decreasing: bool
    violations: int
    total_steps: int

    @property
    def certified(self) -> bool:
        return self.positive_definite and self.monotonically_decreasing

    @property
    def safe_fraction(self) -> float:
        return 1.0 - self.violations / max(self.total_steps, 1)


class LyapunovStabilityValidator:
    """Empirical Lyapunov certificate over sampled trajectories.

    V(x) = x^T P x with P symmetric positive-definite (default identity).
    A trajectory is certified safe if V(x_t) > 0 for x_t != 0 and
    V(x_{t+1}) - V(x_t) <= tolerance for all t.

    Note: trajectory checking is *monitoring* (necessary evidence, not proof).
    For a formal certificate on the linearized closed loop, use
    `solve_lyapunov_sdp`, which finds P > 0 with A^T P A - P < 0 via
    semidefinite programming — a mathematical proof of local asymptotic
    stability for the discrete-time system x_{t+1} = A x_t.
    """

    def __init__(self, P: Optional[torch.Tensor] = None, tolerance: float = 1e-6) -> None:
        self.P = P
        self.tolerance = tolerance

    def V(self, x: torch.Tensor) -> torch.Tensor:
        if self.P is None:
            return (x * x).sum(dim=-1)
        return torch.einsum("...i,ij,...j->...", x, self.P, x)

    def validate_trajectory(self, states: torch.Tensor) -> LyapunovResult:
        """states: (T, dim) sequence of error/deviation states."""
        v = self.V(states)
        pos = bool((v[states.norm(dim=-1) > 1e-9] > 0).all()) if states.numel() else True
        dv = v[1:] - v[:-1]
        violations = int((dv > self.tolerance).sum().item())
        return LyapunovResult(
            positive_definite=pos,
            monotonically_decreasing=violations == 0,
            violations=violations,
            total_steps=max(len(states) - 1, 0),
        )

    def validate_batch(self, batch_states: torch.Tensor) -> List[LyapunovResult]:
        return [self.validate_trajectory(traj) for traj in batch_states]


@dataclass
class SDPCertificate:
    feasible: bool
    P: Optional[torch.Tensor]
    min_eig_P: float
    max_eig_decrease: float  # largest eigenvalue of A^T P A - P (must be < 0)

    @property
    def certified(self) -> bool:
        return self.feasible and self.min_eig_P > 0 and self.max_eig_decrease < 0


def linearize_closed_loop(policy: "nn.Module", dynamics_step, dim: int,
                          eps: float = 1e-4) -> torch.Tensor:
    """Finite-difference Jacobian A of x_{t+1} = f(x_t, policy(x_t)) at x=0."""
    A = torch.zeros(dim, dim)
    x0 = torch.zeros(dim)
    f0 = dynamics_step(x0, policy)
    for i in range(dim):
        xp = x0.clone(); xp[i] += eps
        A[:, i] = (dynamics_step(xp, policy) - f0) / eps
    return A


def solve_lyapunov_sdp(A: torch.Tensor, margin: float = 1e-3) -> SDPCertificate:
    """Formal discrete-time Lyapunov certificate via SDP.

    Finds P >= margin*I with A^T P A - P <= -margin*I. Feasibility proves the
    linearized closed loop is asymptotically stable (Schur stable), i.e. every
    trajectory in a neighborhood of the equilibrium converges — a mathematical
    certificate, not an empirical check.
    """
    try:
        import cvxpy as cp
    except ImportError:  # pragma: no cover
        return SDPCertificate(False, None, 0.0, 0.0)
    n = A.shape[0]
    An = A.detach().cpu().numpy()
    P = cp.Variable((n, n), symmetric=True)
    cons = [P >> margin * torch.eye(n).numpy(),
            An.T @ P @ An - P << -margin * torch.eye(n).numpy()]
    prob = cp.Problem(cp.Minimize(cp.trace(P)), cons)
    try:
        prob.solve()
    except cp.error.SolverError:  # pragma: no cover
        return SDPCertificate(False, None, 0.0, 0.0)
    if P.value is None:
        return SDPCertificate(False, None, 0.0, 0.0)
    Pt = torch.tensor(P.value, dtype=torch.float32)
    eig_P = torch.linalg.eigvalsh(Pt)
    M = torch.tensor(An.T @ P.value @ An - P.value, dtype=torch.float32)
    eig_M = torch.linalg.eigvalsh(M)
    return SDPCertificate(True, Pt, float(eig_P.min()), float(eig_M.max()))


def identify_linear_dynamics(trajectories: List[torch.Tensor],
                             ridge: float = 1e-3) -> Tuple[torch.Tensor, float]:
    """Ridge least-squares system identification: fit A minimizing
    ||e_{t+1} - A e_t||^2 + ridge*||A||^2 across all (e_t, e_{t+1}) pairs
    pooled from `trajectories` (each a (T, dim) tensor of the same
    error-state quantity fed to LyapunovStabilityValidator elsewhere).

    Unlike `linearize_closed_loop` (an EXACT finite-difference Jacobian of a
    known closed-form `dynamics_step`), this identifies an APPROXIMATE
    linear model from data — the only option when no closed-form dynamics
    function exists (e.g. a black-box physics simulator such as CoppeliaSim,
    see scripts/rlbench_eval.py). The returned `A` is only as good as the
    data: the returned r_squared (computed on a held-out 20% split, NOT the
    training data) reports how well `A` actually explains unseen
    transitions. A low r_squared means `solve_lyapunov_sdp(A)` would be
    certifying an unfaithful model of the real system, not the real system
    itself — always report r_squared alongside the certificate; never claim
    this proves anything about the true nonlinear simulator, only about the
    identified linear approximation (same "formal-for-the-model vs
    empirical-for-the-system" honesty split solve_lyapunov_sdp already
    enforces for the analytic case).
    """
    e_t, e_tp1 = [], []
    for traj in trajectories:
        if traj.shape[0] < 2:
            continue
        e_t.append(traj[:-1])
        e_tp1.append(traj[1:])
    if not e_t:
        raise ValueError("identify_linear_dynamics: need >=1 trajectory with >=2 steps")
    X = torch.cat(e_t, dim=0).float()    # (N, dim) = e_t
    Y = torch.cat(e_tp1, dim=0).float()  # (N, dim) = e_{t+1}
    dim = X.shape[1]

    n = X.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    n_train = max(1, int(0.8 * n))
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    X_train, Y_train = X[train_idx], Y[train_idx]

    # Multivariate ridge regression: Y_train ~ X_train @ A^T
    # Normal equations: (X^T X + ridge*I) @ A^T = X^T Y
    XtX = X_train.T @ X_train + ridge * torch.eye(dim)
    XtY = X_train.T @ Y_train
    A = torch.linalg.solve(XtX, XtY).T  # (dim, dim)

    if test_idx.numel() > 0:
        X_test, Y_test = X[test_idx], Y[test_idx]
        pred = X_test @ A.T
        ss_res = ((Y_test - pred) ** 2).sum().item()
        ss_tot = ((Y_test - Y_test.mean(dim=0)) ** 2).sum().item()
        r_squared = 1.0 - ss_res / max(ss_tot, 1e-12)
    else:  # pragma: no cover — only with a single-transition dataset
        r_squared = float("nan")

    return A, r_squared


@dataclass
class TrainingResult:
    steps_to_convergence: int
    wall_clock_seconds: float
    final_loss: float
    loss_history: List[float] = field(default_factory=list)


class VLAMTrainer:
    """Minimal deterministic trainer: AdamW + cosine LR, optional QI penalty."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        max_steps: int = 500,
        convergence_loss: float = 0.05,
        qi_regularizer: Optional[QuantumInspiredRegularization] = None,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        torch.manual_seed(seed)
        self.model = model.to(device)
        self.device = device
        self.max_steps = max_steps
        self.convergence_loss = convergence_loss
        self.qi = qi_regularizer
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=max_steps)

    def train(
        self,
        batch_fn: Callable[[], tuple],
        loss_fn: Optional[nn.Module] = None,
        qi_every: int = 10,
    ) -> TrainingResult:
        loss_fn = loss_fn or nn.MSELoss()
        history: List[float] = []
        t0 = time.perf_counter()
        step = self.max_steps
        for i in range(self.max_steps):
            x, y = batch_fn()
            x, y = x.to(self.device), y.to(self.device)
            self.opt.zero_grad()
            loss = loss_fn(self.model(x), y)
            if self.qi is not None and i % qi_every == 0:
                loss = loss + self.qi(self.model)
            loss.backward()
            self.opt.step()
            self.sched.step()
            history.append(float(loss.item()))
            if loss.item() < self.convergence_loss:
                step = i + 1
                break
        return TrainingResult(
            steps_to_convergence=step,
            wall_clock_seconds=time.perf_counter() - t0,
            final_loss=history[-1] if history else math.inf,
            loss_history=history,
        )
