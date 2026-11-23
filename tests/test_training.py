import torch
import torch.nn as nn

from src.training.vlam_trainer import (
    LyapunovStabilityValidator, QuantumInspiredRegularization,
    VLAMTrainer,
)
from src.training.rl_alignment import GRPOAligner, PQCPolicyHead


_X = torch.randn(32, 16, generator=torch.Generator().manual_seed(123))
_Y = _X @ torch.eye(16, 4)


def _batch_fn():
    return _X, _Y


class TestTrainer:
    def test_loss_decreases(self):
        model = nn.Linear(16, 4)
        r = VLAMTrainer(model, max_steps=100, seed=0).train(_batch_fn)
        assert r.loss_history[-1] < r.loss_history[0]

    def test_qi_regularizer_scalar(self):
        reg = QuantumInspiredRegularization(weight=1e-3, min_features=8)
        p = reg(nn.Sequential(nn.Linear(16, 16)))
        assert p.ndim == 0 and p.item() >= 0

    def test_deterministic(self):
        torch.manual_seed(0)
        r1 = VLAMTrainer(nn.Linear(16, 4), max_steps=20, seed=7).train(_batch_fn)
        torch.manual_seed(0)
        r2 = VLAMTrainer(nn.Linear(16, 4), max_steps=20, seed=7).train(_batch_fn)
        assert r1.loss_history == r2.loss_history


class TestLyapunov:
    def test_decaying_trajectory_certified(self):
        t = torch.arange(50).float().unsqueeze(1)
        states = torch.exp(-0.1 * t) * torch.ones(50, 4)
        r = LyapunovStabilityValidator().validate_trajectory(states)
        assert r.certified and r.safe_fraction == 1.0

    def test_growing_trajectory_rejected(self):
        t = torch.arange(50).float().unsqueeze(1)
        states = torch.exp(0.1 * t) * torch.ones(50, 4)
        r = LyapunovStabilityValidator().validate_trajectory(states)
        assert not r.monotonically_decreasing


class TestPQC:
    def test_bounded_output(self):
        head = PQCPolicyHead(16, 4)
        y = head(torch.randn(8, 16))
        assert y.shape == (8, 4)
        assert torch.isfinite(y).all()

    def test_grpo_improves_reward(self):
        def reward(states, actions):
            target = torch.tanh(states[:, :4])
            return 1.0 - ((actions - target) ** 2).mean(dim=-1)

        head = PQCPolicyHead(8, 4)
        res = GRPOAligner(head, seed=0, lr=1e-2).align(
            reward, state_dim=8, max_rollouts=3000, target_reward=0.9)
        h = res.reward_history
        early = sum(h[:10]) / 10
        late = sum(h[-10:]) / 10
        assert late > early, f"reward did not improve: {early:.3f} -> {late:.3f}"

    def test_grpo_rollout_fn_improves_reward(self):
        """rollout_fn is the extension point scripts/rlbench_rl_experiment.py
        uses to feed real RLBench (state, action, reward) triples instead of
        the synthetic bandit env_reward — exercised here with a synthetic
        stand-in so the test stays fast and RLBench-free, but through the
        exact same align() code path."""
        state_dim, action_dim = 8, 4
        target = torch.randn(action_dim)

        def rollout_fn(policy, group_size):
            states = torch.randn(group_size, state_dim)
            with torch.no_grad():
                mean = policy(states)
            actions = mean + 0.1 * torch.randn_like(mean)
            rewards = 1.0 - ((torch.tanh(actions) - target) ** 2).mean(dim=-1)
            return states, actions, rewards

        head = PQCPolicyHead(state_dim, action_dim)
        res = GRPOAligner(head, seed=0, lr=1e-2).align(
            rollout_fn=rollout_fn, max_rollouts=3000, target_reward=0.9)
        h = res.reward_history
        early = sum(h[:10]) / 10
        late = sum(h[-10:]) / 10
        assert late > early, f"reward did not improve: {early:.3f} -> {late:.3f}"

    def test_grpo_align_requires_exactly_one_mode(self):
        head = PQCPolicyHead(8, 4)
        aligner = GRPOAligner(head, seed=0)
        import pytest
        with pytest.raises(AssertionError):
            aligner.align()  # neither env_reward/state_dim nor rollout_fn given


class TestFormalCertificate:
    def test_stable_system_certified(self):
        from src.training.vlam_trainer import solve_lyapunov_sdp
        A = torch.eye(4) * 0.8
        cert = solve_lyapunov_sdp(A)
        assert cert.certified

    def test_unstable_system_rejected(self):
        from src.training.vlam_trainer import solve_lyapunov_sdp
        cert = solve_lyapunov_sdp(torch.eye(4) * 1.3)
        assert not cert.certified


class TestSystemIdentification:
    def test_recovers_known_linear_system(self):
        from src.training.vlam_trainer import identify_linear_dynamics
        g = torch.Generator().manual_seed(0)
        dim = 5
        A_true = 0.7 * torch.eye(dim) + 0.05 * torch.randn(dim, dim, generator=g)
        trajs = []
        for _ in range(15):
            x = torch.randn(dim, generator=g)
            traj = [x]
            for _ in range(30):
                x = A_true @ x + 0.01 * torch.randn(dim, generator=g)
                traj.append(x)
            trajs.append(torch.stack(traj))
        A_hat, r2 = identify_linear_dynamics(trajs)
        assert (A_hat - A_true).norm() / A_true.norm() < 0.05
        assert r2 > 0.9

    def test_feeds_into_sdp_certificate(self):
        from src.training.vlam_trainer import identify_linear_dynamics, solve_lyapunov_sdp
        g = torch.Generator().manual_seed(1)
        dim = 3
        A_true = torch.eye(dim) * 0.5
        trajs = []
        for _ in range(10):
            x = torch.randn(dim, generator=g)
            traj = [x]
            for _ in range(20):
                x = A_true @ x
                traj.append(x)
            trajs.append(torch.stack(traj))
        A_hat, _ = identify_linear_dynamics(trajs)
        cert = solve_lyapunov_sdp(A_hat)
        assert cert.certified

    def test_raises_on_empty_trajectories(self):
        from src.training.vlam_trainer import identify_linear_dynamics
        try:
            identify_linear_dynamics([torch.randn(1, 3)])  # single step, no transitions
            assert False, "expected ValueError"
        except ValueError:
            pass


class TestPennyLanePQC:
    def test_forward_backward(self):
        pytest = __import__("pytest")
        try:
            from src.training.rl_alignment import PennyLanePQCHead
        except Exception:
            pytest.skip("pennylane unavailable")
        head = PennyLanePQCHead(8, 4, n_layers=1)
        y = head(torch.randn(3, 8))
        assert y.shape == (3, 4) and (y.abs() <= 1.01).all()
        y.sum().backward()
        assert any(p.grad is not None for p in head.qlayer.parameters())
