"""Hybrid RL alignment: GRPO-style updates with a PQC policy head.

Two implementations:

- PennyLanePQCHead (primary): a genuine parameterized quantum circuit —
  angle embedding + StronglyEntanglingLayers (RY/RZ rotations with ring CNOT
  entanglement) simulated exactly on `default.qubit`, differentiable via the
  torch interface. Resource assumptions: n_qubits = action_dim, depth =
  n_layers entangling blocks, noiseless statevector simulation, analytic
  expectation values (shot budget: exact / infinite-shot limit).

- PQCPolicyHead (fallback surrogate): cosine-feature classical stand-in with
  the same parameter count, used when PennyLane is unavailable and as the
  ablation control demonstrating that entanglement (not merely periodic
  activations) drives any observed difference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Tuple

import torch
import torch.nn as nn

try:
    import pennylane as qml
    _PENNYLANE = True
except ImportError:  # pragma: no cover
    _PENNYLANE = False


class PennyLanePQCHead(nn.Module):
    """Real PQC: angle embedding -> StronglyEntanglingLayers -> <Z_i> readout."""

    def __init__(self, in_features: int, n_qubits: int, n_layers: int = 2) -> None:
        super().__init__()
        if not _PENNYLANE:  # pragma: no cover
            raise RuntimeError("pennylane not installed")
        self.encoder = nn.Linear(in_features, n_qubits)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        shapes = {"weights": qml.StronglyEntanglingLayers.shape(n_layers, n_qubits)}
        self.qlayer = qml.qnn.TorchLayer(circuit, shapes)
        self.readout = nn.Parameter(torch.ones(n_qubits))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.tanh(self.encoder(x)) * math.pi
        z = self.qlayer(a)  # (batch, n_qubits) in [-1, 1]
        return z * self.readout

    def circuit_resources(self) -> dict:
        """Hardware-pathway declaration for the technical report."""
        return {
            "n_qubits": self.n_qubits,
            "entangling_layers": self.n_layers,
            "gate_count_estimate": self.n_layers * self.n_qubits * 4,  # 3 rot + 1 CNOT
            "simulator": "pennylane default.qubit (exact statevector)",
            "shots": "analytic (infinite-shot limit)",
        }


class PQCPolicyHead(nn.Module):
    """Classical cosine-feature surrogate (ablation control, no entanglement)."""

    def __init__(self, in_features: int, n_qubits: int, n_layers: int = 2) -> None:
        super().__init__()
        self.encoder = nn.Linear(in_features, n_qubits)
        self.thetas = nn.Parameter(torch.randn(n_layers, n_qubits, 2) * 0.1)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.readout = nn.Parameter(torch.ones(n_qubits))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # angle encoding
        a = torch.tanh(self.encoder(x)) * math.pi
        # RY-RZ rotations act on Bloch z-expectation: <Z> = cos(angle_total)
        for layer in range(self.n_layers):
            a = a + self.thetas[layer, :, 0]
            a = torch.cos(a) * math.pi * torch.sigmoid(self.thetas[layer, :, 1])
        z_expect = torch.cos(a)  # bounded in [-1, 1]
        return z_expect * self.readout


@dataclass
class GRPOResult:
    rollouts_used: int
    final_mean_reward: float
    reward_history: List[float]


class GRPOAligner:
    """Group-Relative Policy Optimization on a bandit-style control task.

    Compares classical MLP head vs PQC head under identical rollout budgets.
    """

    def __init__(self, policy: nn.Module, group_size: int = 8, lr: float = 5e-3, seed: int = 0):
        torch.manual_seed(seed)
        self.policy = policy
        self.group_size = group_size
        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)

    def align(
        self,
        env_reward: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
        state_dim: int = None,
        max_rollouts: int = 4000,
        target_reward: float = 0.9,
        rollout_fn: Callable[[nn.Module, int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> GRPOResult:
        """env_reward/state_dim: the original bandit-style mode — states are
        i.i.d. Gaussian draws, reward is a closed-form function of
        (state, action). rollout_fn(policy, group_size) -> (states, actions,
        rewards): an alternative for tasks where reward can only be known by
        actually EXECUTING an action (e.g. a real RLBench episode step) —
        states come from real environment resets and rewards from real
        environment feedback, not a synthetic reward function. Exactly one
        of (env_reward, state_dim) or rollout_fn must be given; existing
        bandit-mode callers are unaffected."""
        assert (rollout_fn is not None) != (env_reward is not None and state_dim is not None), \
            "pass either rollout_fn, or both env_reward and state_dim"
        history: List[float] = []
        rollouts = 0
        while rollouts < max_rollouts:
            if rollout_fn is not None:
                states, actions, rewards = rollout_fn(self.policy, self.group_size)
                mean = self.policy(states)
            else:
                states = torch.randn(self.group_size, state_dim)
                mean = self.policy(states)
                noise = 0.1 * torch.randn_like(mean)
                actions = (mean + noise).detach()
                rewards = env_reward(states, torch.tanh(actions))  # (group,)
            rollouts += self.group_size
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            logp = -((actions - mean) ** 2).sum(dim=-1) / (2 * 0.1 ** 2)
            loss = -(adv.detach() * logp).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            history.append(float(rewards.mean().item()))
            if len(history) >= 5 and sum(history[-5:]) / 5 >= target_reward:
                break
        return GRPOResult(rollouts, history[-1] if history else 0.0, history)
