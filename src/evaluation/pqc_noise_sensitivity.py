"""Hardware pathway: simulation-based analysis of near-term quantum hardware feasibility
for the PQC policy head. Qubit count and circuit depth are already declared
via PennyLanePQCHead.circuit_resources() (src/training/rl_alignment.py) —
this adds the missing piece, noise sensitivity, without needing real quantum
hardware access.

Trains a PennyLanePQCHead on the CPU-proxy RL task (same task and code path
as src/evaluation/rl_experiment.py) under the noiseless default.qubit
simulator, exactly as reported elsewhere in the project. The SAME trained
weights are then re-evaluated on PennyLane's default.mixed density-matrix
simulator with a depolarizing channel inserted after every gate
(qml.insert(..., position="all")), sweeping the per-gate noise probability
from 0 (reproduces the noiseless result) up toward levels representative of
current NISQ hardware (~0.1-1% per-gate error for the best gates, worse for
two-qubit gates — see note below), reporting how much task reward degrades.

This is evaluation-only: frozen weights, no retraining under noise. It is a
first-order "would this policy survive near-term hardware noise" check, not
a claim about any specific real device's calibration.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Sequence

import torch

from src.training.rl_alignment import GRPOAligner, PennyLanePQCHead
from src.evaluation.rl_experiment import make_reward

try:
    import pennylane as qml
    _HAS_PQC = True
except ImportError:  # pragma: no cover
    _HAS_PQC = False


def _noisy_forward(head: "PennyLanePQCHead", noise_prob: float):
    """Rebuilds head's circuit on default.mixed with the SAME trained
    weights, inserting a DepolarizingChannel after every gate when
    noise_prob > 0. Evaluation only — no grad, no training under noise."""
    n_qubits = head.n_qubits
    dev = qml.device("default.mixed", wires=n_qubits)
    weights = dict(head.qlayer.named_parameters())["weights"].detach()

    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    qnode = qml.QNode(circuit, dev, interface="torch")
    if noise_prob > 0:
        qnode = qml.insert(op=qml.DepolarizingChannel, op_args=noise_prob,
                           position="all")(qnode)

    @torch.no_grad()
    def forward(x: torch.Tensor) -> torch.Tensor:
        a = torch.tanh(head.encoder(x)) * math.pi
        outs = [torch.stack(qnode(ai, weights)) for ai in a]
        z = torch.stack(outs)
        return z * head.readout.detach()

    return forward


def run_noise_sensitivity(
    state_dim: int = 8, action_dim: int = 4, seed: int = 0,
    train_rollouts: int = 2000, n_eval_states: int = 64,
    noise_levels: Sequence[float] = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1),
    out_path: str = "pqc_noise_sensitivity.json",
) -> Dict:
    if not _HAS_PQC:
        raise RuntimeError("pennylane not installed")

    torch.manual_seed(seed)
    head = PennyLanePQCHead(state_dim, action_dim, n_layers=2)
    reward_fn = make_reward(action_dim)
    GRPOAligner(head, seed=seed, lr=1e-2).align(
        reward_fn, state_dim=state_dim, max_rollouts=train_rollouts,
        target_reward=0.9)

    gen = torch.Generator().manual_seed(seed + 1)
    eval_states = torch.randn(n_eval_states, state_dim, generator=gen)
    target = torch.tanh(eval_states[:, :action_dim])

    rows = []
    clean_reward = None
    for noise_prob in noise_levels:
        forward = _noisy_forward(head, noise_prob)
        actions = forward(eval_states)
        reward = float((1.0 - ((torch.tanh(actions) - target) ** 2)
                        .mean(dim=-1)).mean())
        if noise_prob == 0.0:
            clean_reward = reward
        rows.append({"noise_prob": noise_prob, "mean_reward": reward})
    for row in rows:
        row["reward_retained_pct"] = (
            100.0 * row["mean_reward"] / clean_reward if clean_reward else None)

    summary = {
        "circuit_resources": head.circuit_resources(),
        "train_rollouts": train_rollouts,
        "n_eval_states": n_eval_states,
        "sweep": rows,
        "note": (
            "Depolarizing noise inserted after every gate "
            "(qml.insert(op=DepolarizingChannel, position='all')) on "
            "PennyLane's default.mixed density-matrix simulator, using the "
            "SAME weights trained noiselessly (default.qubit) elsewhere in "
            "this project — this measures robustness of an already-trained "
            "policy to hardware noise, not training under noise. "
            "noise_prob is a per-gate depolarizing probability, not a "
            "specific device's measured error rate; current superconducting "
            "NISQ hardware single-qubit gate error is roughly 0.05-0.5%, "
            "two-qubit (entangling) gate error roughly 0.5-2% depending on "
            "hardware generation — the swept range brackets this. "
            "reward_retained_pct = 100 * noisy_reward / noise_prob=0_reward "
            "on the SAME eval states; a value near 100% at realistic noise "
            "levels would suggest tolerance to near-term hardware noise, a "
            "sharp drop would suggest this ansatz needs error mitigation "
            "before real-hardware deployment. Simulation only — no physical "
            "quantum hardware used."
        ),
    }
    Path(out_path).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run_noise_sensitivity()
