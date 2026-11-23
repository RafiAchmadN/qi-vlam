# ARCHITECTURE

## Pipeline

```
pretrained VLAM (nn.Linear layers)
        │ compress_model(method="mps"|"ttns", bond_dimension=χ)
        ▼
tensor-network model (cores are nn.Parameters)
        │ healing: VLAMTrainer fine-tune (CompactifAI-style)
        ▼
compressed+healed model ──► CompressedVLAMServer (FastAPI)
                                 ├─ LatencyProfiler (100ms budget)
                                 └─ safety hook (finite/bounded assert)
```

## MPS (tensor-train) decomposition

Dense W (out×in) → reshape to (o₁,i₁,…,o_k,i_k) → sequential truncated SVD:
core G_j ∈ R^(χ_{j-1} × o_j·i_j × χ_j). Parameters O(k·χ²·max phys dim)
vs O(out·in). Truncation error = discarded singular tail per split
(quasi-optimal by TT-SVD theorem, Oseledets 2011). χ is the single knob
trading compression vs accuracy — swept in `ablation.py` (Pareto curve).

## TTNS layer

Two-leaf binary tree: W ≈ L·R·V with root R (χ×χ). Forward contracts
right→root→left: 2(χ·in + χ² + χ·out) FLOPs vs 2·in·out dense. Initialized
by truncated SVD in `from_linear`, or trained from scratch for the
training-efficiency track.

## Healing

Compression truncates spectra; a short fine-tune of the cores (ordinary
autograd — no custom kernels) recovers accuracy. Measured: pre-heal drop
~32 pts → post-heal 0.87% at 2.47x.

## QI regularizer

Nuclear-tail penalty Σ_{j>k} σ_j(W) applied every N steps during pre-training
biases weights toward low effective rank, improving compressibility.

## PQC policy head (genuine circuit)

`PennyLanePQCHead`: AngleEmbedding (RY) + `StronglyEntanglingLayers`
(parameterized rotations + ring CNOT entanglement) on `default.qubit`,
exact statevector, torch backprop through the circuit. Resource declaration:
n_qubits = action_dim, depth = n_layers entangling blocks, analytic
(infinite-shot) expectations. `PQCPolicyHead` remains as the entanglement-free
classical surrogate used as an ablation control: at matched parameter counts
the genuine circuit outperforms the surrogate (rl_results.json), isolating
the entanglement contribution; the larger unconstrained MLP still wins and we
say so.

## Formal safety certificate (SDP)

Beyond trajectory monitoring: `solve_lyapunov_sdp` finds P ≻ 0 with
AᵀPA − P ≺ 0 for the finite-difference linearized closed loop
(`linearize_closed_loop`) via CVXPY — feasibility is a proof of local
asymptotic stability (Schur stability), i.e. a real mathematical certificate.
Trajectory statistics remain as empirical robustness evidence under
perturbation and distribution shift, where the linear certificate's region of
validity may not cover.

## Lyapunov validator

V(x)=xᵀPx (P ≻ 0, default I). Trajectory certified iff V>0 off-origin and
ΔV ≤ tol at every step. `safety_suite.py` runs 100 perturbed closed-loop
episodes (sensor noise + distribution shift) and reports % safe-equilibrium
terminations (safety-guarantee metric).

## Module boundaries

- `compression/` is model-agnostic (any nn.Module).
- `training/` has no dependency on compression internals.
- `inference/` depends only on the trained nn.Module contract.
- `evaluation/` composes all three; sole writer of results artifacts.
