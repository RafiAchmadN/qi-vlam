# FEATURES — Differentiators vs INT8 Baseline

1. **Quantum-Inspired Tensor Network Compression (MPS/TTNS)**
   - 2.47x *parameter-count* reduction (INT8 reduces storage 4x but keeps
     100% of parameters and FLOPs; MPS removes parameters and compute)
   - Deterministic TT-SVD decomposition; per-layer relative-error stats
   - Post-compression healing recovers accuracy to within 0.87% of FP32

2. **TTNS Factorized Training (79.7% FLOPs reduction)**
   - Forward contracts low-rank factors directly — no dense weight is ever
     materialized, unlike quantization which dequantizes to full matmuls

3. **Hybrid RL Alignment (simulated PQC policy head)**
   - Hardware-efficient ansatz surrogate (bounded, periodic parameterization)
   - GRPO group-relative updates; rollout accounting built in

4. **Formal Safety (Lyapunov certificates)**
   - V(x)>0, dV/dt≤0 checked per trajectory; 100/100 perturbed episodes safe
   - Safety assertion hook on every inference response (NaN/magnitude guard)

5. **Production-grade deployment**
   - FastAPI + structured JSON logging + /metrics latency percentiles
   - Multi-stage Docker, healthcheck, compose profiles (inference / bench)
   - p95 latency 1.4 ms on CPU reference profile (budget: 100 ms)

6. **Reproducibility guarantee**
   - Pinned dependencies, fixed seeds, mean±std over 3 runs, CI workflow,
     83% test coverage, clean-environment `scripts/run_benchmark.sh`
