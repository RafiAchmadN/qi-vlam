"""Train, compress, certify and export the on-device reach policy for the app.

Pipeline (all CPU, ~1-2 min):
  1. Expert demos: damped-least-squares IK (`arm.ik_step`) on the canonical
     arm model, with DART-style noise injection so the dataset covers states
     slightly off the expert path (reduces BC compounding error).
  2. Behaviour cloning of a dense MLP: features(q, target) -> joint velocity.
  3. TTNS compression of the hidden layers + healing fine-tune.
  4. Closed-loop evaluation (reach success, final error) of dense vs TTNS.
  5. Lyapunov runtime-assurance data: identify e_{t+1} ~ A e_t from the TTNS
     policy's own closed-loop rollouts (ridge sysid) and solve the SDP for P.
     The app's safety monitor evaluates V(e) = e^T P e online and falls back
     to the IK controller when V stops decreasing (Simplex architecture).
     The certificate is formal for the IDENTIFIED LINEAR model only; r^2 of
     that model on held-out transitions is exported next to it.
  6. Export JSON weights + FK test vectors consumed by the JS app.

    python -m src.robot.train_policy --out app/public/models
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.compression.tensor_network import TTNSLinear, compress_model
from src.robot import arm
from src.training.vlam_trainer import identify_linear_dynamics, solve_lyapunov_sdp

DT = 0.05          # control period (s), 20 Hz — same as the app
PITCH = 2.8        # tool pitch (rad) for near-top-down approach
REACH_TOL = 0.01   # 1 cm


def sample_target(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform([0.10, -0.20, 0.02], [0.28, 0.20, 0.20])


def sample_start(rng: np.random.Generator) -> np.ndarray:
    q = arm.Q_HOME.copy()
    q[:4] += rng.normal(0, [0.6, 0.3, 0.4, 0.4])
    return arm.clamp_q(q)


def make_dataset(n_episodes: int, steps: int, seed: int,
                 noise: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for _ in range(n_episodes):
        q, t = sample_start(rng), sample_target(rng)
        for _ in range(steps):
            qd = arm.ik_step(q, t, pitch=PITCH)
            X.append(arm.policy_features(q, t))
            Y.append(qd / arm.QD_MAX[:4])
            # DART: execute a perturbed action, label stays the clean expert
            q[:4] += (qd + rng.normal(0, noise, 4) * arm.QD_MAX[:4]) * DT
            q = arm.clamp_q(q)
    return np.asarray(X, np.float32), np.asarray(Y, np.float32)


def make_policy(in_dim: int = 11, hidden: int = 128, depth: int = 3) -> nn.Module:
    layers: List[nn.Module] = [nn.Linear(in_dim, hidden), nn.GELU(approximate="tanh")]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), nn.GELU(approximate="tanh")]
    layers += [nn.Linear(hidden, 4), nn.Tanh()]
    return nn.Sequential(*layers)


def fit(model: nn.Module, X: torch.Tensor, Y: torch.Tensor, steps: int, lr: float,
        seed: int, batch: int = 1024) -> float:
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    loss = torch.tensor(0.0)
    for _ in range(steps):
        idx = torch.randint(0, X.shape[0], (batch,), generator=g)
        loss = ((model(X[idx]) - Y[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return float(loss)


@torch.no_grad()
def rollout(model: nn.Module, q: np.ndarray, t: np.ndarray, steps: int = 100):
    errs = []
    for _ in range(steps):
        x = torch.from_numpy(arm.policy_features(q, t)).unsqueeze(0)
        qd = model(x)[0].numpy() * arm.QD_MAX[:4]
        q = q.copy()
        q[:4] += qd * DT
        q = arm.clamp_q(q)
        errs.append(t - arm.fk(q))
    return q, np.asarray(errs)


def closed_loop_eval(model: nn.Module, n: int = 200, seed: int = 123) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    finals, succ, steps_to = [], 0, []
    for _ in range(n):
        q, t = sample_start(rng), sample_target(rng)
        _, errs = rollout(model, q, t)
        d = np.linalg.norm(errs, axis=1)
        finals.append(d[-1])
        hit = np.nonzero(d < REACH_TOL)[0]
        if hit.size:
            succ += 1
            steps_to.append(int(hit[0]) + 1)
    return {"reach_success_pct": 100.0 * succ / n,
            "final_err_mm_median": 1000 * float(np.median(finals)),
            "final_err_mm_p95": 1000 * float(np.percentile(finals, 95)),
            "steps_to_reach_median": float(np.median(steps_to)) if steps_to else float("nan"),
            "n_episodes": n}


def lyapunov_from_rollouts(model: nn.Module, n: int = 150, seed: int = 7) -> Dict:
    rng = np.random.default_rng(seed)
    trajs = []
    for _ in range(n):
        _, errs = rollout(model, sample_start(rng), sample_target(rng), steps=60)
        trajs.append(torch.tensor(errs, dtype=torch.float32))
    A, r2 = identify_linear_dynamics(trajs)
    cert = solve_lyapunov_sdp(A)
    P = cert.P.numpy() if cert.P is not None else np.eye(3)
    P = P / np.linalg.eigvalsh(P).max()  # scale-free: monitor only checks decrease
    return {"A": A.numpy().round(6).tolist(), "P": P.round(6).tolist(),
            "certified": bool(cert.certified), "r_squared_heldout": float(r2),
            "spectral_radius": float(np.abs(np.linalg.eigvals(A.numpy())).max()),
            "scope": ("Formal SDP certificate for the identified linear error model "
                      "e_{t+1}=A e_t only; the app uses V=e^T P e as a runtime monitor "
                      "with IK fallback, which is an empirical safeguard, not a proof "
                      "about the nonlinear arm.")}


def export_layers(model: nn.Module) -> List[Dict]:
    out, mods = [], list(model)
    for i, m in enumerate(mods):
        act = None
        if i + 1 < len(mods):
            nxt = mods[i + 1]
            act = "gelu_tanh" if isinstance(nxt, nn.GELU) else "tanh" if isinstance(nxt, nn.Tanh) else None
        r = lambda t: np.asarray(t.detach().numpy(), dtype=np.float64).round(6).tolist()  # noqa: E731
        if isinstance(m, nn.Linear):
            out.append({"type": "dense", "W": r(m.weight), "b": r(m.bias), "act": act})
        elif isinstance(m, TTNSLinear):
            out.append({"type": "ttns", "right": r(m.right), "root": r(m.root),
                        "left": r(m.left), "b": r(m.bias), "act": act})
    return out


def fk_vectors(n: int = 50, seed: int = 3) -> Dict:
    rng = np.random.default_rng(seed)
    qs = rng.uniform(arm.Q_MIN, arm.Q_MAX, size=(n, 6))
    return {"q": qs.round(6).tolist(), "ee": arm.fk(qs.round(6)).round(9).tolist(),
            "links": list(arm.LINKS)}


def main(out_dir: str = "app/public/models", seed: int = 0, n_episodes: int = 1500,
         steps: int = 4000, heal_steps: int = 1000, chi: int = 16, n_eval: int = 200,
         n_lyap: int = 150, results_path: str = "results/app_policy.json") -> Dict:
    t0 = time.time()
    torch.manual_seed(seed)
    X, Y = make_dataset(n_episodes, 60, seed)
    X, Y = torch.from_numpy(X), torch.from_numpy(Y)
    dense = make_policy()
    fit(dense, X, Y, steps, 2e-3, seed)
    ttns_noheal, rep = compress_model(copy.deepcopy(dense), bond_dimension=chi,
                                      method="ttns", min_features=64)
    ttns = copy.deepcopy(ttns_noheal)
    fit(ttns, X, Y, heal_steps, 5e-4, seed + 1)
    ctrl = copy.deepcopy(dense)
    fit(ctrl, X, Y, heal_steps, 5e-4, seed + 1)  # healing-only control

    n_dense = sum(p.numel() for p in dense.parameters())
    n_ttns = sum(p.numel() for p in ttns.parameters())
    evals = {name: closed_loop_eval(m, n=n_eval) for name, m in
             (("dense", dense), ("healing_ctrl", ctrl), ("ttns_noheal", ttns_noheal),
              ("ttns", ttns))}
    lyap = lyapunov_from_rollouts(ttns, n=n_lyap)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"format": "qi-vla-policy/1", "in_dim": 11, "out_dim": 4, "dt": DT,
            "pitch": PITCH, "qd_max": arm.QD_MAX[:4].tolist(),
            "features": "q0..q3, target xyz, 5*(target-ee), 1"}
    tx = X[:16]
    for name, m in (("dense", dense), ("ttns", ttns)):
        with torch.no_grad():
            ty = m(tx)
        payload = dict(meta, method=name, params=sum(p.numel() for p in m.parameters()),
                       layers=export_layers(m), closed_loop=evals[name],
                       lyapunov=lyap,
                       test_vectors={"x": tx.numpy().round(6).tolist(),
                                     "y": ty.numpy().round(6).tolist()})
        (out / f"policy_{name}.json").write_text(json.dumps(payload))
    (out / "fk_vectors.json").write_text(json.dumps(fk_vectors()))
    summary = {"params_dense": n_dense, "params_ttns": n_ttns,
               "param_ratio": n_dense / n_ttns,
               "compressed_layer_ratio": rep.overall_ratio, "bond_dimension": chi,
               "closed_loop": evals, "lyapunov": {k: v for k, v in lyap.items() if k != "A"},
               "dataset": {"samples": int(X.shape[0]), "episodes": n_episodes},
               "wall_s": time.time() - t0}
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    Path(results_path).write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="app/public/models")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    print(json.dumps(main(a.out, a.seed), indent=2))
