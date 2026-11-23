"""Train a NEW pick-and-place policy on the simulated dataset, compress it with
TTNS, and evaluate it closed-loop in MuJoCo physics — including environments
and dynamics it never saw (sim-to-real stress tests).

    python -m src.sim.collect --out data/qivla_sim                    # DR data
    python -m src.sim.collect --out data/qivla_sim_nodr --no-randomize --seed 100000
    python -m src.sim.train_pickplace --seeds 3                       # -> results/sim_pickplace.json
                                                                      #    app/public/models/pickplace_*.json

Models (per training seed):
  dense_dr        dense MLP trained on domain-randomized data
  dense_nodr      same architecture, nominal-dynamics data only (DR ablation)
  healing_ctrl    dense_dr + the same extra steps the compressed model gets
  ttns_dr         TTNS-compressed dense_dr + healing fine-tune
  ttns_noheal     TTNS-compressed dense_dr, no healing

Evaluation sets (closed loop, success = cube resting on the goal, released):
  train_envs      the 4 training places, randomized dynamics (in-distribution)
  heldout_env     factory_conveyor: unseen place AND worn-servo dynamics
  hard_dynamics   training places with fixed parameters outside the DR range
                  (lag 0.22 s, 3-step latency, 8 mrad backlash, weaker servos)

All evaluation runs through the exported NumPy runtime (`PolicyRuntime`) —
the same layers and math the phone app executes.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from src.robot import arm
from src.sim.features import PP_DIM, PP_OUT, batch_features, pickplace_features

TRAIN_ENVS = ("lab_table", "kitchen_counter", "warehouse_shelf", "classroom_desk")
HELDOUT_ENV = "factory_conveyor"


# ---------------------------------------------------------------- data
def load_dataset(root: str):
    import pyarrow.parquet as pq
    import glob
    files = sorted(glob.glob(os.path.join(root, "data", "chunk-*", "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no dataset under {root!r}; run python -m src.sim.collect")
    cols = ["observation.state", "observation.environment_state", "action.velocity", "episode_index"]
    t = [pq.read_table(f, columns=cols) for f in files]
    get = lambda c: np.concatenate([np.asarray(x.column(c).to_pylist(), dtype=np.float32) for x in t])  # noqa: E731
    state, env, vel = get("observation.state"), get("observation.environment_state"), get("action.velocity")
    ep = np.concatenate([x.column("episode_index").to_numpy() for x in t])
    X = batch_features(state, env)
    Y = np.concatenate([vel[:, :4] / arm.QD_MAX[:4], 2.0 * vel[:, 4:5] - 1.0], axis=1).astype(np.float32)
    return X, Y, ep


def make_policy(hidden: int = 256, depth: int = 3) -> nn.Module:
    layers: List[nn.Module] = [nn.Linear(PP_DIM, hidden), nn.GELU(approximate="tanh")]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), nn.GELU(approximate="tanh")]
    layers += [nn.Linear(hidden, PP_OUT), nn.Tanh()]
    return nn.Sequential(*layers)


def loss_fn(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ((pred[:, :4] - y[:, :4]) ** 2).mean() + 0.5 * ((pred[:, 4] - y[:, 4]) ** 2).mean()


def fit(model: nn.Module, X: torch.Tensor, Y: torch.Tensor, steps: int, lr: float, seed: int,
        batch: int = 2048) -> float:
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
    loss = torch.tensor(0.0)
    for _ in range(steps):
        idx = torch.randint(0, X.shape[0], (batch,), generator=g)
        loss = loss_fn(model(X[idx]), Y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return float(loss)


@torch.no_grad()
def val_loss(model: nn.Module, X: torch.Tensor, Y: torch.Tensor) -> float:
    return float(loss_fn(model(X), Y))


# ---------------------------------------------------------------- closed loop
def policy_episode(spec: Dict, env_id: str, seed: int, randomize: bool,
                   dynamics: Optional[Dict] = None, max_steps: int = 500) -> Dict:
    """Run one episode with an exported policy; mirrors app/src/vla/pickplace.js."""
    from src.robot.policy_runtime import PolicyRuntime
    from src.sim.envs import sample_episode
    from src.sim.expert import MAX_WINDUP, roll_for_yaw
    from src.sim.physics import DT, ArmSim

    rt = PolicyRuntime(spec)
    ep = sample_episode(env_id, seed, randomize=randomize)
    sim = ArmSim(ep, dynamics=dynamics)
    q_cmd = arm.Q_HOME.copy()
    roll = 0.0
    stable = 0
    for k in range(max_steps):
        obs = sim.observe()
        q = obs["q"]
        y = rt.forward(pickplace_features(q, obs["obj"], obs["goal"], ep.goal_kind == "cube"))
        qd = np.clip(y[:4], -1, 1) * arm.QD_MAX[:4]
        grip = 1.0 if y[4] > 0 else 0.0
        if grip > 0.5 and obs["obj"][2] < sim.cube_half + 0.01:
            roll = roll_for_yaw(q[0], obs["obj"][3])
        q_cmd[:4] = np.clip(q_cmd[:4] + qd * DT, q[:4] - MAX_WINDUP, q[:4] + MAX_WINDUP)
        q_cmd[4] += np.clip(roll - q_cmd[4], -arm.QD_MAX[4] * DT, arm.QD_MAX[4] * DT)
        q_cmd[5] = grip
        q_cmd = arm.clamp_q(q_cmd)
        sim.step(q_cmd)
        stable = stable + 1 if sim.success() else 0
        if stable >= 10:  # cube resting on the goal and released for 0.5 s
            return {"success": True, "t": round(sim.t, 2), "collision": sim.collision_steps > 0}
    return {"success": False, "t": round(sim.t, 2), "collision": sim.collision_steps > 0}


def _eval_job(args):
    return policy_episode(*args)


def evaluate(spec: Dict, n_train: int = 60, n_heldout: int = 30, n_hard: int = 40,
             seed0: int = 50_000, workers: int = 0) -> Dict:
    from src.sim.envs import hard_dynamics, load_env
    jobs, tags = [], []
    for k in range(n_train):
        jobs.append((spec, TRAIN_ENVS[k % 4], seed0 + k, True, None)); tags.append("train_envs")
    for k in range(n_heldout):
        jobs.append((spec, HELDOUT_ENV, seed0 + 1000 + k, True, None)); tags.append("heldout_env")
    for k in range(n_hard):
        e = TRAIN_ENVS[k % 4]
        jobs.append((spec, e, seed0 + 2000 + k, False, hard_dynamics(load_env(e)))); tags.append("hard_dynamics")
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        res = list(pool.map(_eval_job, jobs, chunksize=2))
    out = {}
    for tag in ("train_envs", "heldout_env", "hard_dynamics"):
        r = [x for x, t in zip(res, tags) if t == tag]
        if not r:
            continue
        ok = [x for x in r if x["success"]]
        out[tag] = {"success_pct": 100.0 * len(ok) / len(r), "n": len(r),
                    "collision_pct": 100.0 * np.mean([x["collision"] for x in r]),
                    "time_s_median": float(np.median([x["t"] for x in ok])) if ok else None}
    return out


def expert_reference(n_train: int = 60, n_heldout: int = 30, n_hard: int = 40, seed0: int = 50_000) -> Dict:
    """Success of the scripted expert itself on the SAME evaluation episodes."""
    from src.sim.envs import hard_dynamics, load_env, sample_episode
    from src.sim.expert import run_expert
    from src.sim.physics import ArmSim
    sets = {"train_envs": [(TRAIN_ENVS[k % 4], seed0 + k, True, None) for k in range(n_train)],
            "heldout_env": [(HELDOUT_ENV, seed0 + 1000 + k, True, None) for k in range(n_heldout)],
            "hard_dynamics": [(TRAIN_ENVS[k % 4], seed0 + 2000 + k, False,
                               hard_dynamics(load_env(TRAIN_ENVS[k % 4]))) for k in range(n_hard)]}
    out = {}
    for tag, jobs in sets.items():
        ok = 0
        for e, s, rnd, dyn in jobs:
            ok += run_expert(ArmSim(sample_episode(e, s, randomize=rnd), dynamics=dyn), max_steps=500)["success"]
        out[tag] = {"success_pct": 100.0 * ok / max(len(jobs), 1), "n": len(jobs)}
    return out


# ---------------------------------------------------------------- export
def feature_vectors(n: int = 8, seed: int = 5) -> Dict:
    """(q, obj, goal, goal_is_cube) -> features, and wrist-roll cases, for JS parity tests."""
    from src.sim.expert import roll_for_yaw
    rng = np.random.default_rng(seed)
    cases = []
    for k in range(n):
        q = rng.uniform(arm.Q_MIN, arm.Q_MAX)
        obj = np.r_[rng.uniform([0.1, -0.2, 0.0], [0.3, 0.2, 0.1]), rng.uniform(-0.8, 0.8)]
        goal = rng.uniform([0.1, -0.2, 0.0], [0.3, 0.2, 0.05])
        cases.append({"q": q.round(6).tolist(), "obj": obj.round(6).tolist(), "goal": goal.round(6).tolist(),
                      "goal_is_cube": bool(k % 2),
                      "x": pickplace_features(q.round(6), obj.round(6), goal.round(6), bool(k % 2)).astype(float).round(6).tolist(),
                      "roll": roll_for_yaw(float(q.round(6)[0]), float(obj.round(6)[3]))})
    return {"cases": cases}


def export_spec(model: nn.Module, method: str, X_test: torch.Tensor, extra: Dict) -> Dict:
    from src.robot.train_policy import export_layers
    with torch.no_grad():
        ty = model(X_test)
    return dict({"format": "qi-vla-policy/1", "kind": "pickplace", "in_dim": PP_DIM, "out_dim": PP_OUT,
                 "dt": 0.05, "qd_max": arm.QD_MAX[:4].tolist(), "method": method,
                 "params": sum(p.numel() for p in model.parameters()),
                 "features": "q0..q3, gripper, tip xyz, 5*(obj-tip), 5*(place-tip), 10*(obj_z-0.015), goal_is_cube, 1",
                 "outputs": "qd0..qd3 / qd_max, gripper (>0 open)",
                 "layers": export_layers(model),
                 "test_vectors": {"x": X_test.numpy().round(6).tolist(), "y": ty.numpy().round(6).tolist()},
                 "feature_vectors": feature_vectors()},
                **extra)


# ---------------------------------------------------------------- main
def run(data_dr: str = "data/qivla_sim", data_nodr: str = "data/qivla_sim_nodr", seeds: int = 3,
        steps: int = 6000, heal_steps: int = 1500, chi: int = 16, hidden: int = 256,
        n_train: int = 60, n_heldout: int = 30, n_hard: int = 40,
        out_models: str = "app/public/models", out_json: str = "results/sim_pickplace.json",
        export: bool = True, workers: int = 0, extra_data: Sequence[str] = ()) -> Dict:
    from src.compression.tensor_network import compress_model
    t0 = time.time()
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    X, Y, ep = load_dataset(data_dr)
    has_nodr = Path(data_nodr, "meta", "info.json").exists()
    if has_nodr:
        Xn, Yn, _ = load_dataset(data_nodr)
    rng = np.random.default_rng(0)
    eps = np.unique(ep)
    n_val = max(1, len(eps) // 10) if len(eps) >= 3 else 0   # tiny datasets (smoke tests): no held-out split
    val_eps = set(rng.choice(eps, size=n_val, replace=False).tolist())
    vmask = np.isin(ep, list(val_eps))
    if not vmask.any():
        vmask_eval = np.ones(len(ep), bool)
    else:
        vmask_eval = vmask
    Xtr, Ytr = X[~vmask], Y[~vmask]
    for root in extra_data:  # e.g. demos recorded in the app (scripts/import_app_demos.py)
        Xe, Ye, _ = load_dataset(root)
        Xtr, Ytr = np.concatenate([Xtr, Xe]), np.concatenate([Ytr, Ye])
    Xt, Yt = torch.from_numpy(Xtr), torch.from_numpy(Ytr)
    Xv, Yv = torch.from_numpy(X[vmask_eval]), torch.from_numpy(Y[vmask_eval])
    ev = lambda spec: evaluate(spec, n_train, n_heldout, n_hard, workers=workers)  # noqa: E731

    runs: Dict[str, List[Dict]] = {}
    best_specs = {}
    for s in range(seeds):
        torch.manual_seed(s)
        dense = make_policy(hidden)
        fit(dense, Xt, Yt, steps, 2e-3, seed=s)
        models = {"dense_dr": dense}
        if has_nodr:
            torch.manual_seed(s)
            dn = make_policy(hidden)
            fit(dn, torch.from_numpy(Xn), torch.from_numpy(Yn), steps, 2e-3, seed=s)
            models["dense_nodr"] = dn
        ctrl = copy.deepcopy(dense)
        fit(ctrl, Xt, Yt, heal_steps, 5e-4, seed=s + 100)
        models["healing_ctrl"] = ctrl
        noheal, rep = compress_model(copy.deepcopy(dense), bond_dimension=chi, method="ttns", min_features=64)
        models["ttns_noheal"] = noheal
        healed = copy.deepcopy(noheal)
        fit(healed, Xt, Yt, heal_steps, 5e-4, seed=s + 100)
        models["ttns_dr"] = healed
        for name, m in models.items():
            spec = export_spec(m, name, Xv[:16], {})
            r = {"seed": s, "params": spec["params"], "val_loss": val_loss(m, Xv, Yv), **ev(spec)}
            runs.setdefault(name, []).append(r)
            print(f"[seed {s}] {name:13s} params={spec['params']:>7,} val={r['val_loss']:.4f} "
                  + " ".join(f"{k}={r[k]['success_pct']:.0f}%" for k in ("train_envs", "heldout_env", "hard_dynamics")),
                  flush=True)
            if s == 0 and name in ("dense_dr", "ttns_dr"):
                best_specs[name] = (m, spec)
        compression_ratio = rep.overall_ratio

    def agg(rs: List[Dict]) -> Dict:
        out = {"params": rs[0]["params"], "val_loss_mean": float(np.mean([r["val_loss"] for r in rs]))}
        for tag in ("train_envs", "heldout_env", "hard_dynamics"):
            v = [r[tag]["success_pct"] for r in rs]
            c = [r[tag]["collision_pct"] for r in rs]
            out[tag] = {"success_mean": float(np.mean(v)), "success_std": float(np.std(v)),
                        "per_seed": v, "collision_mean": float(np.mean(c)), "n_per_seed": rs[0][tag]["n"]}
        return out

    summary = {name: agg(rs) for name, rs in runs.items()}
    expert = expert_reference(n_train, n_heldout, n_hard)
    result = {
        "task": "language-grounded pick-and-place (pads + stacking), MuJoCo friction grasp",
        "dataset": {"dr": data_dr, "nodr": data_nodr if has_nodr else None, "extra": list(extra_data),
                    "frames": int(len(X)),
                    "episodes": int(len(eps)), "val_episodes": len(val_eps)},
        "config": {"seeds": seeds, "steps": steps, "heal_steps": heal_steps, "chi": chi, "hidden": hidden,
                   "compressed_layer_ratio": compression_ratio},
        "expert_reference": expert, "summary": summary, "runs": runs,
        "scope": ("Simulation only. Positions of objects are perceived with injected noise, not "
                  "from camera images. 'hard_dynamics' and 'heldout_env' are sim-to-sim transfer "
                  "tests standing in for sim-to-real; no physical robot was used."),
        "wall_s": round(time.time() - t0, 1)}
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    if export:
        Path(out_models).mkdir(parents=True, exist_ok=True)
        for name, (m, spec) in best_specs.items():
            spec = dict(spec, evaluation=summary[name], expert_reference=expert)
            fname = "pickplace_ttns.json" if name == "ttns_dr" else "pickplace_dense.json"
            (Path(out_models) / fname).write_text(json.dumps(spec), encoding="utf-8")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/qivla_sim")
    p.add_argument("--data-nodr", default="data/qivla_sim_nodr")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--heal-steps", type=int, default=1500)
    p.add_argument("--chi", type=int, default=16)
    p.add_argument("--n-train", type=int, default=60)
    p.add_argument("--n-heldout", type=int, default=30)
    p.add_argument("--n-hard", type=int, default=40)
    p.add_argument("--no-export", action="store_true")
    p.add_argument("--extra-data", nargs="*", default=[], help="additional datasets, e.g. data/app_demos_ds")
    a = p.parse_args()
    r = run(a.data, a.data_nodr, a.seeds, a.steps, a.heal_steps, a.chi, n_train=a.n_train,
            n_heldout=a.n_heldout, n_hard=a.n_hard, export=not a.no_export, extra_data=a.extra_data)
    print(json.dumps({"summary": r["summary"], "expert": r["expert_reference"]}, indent=1))


if __name__ == "__main__":
    main()
