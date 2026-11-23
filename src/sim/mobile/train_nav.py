"""Mobile-robot learning pipeline: dataset from the MuJoCo rover, a learned
local navigation controller (dense + TTNS), closed-loop evaluation.

    python -m src.sim.mobile.train_nav collect --out data/qivla_nav --episodes 600
    python -m src.sim.mobile.train_nav collect --out data/qivla_nav_nodr --episodes 600 --no-randomize --seed 100000
    python -m src.sim.mobile.train_nav train --seeds 3      # -> results/sim_nav.json, app/public/models/nav_*.json

Policy: 25 features (nav.nav_features: 18 lidar sectors, pure-pursuit target
from the A* path, goal distance, last command) -> (v, w) / (v_max, w_max).
The global path stays classical (A*); the learned part is the local
controller, as in Nav2's planner/controller split. Deployed with the lidar
barrier shield (nav.shield_vw); an ablation without the shield is reported.

Evaluation sets: train_maps (home/warehouse/office, randomized dynamics),
heldout_map (hospital: unseen layout + slippery floor, slow motors),
hard_dynamics (training maps, parameters outside the DR range),
unknown_obstacles (3 boxes on the route that are NOT in the planner's map;
the costmap learns them from lidar, as in Nav2). Training data has 0-3.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from src.sim.mobile.geometry import map_index, plan_path, grid_for, robot_spec
from src.sim.mobile.nav import NAV_DIM, NAV_OUT, nav_features

FPS = 20


# ---------------------------------------------------------------- collect
def _episode(args) -> Optional[Dict]:
    map_id, seed, randomize, dart, unknown = args
    try:
        from src.sim.mobile.geometry import sample_nav_episode
        from src.sim.mobile.nav import run_nav
        from src.sim.mobile.physics import RoverSim
        ep = sample_nav_episode(map_id, seed, randomize=randomize, unknown=unknown)
        r = run_nav(RoverSim(ep), record=True, dart=dart, rng=np.random.default_rng(seed + 3))
        if not r["success"]:
            return None
        fr = r["frames"]
        return {"map_id": map_id, "seed": seed, "task": ep.instruction, "goal_place": ep.goal_place,
                "unknown": len(ep.unknown),
                "dynamics": ep.dynamics, "randomized": randomize,
                "state": [np.r_[o["pose"], o["vw"]] for o, _, _ in fr],
                "scan": [o["scan"] for o, _, _ in fr],
                "env": [np.r_[o["goal"], a["target"]] for o, a, _ in fr],
                "action": [np.asarray(lab) for _, _, lab in fr]}
    except Exception as e:  # noqa: BLE001
        print(f"nav episode {map_id}/{seed} failed: {e!r}", flush=True)
        return None


def collect(out: str = "data/qivla_nav", maps: Sequence[str] = ("home", "warehouse", "office"),
            episodes: int = 600, randomize: bool = True, dart: float = 0.15, seed0: int = 0,
            workers: int = 0, unknown: int = -3) -> Dict:
    import pyarrow as pa
    import pyarrow.parquet as pq
    t0 = time.time()
    jobs = [(maps[k % len(maps)], seed0 + k, randomize, dart if k % 2 else 0.0, unknown) for k in range(episodes)]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            res = list(pool.map(_episode, jobs, chunksize=4))
    else:
        res = [_episode(j) for j in jobs]
    kept = [r for r in res if r is not None]
    root = Path(out)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    tasks: Dict[str, int] = {}
    cols: Dict[str, List] = {k: [] for k in ("observation.state", "observation.scan", "observation.environment_state",
                                             "action", "map_index", "episode_index", "frame_index", "index",
                                             "task_index", "timestamp", "next.done")}
    eps, idx = [], 0
    all_maps = map_index()["maps"]
    for e, r in enumerate(kept):
        ti = tasks.setdefault(r["task"], len(tasks))
        T = len(r["state"])
        for f in range(T):
            cols["observation.state"].append(np.asarray(r["state"][f], np.float32).tolist())
            cols["observation.scan"].append(np.asarray(r["scan"][f], np.float32).tolist())
            cols["observation.environment_state"].append(np.asarray(r["env"][f], np.float32).tolist())
            cols["action"].append(np.asarray(r["action"][f], np.float32).tolist())
            cols["map_index"].append(all_maps.index(r["map_id"]))
            cols["episode_index"].append(e)
            cols["frame_index"].append(f)
            cols["index"].append(idx + f)
            cols["task_index"].append(ti)
            cols["timestamp"].append(f / FPS)
            cols["next.done"].append(f == T - 1)
        eps.append({"episode_index": e, "map_id": r["map_id"], "tasks": [r["task"]], "goal_place": r["goal_place"],
                    "length": T, "seed": r["seed"], "randomized": r["randomized"], "unknown_obstacles": r["unknown"],
                    "dynamics": json.dumps(r["dynamics"]), "dataset_from_index": idx, "dataset_to_index": idx + T})
        idx += T
    f32 = lambda n: pa.list_(pa.float32(), n)  # noqa: E731
    schema = {"observation.state": f32(5), "observation.scan": f32(robot_spec()["lidar"]["beams"]),
              "observation.environment_state": f32(4), "action": f32(2)}
    pq.write_table(pa.table({k: pa.array(v, type=schema.get(k)) if k in schema else pa.array(v) for k, v in cols.items()}),
                   root / "data" / "chunk-000" / "file-000.parquet")
    pq.write_table(pa.table({"task_index": list(tasks.values()), "task": list(tasks.keys())}), root / "meta" / "tasks.parquet")
    pq.write_table(pa.Table.from_pylist(eps), root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    stats = {k: {"mean": np.asarray(cols[k], np.float32).mean(0).tolist(), "std": np.asarray(cols[k], np.float32).std(0).tolist()}
             for k in schema}
    (root / "meta" / "stats.json").write_text(json.dumps(stats), encoding="utf-8")
    info = {"codebase_version": "v3.0", "robot_type": "qi_vla_rover_diffdrive", "fps": FPS,
            "total_episodes": len(kept), "total_frames": idx, "total_tasks": len(tasks),
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "features": {
                "observation.state": {"dtype": "float32", "shape": [5], "names": ["x", "y", "yaw", "v_cmd_prev", "w_cmd_prev"]},
                "observation.scan": {"dtype": "float32", "shape": [schema["observation.scan"].list_size], "names": None},
                "observation.environment_state": {"dtype": "float32", "shape": [4], "names": ["goal_x", "goal_y", "target_x", "target_y"]},
                "action": {"dtype": "float32", "shape": [2], "names": ["v", "w"]}},
            "qi_vla": {"generator": "src/sim/mobile/train_nav.py", "maps": list(maps), "randomized": randomize,
                       "dart_noise": dart, "attempted": episodes,
                       "physics": "MuJoCo diff-drive: wheel friction/slip, motor lag, latency, lidar + pose noise"}}
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return {"out": str(root), "episodes": len(kept), "attempted": episodes, "frames": idx, "wall_s": round(time.time() - t0, 1)}


def load_nav(root: str):
    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.join(root, "data", "chunk-*", "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no nav dataset under {root!r}")
    t = [pq.read_table(f) for f in files]
    get = lambda c: np.concatenate([np.asarray(x.column(c).to_pylist(), dtype=np.float64) for x in t])  # noqa: E731
    st, sc, env, act = get("observation.state"), get("observation.scan"), get("observation.environment_state"), get("action")
    X = np.stack([nav_features(s[:3], q, e[2:4], e[0:2], s[3:5]) for s, q, e in zip(st, sc, env)])
    R = robot_spec()
    Y = np.stack([act[:, 0] / R["v_max"], act[:, 1] / R["w_max"]], axis=1).astype(np.float32)
    ep = np.concatenate([x.column("episode_index").to_numpy() for x in t])
    return X.astype(np.float32), np.clip(Y, -1, 1), ep


# ---------------------------------------------------------------- model
def make_policy(hidden: int = 128, depth: int = 3) -> nn.Module:
    layers: List[nn.Module] = [nn.Linear(NAV_DIM, hidden), nn.GELU(approximate="tanh")]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), nn.GELU(approximate="tanh")]
    layers += [nn.Linear(hidden, NAV_OUT), nn.Tanh()]
    return nn.Sequential(*layers)


def fit(model, X, Y, steps, lr, seed, batch=2048) -> float:
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
    loss = torch.tensor(0.0)
    for _ in range(steps):
        i = torch.randint(0, X.shape[0], (batch,), generator=g)
        loss = ((model(X[i]) - Y[i]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
    return float(loss)


def export_spec(model, method: str, Xs: torch.Tensor) -> Dict:
    from src.robot.train_policy import export_layers
    from src.sim.mobile.geometry import boxes, load_map, raycast
    from src.sim.mobile.nav import shield_vw
    with torch.no_grad():
        ys = model(Xs)
    R = robot_spec()
    B = boxes(load_map("home"))
    rng = np.random.default_rng(1)
    cases = []
    for k in range(6):
        pose = [float(rng.uniform(0.5, 5.5)), float(rng.uniform(0.5, 4.5)), float(rng.uniform(-3, 3))]
        scan = raycast(B, *pose)
        tgt, goal, vw = rng.uniform(0, 5, 2), rng.uniform(0, 5, 2), [float(rng.uniform(-0.2, 0.26)), float(rng.uniform(-1.5, 1.5))]
        v, w, _ = shield_vw(0.26, 0.5, scan)
        cases.append({"map": "home", "pose": pose, "scan": scan.round(6).tolist(), "target": tgt.round(6).tolist(),
                      "goal": goal.round(6).tolist(), "vw": vw,
                      "x": nav_features(pose, scan, tgt, goal, vw).astype(float).round(6).tolist(),
                      "shield": [v, w]})
    return {"format": "qi-vla-policy/1", "kind": "nav", "in_dim": NAV_DIM, "out_dim": NAV_OUT, "dt": 0.05,
            "qd_max": [R["v_max"], R["w_max"]], "method": method,
            "params": sum(p.numel() for p in model.parameters()),
            "features": "18 lidar sectors/range_max, target dist/sin/cos (robot frame), goal dist/3, v/vmax, w/wmax, 1",
            "outputs": "v/v_max, w/w_max", "layers": export_layers(model),
            "test_vectors": {"x": Xs.numpy().round(6).tolist(), "y": ys.numpy().round(6).tolist()},
            "parity_cases": cases}


# ---------------------------------------------------------------- eval
def _eval_job(args):
    spec, map_id, seed, randomize, dyn, shield, unknown = args[:7]
    max_t = args[7] if len(args) > 7 else 60.0
    from src.robot.policy_runtime import PolicyRuntime
    from src.sim.mobile.geometry import sample_nav_episode
    from src.sim.mobile.nav import run_nav
    from src.sim.mobile.physics import RoverSim
    ep = sample_nav_episode(map_id, seed, randomize=randomize, unknown=unknown)
    sim = RoverSim(ep, dynamics=dyn)
    if spec is None:
        r = run_nav(sim, "expert", shield=shield, max_t=max_t)
    else:
        r = run_nav(sim, "policy", PolicyRuntime(spec), shield=shield, max_t=max_t)
    p = plan_path(grid_for(map_id), ep.start[:2], ep.goal)
    ref = float(np.sum(np.hypot(*np.diff(np.asarray(p), axis=0).T))) if p else np.nan
    r.pop("frames")
    r["spl"] = (ref / max(r["path_len"], ref)) if r["success"] else 0.0
    return r


SETS = ("train_maps", "heldout_map", "hard_dynamics", "unknown_obstacles")


def evaluate(spec, n_train=60, n_heldout=30, n_hard=30, seed0=70_000, shield=True, workers=0,
             n_unknown=30, max_t: float = 60.0) -> Dict:
    from src.sim.mobile.geometry import hard_nav_dynamics, load_map
    idx = map_index()
    tm, hm = idx["train_maps"], idx["heldout_maps"][0]
    jobs, tags = [], []
    for k in range(n_train):
        jobs.append((spec, tm[k % len(tm)], seed0 + k, True, None, shield, 0, max_t)); tags.append("train_maps")
    for k in range(n_heldout):
        jobs.append((spec, hm, seed0 + 1000 + k, True, None, shield, 0, max_t)); tags.append("heldout_map")
    for k in range(n_hard):
        m = tm[k % len(tm)]
        jobs.append((spec, m, seed0 + 2000 + k, False, hard_nav_dynamics(load_map(m)), shield, 0, max_t)); tags.append("hard_dynamics")
    for k in range(n_unknown):
        jobs.append((spec, tm[k % len(tm)], seed0 + 3000 + k, True, None, shield, 3, max_t)); tags.append("unknown_obstacles")
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        res = list(pool.map(_eval_job, jobs, chunksize=2))
    out = {}
    for tag in SETS:
        r = [x for x, t in zip(res, tags) if t == tag]
        if not r:
            continue
        ok = [x for x in r if x["success"]]
        out[tag] = {"success_pct": 100.0 * len(ok) / len(r), "n": len(r),
                    "collision_pct": 100.0 * float(np.mean([x["collision"] for x in r])),
                    "spl": float(np.mean([x["spl"] for x in r])),
                    "shield_pct": float(np.mean([x["shield_pct"] for x in r])),
                    "time_s_median": float(np.median([x["t"] for x in ok])) if ok else None}
    return out


def train(data="data/qivla_nav", data_nodr="data/qivla_nav_nodr", seeds=3, steps=8000, heal_steps=2000,
          chi=16, hidden=128, n_train=60, n_heldout=30, n_hard=30, out_models="app/public/models",
          out_json="results/sim_nav.json", export=True, workers=0, n_unknown=30, max_t=60.0) -> Dict:
    from src.compression.tensor_network import compress_model
    t0 = time.time()
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    X, Y, ep = load_nav(data)
    has_nodr = Path(data_nodr, "meta", "info.json").exists()
    if has_nodr:
        Xn, Yn, _ = load_nav(data_nodr)
    rng = np.random.default_rng(0)
    eps = np.unique(ep)
    val = np.isin(ep, rng.choice(eps, size=max(1, len(eps) // 10), replace=False))
    Xt, Yt, Xv, Yv = (torch.from_numpy(a) for a in (X[~val], Y[~val], X[val], Y[val]))
    ev = lambda spec, shield=True: evaluate(spec, n_train, n_heldout, n_hard, shield=shield, workers=workers,  # noqa: E731
                                            n_unknown=n_unknown, max_t=max_t)
    runs: Dict[str, List[Dict]] = {}
    keep = {}
    rep = None
    for s in range(seeds):
        torch.manual_seed(s)
        dense = make_policy(hidden)
        fit(dense, Xt, Yt, steps, 2e-3, s)
        models = {"dense_dr": dense}
        if has_nodr:
            torch.manual_seed(s)
            dn = make_policy(hidden)
            fit(dn, torch.from_numpy(Xn), torch.from_numpy(Yn), steps, 2e-3, s)
            models["dense_nodr"] = dn
        ctrl = copy.deepcopy(dense)
        fit(ctrl, Xt, Yt, heal_steps, 5e-4, s + 100)
        models["healing_ctrl"] = ctrl
        noheal, rep = compress_model(copy.deepcopy(dense), bond_dimension=chi, method="ttns", min_features=64)
        models["ttns_noheal"] = noheal
        healed = copy.deepcopy(noheal)
        fit(healed, Xt, Yt, heal_steps, 5e-4, s + 100)
        models["ttns_dr"] = healed
        for name, m in models.items():
            spec = export_spec(m, name, Xv[:16])
            with torch.no_grad():
                vl = float(((m(Xv) - Yv) ** 2).mean())
            r = {"seed": s, "params": spec["params"], "val_loss": vl, **ev(spec)}
            runs.setdefault(name, []).append(r)
            if name == "dense_dr":
                rn = {"seed": s, "params": spec["params"], "val_loss": vl, **ev(spec, shield=False)}
                runs.setdefault("dense_dr_noshield", []).append(rn)
            print(f"[seed {s}] {name:13s} params={spec['params']:>6,} val={vl:.4f} "
                  + " ".join(f"{k}={r[k]['success_pct']:.0f}%/coll{r[k]['collision_pct']:.0f}%"
                             for k in SETS), flush=True)
            if s == 0 and name in ("dense_dr", "ttns_dr"):
                keep[name] = spec

    def agg(rs):
        out = {"params": rs[0]["params"], "val_loss_mean": float(np.mean([r["val_loss"] for r in rs]))}
        for tag in SETS:
            g = lambda k: [r[tag][k] for r in rs]  # noqa: E731
            out[tag] = {"success_mean": float(np.mean(g("success_pct"))), "success_std": float(np.std(g("success_pct"))),
                        "collision_mean": float(np.mean(g("collision_pct"))), "spl_mean": float(np.mean(g("spl"))),
                        "shield_mean": float(np.mean(g("shield_pct"))), "per_seed": g("success_pct"),
                        "n_per_seed": rs[0][tag]["n"]}
        return out

    summary = {k: agg(v) for k, v in runs.items()}
    expert = evaluate(None, n_train, n_heldout, n_hard, workers=workers, n_unknown=n_unknown, max_t=max_t)
    result = {"task": "language-goal navigation (A* global path + learned local controller), MuJoCo diff-drive",
              "dataset": {"dr": data, "nodr": data_nodr if has_nodr else None, "frames": int(len(X)), "episodes": int(len(eps))},
              "config": {"seeds": seeds, "steps": steps, "heal_steps": heal_steps, "chi": chi, "hidden": hidden,
                         "compressed_layer_ratio": rep.overall_ratio if rep else None},
              "expert_reference": expert, "summary": summary, "runs": runs,
              "scope": ("Simulation only (MuJoCo physics: friction, slip, motor lag, latency, lidar/pose noise). "
                        "Localization is ground truth + noise (no SLAM). heldout_map and hard_dynamics are "
                        "sim-to-sim transfer tests; no physical robot was used."),
              "wall_s": round(time.time() - t0, 1)}
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    if export:
        for name, spec in keep.items():
            spec = dict(spec, evaluation=summary[name], expert_reference=expert)
            (Path(out_models) / ("nav_ttns.json" if name == "ttns_dr" else "nav_dense.json")).write_text(json.dumps(spec), encoding="utf-8")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--out", default="data/qivla_nav")
    c.add_argument("--maps", default="home,warehouse,office")
    c.add_argument("--episodes", type=int, default=600)
    c.add_argument("--no-randomize", action="store_true")
    c.add_argument("--seed", type=int, default=0)
    t = sub.add_parser("train")
    t.add_argument("--data", default="data/qivla_nav")
    t.add_argument("--data-nodr", default="data/qivla_nav_nodr")
    t.add_argument("--seeds", type=int, default=3)
    t.add_argument("--steps", type=int, default=8000)
    t.add_argument("--heal-steps", type=int, default=2000)
    a = p.parse_args()
    if a.cmd == "collect":
        print(json.dumps(collect(a.out, a.maps.split(","), a.episodes, not a.no_randomize, seed0=a.seed), indent=1))
    else:
        r = train(a.data, a.data_nodr, a.seeds, a.steps, a.heal_steps)
        print(json.dumps({"summary": r["summary"], "expert": r["expert_reference"]}, indent=1))


if __name__ == "__main__":
    main()
