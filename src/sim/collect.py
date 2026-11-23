"""Generate a robot-learning dataset from the MuJoCo environments.

Demonstrations come from the scripted physics expert (`src/sim/expert.py`)
running in randomized episodes of the chosen environments ("places"). The
output follows the LeRobot v3.0 *layout* (parquet + meta/info.json,
meta/tasks.parquet, meta/episodes/..., meta/stats.json) and LeRobot's SO-100
conventions (observation.state = joint positions, action = absolute joint
targets), so it can be read by `src/vla_data/libero.py::load_libero` and
inspected with ordinary parquet tools. It is NOT validated against the
`lerobot` package itself (not installed here) — treat it as LeRobot-style.

    python -m src.sim.collect --out data/qivla_sim --episodes 400 --envs lab_table,kitchen_counter
    python -m src.sim.collect --out data/qivla_sim_nodr --no-randomize
    python -m src.sim.collect --images           # + 160x128 overhead PNG per frame

Columns (per frame, 20 Hz):
  observation.state              float32[6]   measured joints (rad), gripper 0..1
  observation.environment_state  float32[32]  target object xyz+yaw (4), goal support xyz (3),
                                              goal_is_cube (1), 5 object slots xyz+yaw (zeros if
                                              absent), 2 pad slots xy — perceived (noisy)
  action                         float32[6]   joint position targets sent to the servos
  action.velocity                float32[5]   expert joint velocity qd0..3 (rad/s), gripper 0/1
  phase_index, env_index, episode_index, frame_index, index, task_index, timestamp, next.done
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

FPS = 20


def _episode_job(args) -> Optional[Dict]:
    try:
        return _episode(*args)
    except Exception as e:  # one bad episode must not kill a long collection run
        print(f"episode {args[:2]} failed: {e!r}", flush=True)
        return None


def _episode(env_id, seed, randomize, dart, images, max_steps, keep_failed) -> Optional[Dict]:
    from src.sim.envs import env_state_vector, sample_episode
    from src.sim.expert import PHASES, PickPlaceExpert
    from src.sim.physics import ArmSim

    ep = sample_episode(env_id, seed, randomize=randomize)
    sim = ArmSim(ep)
    ex = PickPlaceExpert(travel_z=sim.spec["travel_z"], dart_noise=dart,
                         rng=np.random.default_rng(seed + 7))
    cols: Dict[str, List] = {k: [] for k in ("state", "env", "action", "vel", "phase", "img")}
    renderer = None
    if images:
        import mujoco
        renderer = mujoco.Renderer(sim.model, height=128, width=160)
    for _ in range(max_steps):
        obs = sim.observe()
        a = ex.act(obs, sim.cube_half)
        env_state = env_state_vector(obs["obj"], obs["goal"], ep.goal_kind == "cube", obs["objects"], ep.pads)
        cols["state"].append(obs["q"].astype(np.float32))
        cols["env"].append(env_state.astype(np.float32))
        cols["action"].append(a["q_cmd"].astype(np.float32))
        cols["vel"].append(np.r_[a["qd"], a["gripper"]].astype(np.float32))
        cols["phase"].append(PHASES.index(a["phase"]))
        if renderer is not None:
            renderer.update_scene(sim.data, camera="top")
            from PIL import Image
            buf = io.BytesIO()
            Image.fromarray(renderer.render()).save(buf, format="PNG")
            cols["img"].append(buf.getvalue())
        sim.step(a["q_cmd"])
        if ex.done:
            break
    if renderer is not None:
        renderer.close()
    ok = sim.success()
    if not ok and not keep_failed:
        return None
    return {"env_id": env_id, "seed": seed, "task": ep.instruction, "success": bool(ok),
            "randomized": randomize, "dynamics": ep.dynamics, "config": ep.to_json(),
            "collision_steps": sim.collision_steps, **{k: v for k, v in cols.items()}}


def collect(out: str = "data/qivla_sim", envs: Sequence[str] = ("lab_table", "kitchen_counter",
            "warehouse_shelf", "classroom_desk"), episodes: int = 400, randomize: bool = True,
            dart: float = 0.1, images: bool = False, seed0: int = 0, workers: int = 0,
            max_steps: int = 500, keep_failed: bool = False) -> Dict:
    t0 = time.time()
    jobs = []
    for k in range(episodes):
        env_id = envs[k % len(envs)]
        d = dart if k % 2 else 0.0  # half the episodes clean, half with DART perturbations
        jobs.append((env_id, seed0 + k, randomize, d, images, max_steps, keep_failed))
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_episode_job, jobs, chunksize=4))
    else:
        results = [_episode_job(j) for j in jobs]
    kept = [r for r in results if r is not None]
    extra = {"generator": "src/sim/collect.py", "envs": list(envs), "randomized": randomize,
             "dart_noise": dart, "attempted_episodes": episodes,
             "expert_success_rate": float(np.mean([r is not None and r["success"] for r in results])),
             "physics": "MuJoCo, friction grasp, servo lag/latency/backlash/noise model"}
    frames, n_tasks = write_dataset(kept, out, list(envs), images, extra)
    return {"out": str(out), "episodes_kept": len(kept), "attempted": episodes, "frames": frames,
            "tasks": n_tasks, "wall_s": round(time.time() - t0, 1),
            "per_env_kept": {e: sum(r["env_id"] == e for r in kept) for e in envs}}


def write_dataset(kept: List[Dict], out: str, env_list: List[str], images: bool = False,
                  extra: Optional[Dict] = None):
    """Write episodes (dicts with state/env/action/vel/phase[/img] per frame) as a
    LeRobot-style dataset. Shared by the simulator collector and the app-demo importer."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(out)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    task_ids: Dict[str, int] = {}
    rows: Dict[str, List] = {k: [] for k in (
        "observation.state", "observation.environment_state", "action", "action.velocity",
        "phase_index", "env_index", "episode_index", "frame_index", "index", "task_index",
        "timestamp", "next.done")}
    if images:
        rows["observation.images.top"] = []
    ep_rows: List[Dict] = []
    idx = 0
    for e, r in enumerate(kept):
        ti = task_ids.setdefault(r["task"], len(task_ids))
        T = len(r["state"])
        for f in range(T):
            rows["observation.state"].append(np.asarray(r["state"][f], dtype=np.float32).tolist())
            rows["observation.environment_state"].append(np.asarray(r["env"][f], dtype=np.float32).tolist())
            rows["action"].append(np.asarray(r["action"][f], dtype=np.float32).tolist())
            rows["action.velocity"].append(np.asarray(r["vel"][f], dtype=np.float32).tolist())
            rows["phase_index"].append(r["phase"][f])
            rows["env_index"].append(env_list.index(r["env_id"]) if r["env_id"] in env_list else -1)
            rows["episode_index"].append(e)
            rows["frame_index"].append(f)
            rows["index"].append(idx + f)
            rows["task_index"].append(ti)
            rows["timestamp"].append(f / FPS)
            rows["next.done"].append(f == T - 1)
            if images:
                rows["observation.images.top"].append({"bytes": r["img"][f], "path": None})
        ep_rows.append({"episode_index": e, "env_id": r["env_id"], "tasks": [r["task"]], "length": T,
                        "success": bool(r.get("success", True)), "randomized": bool(r.get("randomized", False)),
                        "seed": int(r.get("seed", -1)), "collision_steps": int(r.get("collision_steps", 0)),
                        "source": r.get("source", "sim_expert"),
                        "dynamics": json.dumps(r.get("dynamics", {})), "config": json.dumps(r.get("config", {})),
                        "dataset_from_index": idx, "dataset_to_index": idx + T})
        idx += T
    f32 = lambda n: pa.list_(pa.float32(), n)  # noqa: E731
    from src.sim.envs import ENV_STATE_DIM, GRASPABLE, PADS
    schema = {"observation.state": f32(6), "observation.environment_state": f32(ENV_STATE_DIM), "action": f32(6),
              "action.velocity": f32(5)}
    table = pa.table({k: pa.array(v, type=schema.get(k)) if k in schema else pa.array(v)
                      for k, v in rows.items()})
    pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")
    pq.write_table(pa.table({"task_index": list(task_ids.values()), "task": list(task_ids.keys())}),
                   root / "meta" / "tasks.parquet")
    pq.write_table(pa.Table.from_pylist(ep_rows), root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    def stats(name: str) -> Dict:
        a = np.asarray(rows[name], dtype=np.float32)
        return {"mean": a.mean(0).tolist(), "std": a.std(0).tolist(), "min": a.min(0).tolist(),
                "max": a.max(0).tolist(), "count": [len(a)]}
    (root / "meta" / "stats.json").write_text(json.dumps(
        {k: stats(k) for k in schema}, indent=1), encoding="utf-8")
    from src.robot import arm
    features = {
        "observation.state": {"dtype": "float32", "shape": [6], "names": arm.JOINT_NAMES},
        "observation.environment_state": {"dtype": "float32", "shape": [ENV_STATE_DIM], "names": [
            "obj_x", "obj_y", "obj_z", "obj_yaw", "goal_x", "goal_y", "goal_z", "goal_is_cube",
            *[f"{c}_{a}" for c in GRASPABLE for a in ("x", "y", "z", "yaw")],
            *[f"{c}_{a}" for c in PADS for a in ("x", "y")]]},
        "action": {"dtype": "float32", "shape": [6], "names": arm.JOINT_NAMES},
        "action.velocity": {"dtype": "float32", "shape": [5],
                            "names": ["qd0", "qd1", "qd2", "qd3", "gripper"]},
        **{k: {"dtype": "int64", "shape": [1], "names": None} for k in (
            "phase_index", "env_index", "episode_index", "frame_index", "index", "task_index")},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "next.done": {"dtype": "bool", "shape": [1], "names": None},
    }
    if images:
        features["observation.images.top"] = {"dtype": "image", "shape": [128, 160, 3],
                                              "names": ["height", "width", "channel"]}
    info = {"codebase_version": "v3.0", "robot_type": "qi_vla_arm_6dof", "fps": FPS,
            "total_episodes": len(kept), "total_frames": idx, "total_tasks": len(task_ids),
            "chunks_size": 1000, "splits": {"train": f"0:{len(kept)}"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "features": features,
            "qi_vla": extra or {}}
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return idx, len(task_ids)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a LeRobot-style dataset from MuJoCo envs")
    p.add_argument("--out", default="data/qivla_sim")
    p.add_argument("--envs", default="lab_table,kitchen_counter,warehouse_shelf,classroom_desk")
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--no-randomize", action="store_true")
    p.add_argument("--dart", type=float, default=0.1)
    p.add_argument("--images", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=0)
    a = p.parse_args()
    print(json.dumps(collect(a.out, a.envs.split(","), a.episodes, not a.no_randomize, a.dart,
                             a.images, a.seed, a.workers), indent=2))


if __name__ == "__main__":
    main()
