"""Run a policy trained on real data — on recorded real episodes, or on a PHYSICAL arm.

1) Replay (no hardware): feed held-out real episodes (camera frames + measured
   joints) through the full control loop at the dataset rate, with action
   chunking + temporal ensembling exactly as on the robot, and report how far
   the commanded actions are from what the human teleoperator did, plus the
   per-decision CPU latency and whether the loop keeps real time.

       python -m src.real.deploy replay --model app/public/models/real_cnn_ttns.json --dataset svla_so101_pickplace
       (--model accepts the exported .json — the same file the app runs — or a .pt checkpoint)

2) Physical robot (SO-100 / SO-101 follower through the `lerobot` package):

       pip install lerobot                     # provides the Feetech motor bus + cameras
       python -m src.real.deploy robot --model models/real/cnn_ttns.pt --dataset svla_so101_pickplace \\
           --robot so101 --port COM5 --camera 0 --dry-run          # observe + print, send nothing
       python -m src.real.deploy robot ... --max-delta 3           # then for real, slowly

   Safety on the robot side of this script (in addition to the servo limits):
   * actions are clamped to the min/max the training data ever commanded
   * per-step change limited to --max-delta units (SO-101: % of calibrated range)
   * --dry-run never sends; Ctrl-C holds the current measured pose and disconnects
   * the camera must be placed like the dataset's (e.g. 'up' = above the table);
     a different viewpoint is out of distribution — expect failures, not magic.
   This path is NOT tested on hardware in this repository.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from src.real.models import temporal_ensemble


class Controller:
    """Stateful closed-loop wrapper: obs -> safe joint targets (dataset units)."""

    def __init__(self, ckpt: Dict, dataset: str, max_delta: float = 5.0, ensemble: bool = True) -> None:
        self.m = ckpt["model"].eval()
        self.size = ckpt.get("size", 64)
        self.names = ckpt["names"]
        if dataset not in self.names:
            raise ValueError(f"model was trained on {self.names}, not {dataset!r}")
        self.n = ckpt["norm"][dataset]
        self.task = torch.nn.functional.one_hot(torch.tensor([self.names.index(dataset)]), len(self.names)).float()
        f = lambda k, s: np.asarray(self.n[k][s], np.float32)  # noqa: E731
        self.sm, self.ss, self.am, self.asd = f("state", "mean"), f("state", "std"), f("action", "mean"), f("action", "std")
        self.amin, self.amax = f("action", "min"), f("action", "max")
        self.max_delta, self.ensemble = max_delta, ensemble
        self.reset()

    def reset(self) -> None:
        self.chunks: List[torch.Tensor] = []
        self.last = None

    def preprocess(self, rgb: np.ndarray) -> torch.Tensor:
        if rgb.shape[0] != self.size or rgb.shape[1] != self.size:
            from PIL import Image
            rgb = np.asarray(Image.fromarray(rgb).resize((self.size, self.size), Image.BILINEAR))
        return torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255.0

    @torch.no_grad()
    def act(self, rgb: np.ndarray, state: np.ndarray) -> Dict:
        t0 = time.perf_counter()
        st = torch.from_numpy(((np.asarray(state, np.float32) - self.sm) / self.ss))[None]
        chunk = self.m(self.preprocess(rgb), st, self.task)[0]
        self.chunks = (self.chunks + [chunk])[-chunk.shape[0]:]
        a_n = temporal_ensemble(self.chunks) if self.ensemble else chunk[0]
        a = a_n.numpy() * self.asd + self.am
        a = np.clip(a, self.amin, self.amax)
        ref = self.last if self.last is not None else np.asarray(state, np.float32)
        a = np.clip(a, ref - self.max_delta, ref + self.max_delta)
        self.last = a
        return {"action": a, "latency_ms": 1000 * (time.perf_counter() - t0)}


def load_checkpoint(model: str) -> Dict:
    """.pt (PyTorch checkpoint from train_real) or .json (exported qi-vla-vision/1, NumPy runtime)."""
    if str(model).endswith(".json"):
        from src.real.runtime import as_checkpoint
        return as_checkpoint(model)
    return torch.load(model, weights_only=False)


def replay(model: str, dataset: str, episodes: int = 5, seed: int = 0, max_delta: float = 1e9) -> Dict:
    from src.real.data import episode_split, prepare
    ckpt = load_checkpoint(model)
    torch.set_num_threads(1)
    d = prepare(dataset, size=ckpt.get("size", 64))
    _, _, test = episode_split(d.episode, seed)
    ctl = Controller(ckpt, dataset, max_delta=max_delta)
    per_ep, lat = [], []
    period = 1.0 / d.fps
    for e in sorted(test)[:episodes]:
        m = d.episode == e
        ctl.reset()
        errs, grips = [], []
        mid = 0.5 * (ctl.amin[5] + ctl.amax[5])
        for img, st, a_true in zip(d.img[m], d.state[m], d.action[m]):
            out = ctl.act(img, st)
            lat.append(out["latency_ms"])
            errs.append(np.abs(out["action"][:5] - a_true[:5]))
            grips.append((out["action"][5] > mid) == (a_true[5] > mid))
        errs = np.asarray(errs)
        per_ep.append({"episode": int(e), "steps": int(m.sum()), "joint_mae": float(errs.mean()),
                       "joint_mae_per_joint": errs.mean(0).round(2).tolist(), "grip_acc": float(np.mean(grips))})
    res = {"model": model, "dataset": dataset, "kind": ckpt["kind"], "episodes": per_ep,
           "joint_mae": float(np.mean([p["joint_mae"] for p in per_ep])),
           "grip_acc": float(np.mean([p["grip_acc"] for p in per_ep])),
           "latency_ms_p50": float(np.percentile(lat, 50)), "latency_ms_p95": float(np.percentile(lat, 95)),
           "control_period_ms": 1000 * period,
           "real_time": bool(np.percentile(lat, 95) < 1000 * period),
           "units": "dataset joint units (SO-101: % of calibrated range); errors vs the human teleoperator's commands"}
    return res


def robot_loop(model: str, dataset: str, robot: str, port: str, camera: int, cam_name: str,
               hz: float, max_delta: float, dry_run: bool, seconds: float) -> None:
    try:
        if robot == "so101":
            from lerobot.robots.so101_follower import SO101Follower as Robot, SO101FollowerConfig as Cfg
        else:
            from lerobot.robots.so100_follower import SO100Follower as Robot, SO100FollowerConfig as Cfg
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    except ImportError as e:  # pragma: no cover - hardware path
        raise SystemExit(f"LeRobot is required for the physical robot ({e}). pip install lerobot") from e
    ckpt = load_checkpoint(model)
    ctl = Controller(ckpt, dataset, max_delta=max_delta)
    cfg = Cfg(port=port, id=f"qi_vla_{robot}",
              cameras={cam_name: OpenCVCameraConfig(index_or_path=camera, width=640, height=480, fps=30)})
    bot = Robot(cfg)
    bot.connect()
    keys = [k for k in bot.action_features]          # e.g. "shoulder_pan.pos", ...
    print(f"connected {robot} on {port}; joints {keys}; {'DRY RUN' if dry_run else 'SENDING ACTIONS'}")
    t_end = time.time() + seconds
    try:
        while time.time() < t_end:
            t0 = time.perf_counter()
            obs = bot.get_observation()
            state = np.asarray([obs[k] for k in keys], np.float32)
            out = ctl.act(np.asarray(obs[cam_name]), state)
            if dry_run:
                print(np.round(out["action"], 1), f"{out['latency_ms']:.1f} ms")
            else:
                bot.send_action({k: float(v) for k, v in zip(keys, out["action"])})
            time.sleep(max(0.0, 1.0 / hz - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        print("stop: holding current pose")
    finally:
        if not dry_run:
            obs = bot.get_observation()
            bot.send_action({k: float(obs[k]) for k in keys})
        bot.disconnect()


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay")
    r.add_argument("--model", default="app/public/models/real_cnn_ttns.json")
    r.add_argument("--dataset", default="svla_so101_pickplace")
    r.add_argument("--episodes", type=int, default=5)
    r.add_argument("--out", default=None)
    b = sub.add_parser("robot")
    b.add_argument("--model", default="app/public/models/real_cnn_ttns.json")
    b.add_argument("--dataset", default="svla_so101_pickplace")
    b.add_argument("--robot", default="so101", choices=["so101", "so100"])
    b.add_argument("--port", required=True)
    b.add_argument("--camera", type=int, default=0)
    b.add_argument("--cam-name", default="up")
    b.add_argument("--hz", type=float, default=10.0)
    b.add_argument("--max-delta", type=float, default=3.0)
    b.add_argument("--seconds", type=float, default=30.0)
    b.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.cmd == "replay":
        res = replay(a.model, a.dataset, a.episodes)
        print(json.dumps({k: v for k, v in res.items() if k != "episodes"}, indent=1))
        if a.out:
            Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    else:
        robot_loop(a.model, a.dataset, a.robot, a.port, a.camera, a.cam_name, a.hz, a.max_delta, a.dry_run, a.seconds)


if __name__ == "__main__":
    main()
