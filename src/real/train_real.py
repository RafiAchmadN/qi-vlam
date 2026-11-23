"""Train and compare policies on REAL robot demonstrations (SO-100 / SO-101).

    python -m src.real.download                  # real data (LeRobot hub)
    python -m src.real.train_real --seeds 3      # -> results/real_benchmark.json, models/real/*.pt

One multi-task policy is trained on all available real datasets (task given
as a one-hot; each dataset is normalized with its own statistics), split
PER EPISODE (held-out test episodes never seen in training), 3 seeds.

Metrics on held-out real episodes (open-loop, like LIBERO offline eval):
  nmse_first   normalized MSE of the next action      (1.0 = predicting the mean)
  nmse_chunk   normalized MSE over the whole K-step chunk
  motion_nmse  error of the predicted DISPLACEMENT from the current pose over the chunk,
               normalized so 1.0 = "don't move" (the honest metric: next-action nMSE is
               dominated by pose autocorrelation and flatters camera-blind models)
  joint_mae    mean |error| of the 5 arm joints in the dataset's own units
               (SO-101: % of the calibrated range, i.e. -100..100)
  grip_acc     open/closed agreement of the gripper command
plus params, storage, single-thread CPU latency per decision and train time.
Closed-loop success needs the physical robot (src/real/deploy.py).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from src.real.data import ROBOT_OF, available, chunk_actions, episode_split, prepare, stats
from src.real.models import HeadPolicy, Policy, augment, compress_head, count, int8_head

K = 10
MODELS = ("state_mlp", "cnn_mlp", "act_tf", "healing_ctrl", "cnn_ttns", "cnn_mps", "cnn_ttns_noheal",
          "cnn_int8", "head_tanh", "head_surrogate", "head_pqc")


class Bundle:
    """All datasets concatenated, normalized per dataset, with episode splits."""

    def __init__(self, names: Sequence[str], seed: int, size: int = 64) -> None:
        self.names = list(names)
        self.norm: Dict[str, Dict] = {}
        parts = []
        for ti, n in enumerate(self.names):
            d = prepare(n, size=size)
            tr, va, te = episode_split(d.episode, seed)
            ns, na = stats(d.state), stats(d.action)
            self.norm[n] = {"state": ns, "action": na, "task": d.task, "robot": ROBOT_OF.get(n, "unknown"),
                            "task_index": ti, "fps": d.fps}
            sm, ss = np.asarray(ns["mean"], np.float32), np.asarray(ns["std"], np.float32)
            am, asd = np.asarray(na["mean"], np.float32), np.asarray(na["std"], np.float32)
            split = np.array([0 if e in tr else 1 if e in va else 2 for e in d.episode])
            parts.append(dict(img=d.img, state=(d.state - sm) / ss, act=(chunk_actions(d.action, d.episode, K) - am) / asd,
                              raw_act=d.action, raw_state=d.state, task=np.full(len(d.state), ti), split=split,
                              raw_chunk=chunk_actions(d.action, d.episode, K),
                              episode=d.episode + 100_000 * ti, asd=np.broadcast_to(asd, d.action.shape)))
        cat = lambda k: np.concatenate([p[k] for p in parts])  # noqa: E731
        self.img, self.state, self.act = cat("img"), cat("state").astype(np.float32), cat("act").astype(np.float32)
        self.raw_act, self.task, self.split, self.episode = cat("raw_act"), cat("task"), cat("split"), cat("episode")
        self.raw_state, self.raw_chunk = cat("raw_state"), cat("raw_chunk")
        self.asd = cat("asd")
        self.T = len(self.names)

    def tensors(self, idx: np.ndarray):
        img = torch.from_numpy(self.img[idx]).permute(0, 3, 1, 2).float() / 255.0
        task = torch.nn.functional.one_hot(torch.from_numpy(self.task[idx]).long(), self.T).float()
        return img, torch.from_numpy(self.state[idx]), task, torch.from_numpy(self.act[idx])


def train(model, B: Bundle, steps: int, lr: float, seed: int, batch: int = 128, aug: bool = True,
          params=None) -> float:
    g = torch.Generator().manual_seed(seed)
    tr = np.nonzero(B.split == 0)[0]
    opt = torch.optim.AdamW(params if params is not None else model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
    rng = np.random.default_rng(seed)
    loss = torch.tensor(0.0)
    model.train()
    for _ in range(steps):
        idx = rng.choice(tr, batch)
        img, st, tk, y = B.tensors(idx)
        if aug:
            img = augment(img, g)
        st = st + 0.01 * torch.randn(st.shape, generator=g)      # small proprioceptive noise
        loss = ((model(img, st, tk) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()
    model.eval()
    return float(loss)


@torch.no_grad()
def evaluate(model, B: Bundle, split: int = 2) -> Dict:
    idx = np.nonzero(B.split == split)[0]
    preds = []
    for s in range(0, len(idx), 512):
        img, st, tk, _ = B.tensors(idx[s:s + 512])
        preds.append(model(img, st, tk).numpy())
    P = np.concatenate(preds)
    Y = B.act[idx]
    out = {"n_frames": int(len(idx))}
    per = {}
    for ti, n in enumerate(B.names):
        m = B.task[idx] == ti
        if not m.any():
            continue
        e2 = (P[m] - Y[m]) ** 2
        asd = np.asarray(B.norm[n]["action"]["std"])
        amean = np.asarray(B.norm[n]["action"]["mean"])
        raw_p = P[m, 0] * asd + amean
        raw_y = Y[m, 0] * asd + amean
        mid = 0.5 * (np.asarray(B.norm[n]["action"]["min"])[5] + np.asarray(B.norm[n]["action"]["max"])[5])
        # motion error: predicted vs true DISPLACEMENT from the current pose over the chunk (arm joints),
        # normalized so that 1.0 = "stay where you are" (copying the current state)
        chunk_p = P[m] * asd + amean
        cur = B.raw_state[idx][m][:, None, :5]
        true_d = B.raw_chunk[idx][m][..., :5] - cur
        motion = float(((chunk_p[..., :5] - cur - true_d) ** 2).mean() / max((true_d ** 2).mean(), 1e-9))
        per[n] = {"nmse_first": float(e2[:, 0].mean()), "nmse_chunk": float(e2.mean()), "motion_nmse": motion,
                  "joint_mae": float(np.abs(raw_p[:, :5] - raw_y[:, :5]).mean()),
                  "grip_acc": float(((raw_p[:, 5] > mid) == (raw_y[:, 5] > mid)).mean()),
                  "frames": int(m.sum())}
    out["per_dataset"] = per
    for k in ("nmse_first", "nmse_chunk", "motion_nmse", "joint_mae", "grip_acc"):
        out[k] = float(np.mean([v[k] for v in per.values()]))
    return out


def latency_ms(model, B: Bundle, n: int = 50) -> float:
    torch.set_num_threads(1)
    img, st, tk, _ = B.tensors(np.array([0]))
    with torch.no_grad():
        for _ in range(3):
            model(img, st, tk)
        t0 = time.perf_counter()
        for _ in range(n):
            model(img, st, tk)
    return 1000 * (time.perf_counter() - t0) / n


def run(names: Sequence[str] = (), seeds: int = 3, steps: int = 4000, heal_steps: int = 800, chi: int = 16,
        models: Sequence[str] = MODELS, head_steps: int = 500, out_json: str = "results/real_benchmark.json",
        out_models: str = "models/real") -> Dict:
    from src.training.rl_alignment import _PENNYLANE
    t_all = time.time()
    names = list(names) or available()
    if not names:
        raise FileNotFoundError("no real datasets under data/real — run python -m src.real.download")
    models = [m for m in models if m != "head_pqc" or _PENNYLANE]
    runs: Dict[str, List[Dict]] = {}
    for s in range(seeds):
        B = Bundle(names, seed=s)
        torch.manual_seed(s)
        nthreads = max(1, (os.cpu_count() or 2) - 1)
        trained: Dict[str, torch.nn.Module] = {}

        def record(name, m, t_train):
            torch.set_num_threads(nthreads)
            r = {"seed": s, "train_s": round(t_train, 1), **count(m), "latency_ms": latency_ms(m, B), **evaluate(m, B)}
            torch.set_num_threads(nthreads)
            runs.setdefault(name, []).append(r)
            print(f"[seed {s}] {name:15s} params={r['params']:>8,} nmse_first={r['nmse_first']:.3f} "
                  f"chunk={r['nmse_chunk']:.3f} motion={r['motion_nmse']:.3f} joint_mae={r['joint_mae']:.2f} grip={r['grip_acc']:.3f} "
                  f"lat={r['latency_ms']:.2f}ms", flush=True)
            Path(out_json).parent.mkdir(parents=True, exist_ok=True)
            Path(out_json).write_text(json.dumps({"partial": True, "runs": runs}, indent=1), encoding="utf-8")

        torch.set_num_threads(nthreads)
        for kind, lr, st in (("state_mlp", 1e-3, steps), ("cnn_mlp", 1e-3, steps), ("act_tf", 5e-4, steps)):
            if kind not in models and not (kind == "cnn_mlp"):
                continue
            torch.manual_seed(s)
            m = Policy(kind, B.T, K)
            t0 = time.time()
            train(m, B, st, lr, seed=s, aug=kind != "state_mlp")
            trained[kind] = m
            if kind in models:
                record(kind, m, time.time() - t0)
        base = trained["cnn_mlp"]
        enc_params = [p for n_, p in base.named_parameters() if n_.startswith("enc.")]
        for kind in models:
            if kind in trained or kind in ("state_mlp", "act_tf"):
                continue
            t0 = time.time()
            if kind == "healing_ctrl":
                m = copy.deepcopy(base)
                train(m, B, heal_steps, 3e-4, seed=s + 100, params=list(m.head.parameters()))
            elif kind in ("cnn_ttns", "cnn_mps", "cnn_ttns_noheal"):
                m = compress_head(base, "mps" if kind == "cnn_mps" else "ttns", chi)
                if kind != "cnn_ttns_noheal":
                    train(m, B, heal_steps, 3e-4, seed=s + 100, params=list(m.head.parameters()))
            elif kind == "cnn_int8":
                m = int8_head(base)
            elif kind.startswith("head_"):
                m = HeadPolicy(base, kind[5:])
                train(m, B, head_steps, 3e-3, seed=s + 200, batch=64 if kind == "head_pqc" else 128,
                      params=[p for p in m.parameters() if p.requires_grad])
            else:
                continue
            trained[kind] = m
            record(kind, m, time.time() - t0)
        del enc_params
        if s == 0:
            Path(out_models).mkdir(parents=True, exist_ok=True)
            for kind in ("cnn_mlp", "cnn_ttns", "act_tf", "state_mlp"):
                if kind in trained:
                    torch.save({"kind": kind, "k": K, "names": names, "norm": B.norm, "size": 64,
                                "model": trained[kind]}, Path(out_models) / f"{kind}.pt")
    summary = {}
    for name, rs in runs.items():
        e = {k: rs[0][k] for k in ("params", "storage_kb")}
        for k in ("nmse_first", "nmse_chunk", "motion_nmse", "joint_mae", "grip_acc", "latency_ms", "train_s"):
            v = [r[k] for r in rs]
            e[k] = {"mean": float(np.mean(v)), "std": float(np.std(v))}
        e["per_dataset"] = {n: {k: float(np.mean([r["per_dataset"][n][k] for r in rs]))
                                for k in ("nmse_first", "nmse_chunk", "motion_nmse", "joint_mae", "grip_acc")}
                            for n in rs[0]["per_dataset"]}
        summary[name] = e
    B0 = Bundle(names, seed=0)
    result = {"datasets": {n: {"robot": B0.norm[n]["robot"], "task": B0.norm[n]["task"],
                               "frames": int((B0.task == i).sum()),
                               "episodes": int(len(np.unique(B0.episode[B0.task == i])))} for i, n in enumerate(names)},
              "config": {"seeds": seeds, "steps": steps, "heal_steps": heal_steps, "chi": chi, "chunk_k": K,
                         "image": "64x64 RGB main camera, 10 Hz", "split": "per episode 75/10/15"},
              "summary": summary, "runs": runs,
              "scope": ("REAL teleoperated demonstrations from physical SO-100/SO-101 arms (LeRobot hub). Offline, "
                        "open-loop action prediction on held-out episodes; closed-loop task success requires the "
                        "physical robot (src/real/deploy.py). PQC simulated noiselessly."),
              "wall_s": round(time.time() - t_all, 1)}
    Path(out_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--heal-steps", type=int, default=800)
    p.add_argument("--models", default=",".join(MODELS))
    a = p.parse_args()
    r = run([d for d in a.datasets.split(",") if d], a.seeds, a.steps, a.heal_steps, models=a.models.split(","))
    print(json.dumps({k: {m: v[m]["mean"] for m in ("motion_nmse", "nmse_first", "joint_mae", "grip_acc")} for k, v in r["summary"].items()}, indent=1))


if __name__ == "__main__":
    main()
