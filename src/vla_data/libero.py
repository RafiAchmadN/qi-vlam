"""Loader for the open LIBERO dataset (LeRobot v3.0 format).

Source: https://huggingface.co/datasets/lerobot/libero (Apache-2.0 card; the
underlying LIBERO demonstrations are from Liu et al., NeurIPS 2023 D&B,
arXiv:2306.03310). 1,693 teleoperated Franka Panda episodes, 273,465 frames
at 10 Hz, 40 natural-language tasks in four suites.

Only the parquet part (proprioceptive state, action, task index — ~21 MB) is
needed here; the 1.9 GB of AV1 camera video is optional and not used by the
default benchmark. Download with `python scripts/download_libero.py`.

    observation.state (8): eef xyz (3), eef axis-angle (3), gripper qpos (2)
    action (7): delta eef xyz (3), delta axis-angle (3), gripper (-1 open / +1 close)
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from src.vla_data.text import encode_many

DEFAULT_ROOT = os.path.join("data", "libero")

# task_index ranges per suite in the lerobot/libero conversion (verified from
# meta/tasks.parquet: 0-9 are the two-step LIBERO-10/Long tasks, 10-19 share one
# kitchen scene with different goals, 20-29 are "pick up <object> ... basket",
# 30-39 are "pick up the black bowl <spatial relation> ... plate").
SUITES = {"libero_10": range(0, 10), "libero_goal": range(10, 20),
          "libero_object": range(20, 30), "libero_spatial": range(30, 40)}


def suite_of(task_index: int) -> str:
    for name, rng in SUITES.items():
        if task_index in rng:
            return name
    return "unknown"


@dataclass
class LiberoData:
    state: np.ndarray          # (N, 8) float32
    action: np.ndarray         # (N, 7) float32
    episode: np.ndarray        # (N,) int64
    task: np.ndarray           # (N,) int64
    frame: np.ndarray          # (N,) int64
    tasks: Dict[int, str]      # task_index -> instruction

    @property
    def n_episodes(self) -> int:
        return int(len(np.unique(self.episode)))


def load_libero(root: str = DEFAULT_ROOT, max_files: Optional[int] = None) -> LiberoData:
    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(root, "data", "chunk-*", "*.parquet")))
    if not files:
        raise FileNotFoundError(
            f"no LIBERO parquet files under {root!r}; run scripts/download_libero.py")
    if max_files:
        files = files[:max_files]
    cols = ["observation.state", "action", "episode_index", "task_index", "frame_index"]
    parts = [pq.read_table(f, columns=cols) for f in files]
    st, ac, ep, tk, fr = [], [], [], [], []
    for t in parts:
        st.append(np.asarray(t.column("observation.state").to_pylist(), dtype=np.float32))
        ac.append(np.asarray(t.column("action").to_pylist(), dtype=np.float32))
        ep.append(t.column("episode_index").to_numpy())
        tk.append(t.column("task_index").to_numpy())
        fr.append(t.column("frame_index").to_numpy())
    rows = pq.read_table(os.path.join(root, "meta", "tasks.parquet")).to_pylist()
    tasks = {}
    for r in rows:
        text = r.get("task") or r.get("__index_level_0__")
        tasks[int(r["task_index"])] = str(text)
    data = LiberoData(np.concatenate(st), np.concatenate(ac), np.concatenate(ep),
                      np.concatenate(tk), np.concatenate(fr), tasks)
    order = np.lexsort((data.frame, data.episode))
    return LiberoData(data.state[order], data.action[order], data.episode[order],
                      data.task[order], data.frame[order], tasks)


def episode_split(data: LiberoData, seed: int = 0, val_frac: float = 0.1,
                  test_frac: float = 0.1) -> Dict[str, np.ndarray]:
    """Split by EPISODE (never by frame) stratified per task, so test frames
    come from demonstrations the model never saw any frame of."""
    rng = np.random.default_rng(seed)
    out: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    for t in sorted(np.unique(data.task)):
        eps = np.unique(data.episode[data.task == t])
        rng.shuffle(eps)
        n_te = max(1, int(round(len(eps) * test_frac)))
        n_va = max(1, int(round(len(eps) * val_frac)))
        out["test"] += eps[:n_te].tolist()
        out["val"] += eps[n_te:n_te + n_va].tolist()
        out["train"] += eps[n_te + n_va:].tolist()
    return {k: np.isin(data.episode, np.array(v)) for k, v in out.items()}


def build_features(data: LiberoData, text_dim: int = 64) -> np.ndarray:
    """Per-frame policy input: [state_t, state_t - state_{t-1}, lang(task)].

    The velocity term uses only the past (state at t-1 of the same episode,
    zero at t=0), so it is causal and available at deployment.
    """
    prev = np.roll(data.state, 1, axis=0)
    first = data.frame == 0
    prev[first] = data.state[first]
    vel = (data.state - prev) * 10.0  # 10 Hz -> per-second scale
    keys = sorted(data.tasks)
    lang_table = encode_many([data.tasks[i] for i in keys], text_dim)
    row = {k: r for r, k in enumerate(keys)}  # task_index need not be contiguous
    lang = lang_table[np.array([row[int(t)] for t in data.task])]
    return np.concatenate([data.state, vel, lang], axis=1).astype(np.float32)
