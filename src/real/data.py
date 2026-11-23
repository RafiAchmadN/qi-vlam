"""Real-robot datasets (LeRobot v3.0) -> aligned (image, state, action) arrays.

Decodes the AV1 camera video with PyAV (libdav1d), aligns every kept parquet
row to its video frame through the episode's `videos/<cam>/from_timestamp`,
subsamples 30 fps -> `hz`, resizes to `size`x`size` RGB, and caches the
result as `data/real/<name>/cache_<cam>_<size>_<hz>.npz`.

Units are whatever the dataset recorded (SO-101 sets from recent LeRobot:
joints normalized to [-100, 100], gripper [0, 100]; older SO-100 sets:
degrees-like). Models normalize per dataset with the stored mean/std, and
the deployment code clamps to each dataset's recorded min/max.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

ROOT = Path("data/real")
MAIN_CAMERA = {"svla_so101_pickplace": "observation.images.up", "svla_so100_pickplace": "observation.images.top",
               "svla_so100_stacking": "observation.images.top", "svla_so100_sorting": "observation.images.top",
               "koch_pick_place_5_lego": "observation.images.phone"}
ROBOT_OF = {"svla_so101_pickplace": "so101", "svla_so100_pickplace": "so100", "svla_so100_stacking": "so100",
            "svla_so100_sorting": "so100", "koch_pick_place_5_lego": "koch"}


@dataclass
class RealData:
    name: str
    img: np.ndarray        # (N, S, S, 3) uint8
    state: np.ndarray      # (N, 6) float32
    action: np.ndarray     # (N, 6) float32
    episode: np.ndarray    # (N,) int
    frame: np.ndarray      # (N,) int, index within the subsampled episode
    task: str
    fps: float

    @property
    def episodes(self) -> np.ndarray:
        return np.unique(self.episode)


def _read_parquet_dir(d: Path):
    import pyarrow.parquet as pq
    import pyarrow as pa
    files = sorted(glob.glob(str(d / "*" / "*.parquet")))
    return pa.concat_tables([pq.read_table(f) for f in files])


def prepare(name: str, camera: Optional[str] = None, size: int = 64, hz: int = 10,
            root: Path = ROOT, force: bool = False) -> RealData:
    camera = camera or MAIN_CAMERA[name]
    base = root / name
    cache = base / f"cache_{camera.split('.')[-1]}_{size}_{hz}.npz"
    info = json.loads((base / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = float(info["fps"])
    import pyarrow.parquet as pq
    tasks = pq.read_table(base / "meta" / "tasks.parquet").to_pylist()
    task = str(tasks[0].get("task") or tasks[0].get("__index_level_0__"))
    if cache.exists() and not force:
        z = np.load(cache)
        return RealData(name, z["img"], z["state"], z["action"], z["episode"], z["frame"], task, float(hz))
    import av
    step = max(1, int(round(fps / hz)))
    data = _read_parquet_dir(base / "data")
    st = np.asarray(data.column("observation.state").to_pylist(), np.float32)
    ac = np.asarray(data.column("action").to_pylist(), np.float32)
    ep = data.column("episode_index").to_numpy()
    fi = data.column("frame_index").to_numpy()
    ts = data.column("timestamp").to_numpy()
    keep = fi % step == 0
    eps = _read_parquet_dir(base / "meta" / "episodes").to_pylist()
    vkey = f"videos/{camera}"
    by_ep = {e["episode_index"]: e for e in eps}
    # which global video frame each kept row needs, grouped by video file
    need: Dict[str, Dict[int, List[int]]] = {}
    rows = np.nonzero(keep)[0]
    for r in rows:
        e = by_ep[int(ep[r])]
        path = base / info["video_path"].format(video_key=camera, chunk_index=e[f"{vkey}/chunk_index"],
                                                file_index=e[f"{vkey}/file_index"])
        f = int(round((e[f"{vkey}/from_timestamp"] + float(ts[r])) * fps))
        need.setdefault(str(path), {}).setdefault(f, []).append(int(r))
    img = np.zeros((len(st), size, size, 3), np.uint8)
    got = np.zeros(len(st), bool)
    for path, frames in need.items():
        with av.open(path) as c:
            s = c.streams.video[0]
            s.thread_type = "AUTO"
            last = max(frames)
            for fr in c.decode(s):
                k = int(round(float(fr.time) * fps))
                if k in frames:
                    a = fr.to_ndarray(width=size, height=size, format="rgb24")
                    for r in frames[k]:
                        img[r] = a
                        got[r] = True
                if k > last:
                    break
    ok = keep & got
    sub_frame = (fi // step).astype(np.int64)
    out = RealData(name, img[ok], st[ok], ac[ok], ep[ok].astype(np.int64), sub_frame[ok], task, float(hz))
    np.savez_compressed(cache, img=out.img, state=out.state, action=out.action, episode=out.episode, frame=out.frame)
    print(f"{name}: {ok.sum()} frames ({(keep & ~got).sum()} without video frame) from {len(out.episodes)} episodes", flush=True)
    return out


def episode_split(eps: np.ndarray, seed: int, val: float = 0.1, test: float = 0.15):
    rng = np.random.default_rng(seed)
    e = rng.permutation(np.unique(eps))
    nt, nv = max(1, int(round(len(e) * test))), max(1, int(round(len(e) * val)))
    return set(e[nv + nt:].tolist()), set(e[:nv].tolist()), set(e[nv:nv + nt].tolist())


def chunk_actions(action: np.ndarray, episode: np.ndarray, k: int) -> np.ndarray:
    """(N, k, 6): actions t..t+k-1 within the same episode (last one repeated at the end)."""
    n = len(action)
    out = np.repeat(action[:, None], k, axis=1).copy()
    for j in range(1, k):
        idx = np.minimum(np.arange(n) + j, n - 1)
        same = episode[idx] == episode
        out[:, j] = np.where(same[:, None], action[idx], out[:, j - 1])
    return out


def stats(x: np.ndarray) -> Dict[str, List[float]]:
    return {"mean": x.mean(0).tolist(), "std": (x.std(0) + 1e-3).tolist(), "min": x.min(0).tolist(), "max": x.max(0).tolist()}


def available(names: Sequence[str] = tuple(MAIN_CAMERA), root: Path = ROOT) -> List[str]:
    return [n for n in names if (root / n / "meta" / "info.json").exists()
            and glob.glob(str(root / n / "videos" / MAIN_CAMERA[n] / "*" / "*.mp4"))]


if __name__ == "__main__":
    for n in available():
        d = prepare(n)
        print(n, d.img.shape, d.task, len(d.episodes), "episodes", os.path.getsize(next((ROOT / n).glob("cache_*.npz"))) // 1_000_000, "MB")
