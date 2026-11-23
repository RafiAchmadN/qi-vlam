"""Download REAL robot datasets (LeRobot v3.0 format) from the Hugging Face Hub.

These are teleoperated demonstrations recorded on physical low-cost arms:
  svla_so101_pickplace   SO-101, 50 eps, 30 fps, cameras up/side      (SmolVLA training data)
  svla_so100_pickplace   SO-100, 50 eps, cameras top/wrist
  svla_so100_stacking    SO-100, 56 eps
  svla_so100_sorting     SO-100, 52 eps
  koch_pick_place_5_lego Koch v1.1, 50 eps, cameras laptop/phone

    python -m src.real.download                              # parquet + main camera of the SO-100/101 sets
    python -m src.real.download --datasets koch_pick_place_5_lego --cameras observation.images.phone

Files go to data/real/<name>/ (gitignored). Only the public HTTP API is used.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

HF = "https://huggingface.co"
DEFAULT = {
    "svla_so101_pickplace": ["observation.images.up"],
    "svla_so100_pickplace": ["observation.images.top"],
    "svla_so100_stacking": ["observation.images.top"],
    "svla_so100_sorting": ["observation.images.top"],
}
ROOT = Path("data/real")


def _json(url: str):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read())


def list_files(repo: str) -> List[Dict]:
    return [f for f in _json(f"{HF}/api/datasets/lerobot/{repo}/tree/main?recursive=true") if f["type"] == "file"]


def fetch(repo: str, path: str, dst: Path, size: Optional[int] = None) -> str:
    if dst.exists() and (size is None or dst.stat().st_size == size):
        return f"skip {path}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    urllib.request.urlretrieve(f"{HF}/datasets/lerobot/{repo}/resolve/main/{path}", tmp)
    os.replace(tmp, dst)
    return f"ok {path} ({(size or 0) / 1e6:.1f} MB)"


def download(datasets: Dict[str, List[str]] = DEFAULT, root: Path = ROOT) -> None:
    jobs = []
    for repo, cams in datasets.items():
        for f in list_files(repo):
            p = f["path"]
            keep = p.startswith("meta/") or p.startswith("data/") or any(p.startswith(f"videos/{c}/") for c in cams)
            if keep:
                jobs.append((repo, p, root / repo / p, f.get("size")))
    total = sum(j[3] or 0 for j in jobs) / 1e6
    print(f"{len(jobs)} files, {total:.0f} MB")
    with ThreadPoolExecutor(8) as pool:
        for msg in pool.map(lambda j: fetch(*j), jobs):
            if not msg.startswith("skip"):
                print(msg, flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default=",".join(DEFAULT))
    p.add_argument("--cameras", default=None, help="comma list; default = main camera per dataset")
    a = p.parse_args()
    sets = {}
    for d in a.datasets.split(","):
        sets[d] = a.cameras.split(",") if a.cameras else DEFAULT.get(d, [])
    download(sets)


if __name__ == "__main__":
    main()
