"""Download the parquet (state/action/language) part of lerobot/libero.

~21 MB, 377 parquet files + metadata. Videos (1.9 GB, AV1) are skipped unless
--videos is passed. Uses only the public Hugging Face resolve endpoint.

    python scripts/download_libero.py [--root data/libero] [--videos]
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

REPO = "lerobot/libero"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main"
RAW = f"https://huggingface.co/datasets/{REPO}/resolve/main"


def _list(path: str):
    with urllib.request.urlopen(f"{API}/{path}?recursive=true") as r:
        return [x["path"] for x in json.load(r) if x["type"] == "file"]


def _get(rel: str, root: str) -> None:
    dst = os.path.join(root, rel)
    if os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    urllib.request.urlretrieve(f"{RAW}/{rel}", dst + ".part")
    os.replace(dst + ".part", dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join("data", "libero"))
    ap.add_argument("--videos", action="store_true")
    a = ap.parse_args()
    files = _list("meta") + _list("data") + (_list("videos") if a.videos else [])
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(lambda f: _get(f, a.root), files))
    print(f"downloaded {len(files)} files to {a.root}")


if __name__ == "__main__":
    main()
