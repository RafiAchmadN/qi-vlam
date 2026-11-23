"""Merge an extra real-data run (extra model kinds) into an existing benchmark JSON.

    python scripts/merge_real_results.py results/real_benchmark_koch.json results/real_benchmark_koch_extra.json

Both runs must use the same datasets, seeds and step counts (same held-out
episodes). The extra run retrains `cnn_mlp` as the base of the compressed /
bottleneck heads; that re-run is NOT merged over the original but stored under
`reproducibility` so the seed-to-seed agreement of the two runs is visible.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def merge(base_path: str, extra_path: str) -> dict:
    base = json.loads(Path(base_path).read_text(encoding="utf-8"))
    extra = json.loads(Path(extra_path).read_text(encoding="utf-8"))
    if set(base["datasets"]) != set(extra["datasets"]):
        raise ValueError("datasets differ")
    for k in ("seeds", "steps", "heal_steps", "chi", "chunk_k"):
        if base["config"][k] != extra["config"][k]:
            raise ValueError(f"config.{k} differs: {base['config'][k]} vs {extra['config'][k]}")
    shared = sorted(set(base["summary"]) & set(extra["summary"]))
    repro = {}
    for k in shared:
        a, b = base["summary"][k]["motion_nmse"], extra["summary"][k]["motion_nmse"]
        repro[k] = {"original": a, "rerun": b,
                    "per_seed": [[ra["motion_nmse"], rb["motion_nmse"]]
                                 for ra, rb in zip(base["runs"][k], extra["runs"][k])]}
    for k, v in extra["summary"].items():
        if k not in base["summary"]:
            base["summary"][k] = v
            base["runs"][k] = extra["runs"][k]
    base["reproducibility"] = repro
    base["wall_s"] = round(base.get("wall_s", 0) + extra.get("wall_s", 0), 1)
    Path(base_path).write_text(json.dumps(base, indent=2), encoding="utf-8")
    return base


if __name__ == "__main__":
    r = merge(sys.argv[1], sys.argv[2])
    for k, v in sorted(r["summary"].items(), key=lambda kv: kv[1]["motion_nmse"]["mean"]):
        print(f"{k:16s} {v['params']:>8,} {v['storage_kb']:6.0f}KB motion={v['motion_nmse']['mean']:.3f}"
              f"+-{v['motion_nmse']['std']:.3f} lat={v['latency_ms']['mean']:.2f}ms")
    print("reproducibility:", json.dumps(r.get("reproducibility", {}), indent=None)[:400])
