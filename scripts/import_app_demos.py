"""Convert demonstrations recorded in the app (format qi-vla-demos/1) into the
same LeRobot-style dataset the simulator collector writes, so they can be
mixed into training (`python -m src.sim.train_pickplace --data <out>`).

The app records them in the "Data" tab, either downloading a JSON file or
uploading it to the bridge (POST /episodes -> data/app_demos/*.json).

    python scripts/import_app_demos.py data/app_demos/*.json --out data/app_demos_ds

Frame schema written by app/src/data/recorder.js:
  {"q":[6], "q_cmd":[6], "qd":[4], "gripper":0|1, "env_state":[24]}
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sim.collect import write_dataset  # noqa: E402
from src.sim.envs import env_index  # noqa: E402


def load_demos(paths):
    eps = []
    for p in paths:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        if d.get("format") != "qi-vla-demos/1":
            raise ValueError(f"{p}: not a qi-vla-demos/1 file")
        for e in d["episodes"]:
            fr = e["frames"]
            if len(fr) < 5:
                continue
            eps.append({"env_id": e.get("env", "lab_table"), "task": e.get("instruction", ""),
                        "success": bool(e.get("success", True)), "source": f"app:{d.get('mode', 'sim')}",
                        "state": [f["q"] for f in fr], "env": [f["env_state"] for f in fr],
                        "action": [f["q_cmd"] for f in fr],
                        "vel": [list(f["qd"]) + [f["gripper"]] for f in fr],
                        "phase": [-1] * len(fr)})
    return eps


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--out", default="data/app_demos_ds")
    p.add_argument("--only-success", action="store_true")
    a = p.parse_args()
    paths = [f for pat in a.files for f in glob.glob(pat)]
    eps = load_demos(paths)
    if a.only_success:
        eps = [e for e in eps if e["success"]]
    if not eps:
        raise SystemExit("no episodes found")
    frames, tasks = write_dataset(eps, a.out, env_index()["envs"], extra={"generator": "scripts/import_app_demos.py",
                                                                          "sources": paths})
    print(json.dumps({"out": a.out, "episodes": len(eps), "frames": frames, "tasks": tasks}))


if __name__ == "__main__":
    main()
