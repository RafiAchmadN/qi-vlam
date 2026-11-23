"""Environment presets ("places") and domain randomization, shared with the app.

The JSON files in `app/public/envs/` are the single source of truth: the app
renders them with Three.js, and this module turns them into MuJoCo scenes
(`src/sim/mjcf.py`) and randomized episode configurations for data
collection / training (`src/sim/collect.py`).

An *episode config* fixes everything random about one episode: object
placement and yaw, which cube goes where, the instruction text, and the
(optionally randomized) dynamics — servo lag, command latency, backlash,
sensor noise, actuator strength, masses, friction. Randomizing dynamics and
perception noise is the standard sim-to-real recipe (domain randomization,
Tobin et al. 2017; Peng et al. 2018): a policy that works across the whole
range is more likely to work on the one real robot whose parameters we do
not know exactly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ENV_DIR = Path(__file__).resolve().parents[2] / "app" / "public" / "envs"
CUBE = 0.03

# Graspable objects share a 30 mm height (cube 30 mm, cylinder r=15 mm), so the
# grasp/place heights and the policy features are shape-independent.
GRASPABLE = ["blue", "cyan", "green", "orange", "red"]   # fixed env_state slot order
PADS = ["purple", "yellow"]
SHAPE_WORD = {"cube": ("kubus", "cube"), "cylinder": ("tabung", "cylinder")}

# Instruction templates (Indonesian + English), mirrored by app/src/vla/language.js
PLACE_TEMPLATES = [
    "ambil {so} {obj} taruh di pad {goal}",
    "pindahkan {so} {obj} ke pad {goal}",
    "pick up the {obj} {so} and place it on the {goal} pad",
    "put the {obj} {so} on the {goal} pad",
]
STACK_TEMPLATES = [
    "tumpuk {so} {obj} di atas kubus {goal}",
    "letakkan {so} {obj} di atas kubus {goal}",
    "stack the {obj} {so} on the {goal} cube",
    "put the {obj} {so} on top of the {goal} cube",
]
COLOR_ID = {"red": "merah", "green": "hijau", "blue": "biru", "yellow": "kuning", "purple": "ungu",
            "orange": "oranye", "cyan": "tosca"}


def _load(name: str) -> Dict:
    return json.loads((ENV_DIR / name).read_text(encoding="utf-8"))


def env_index() -> Dict:
    return _load("index.json")


def load_env(env_id: str) -> Dict:
    spec = _load(f"{env_id}.json")
    if spec["id"] != env_id:
        raise ValueError(f"env file {env_id}.json has id {spec['id']!r}")
    return spec


def randomization() -> Dict:
    return _load("randomization.json")


@dataclass
class EpisodeConfig:
    env_id: str
    seed: int
    cubes: Dict[str, Tuple[float, float, float]]   # color -> (x, y, yaw)
    pads: Dict[str, Tuple[float, float]]           # color -> (x, y)
    obj: str                                        # cube to move
    goal: str                                       # pad or cube to put it on
    goal_kind: str                                  # "pad" | "cube"
    instruction: str
    dynamics: Dict[str, float] = field(default_factory=dict)
    randomized: bool = False
    shapes: Dict[str, str] = field(default_factory=dict)  # color -> "cube" | "cylinder"

    def shape(self, c: str) -> str:
        return self.shapes.get(c, "cube")

    def to_json(self) -> Dict:
        return {"env_id": self.env_id, "seed": self.seed, "shapes": dict(self.shapes),
                "cubes": {k: list(v) for k, v in self.cubes.items()},
                "pads": {k: list(v) for k, v in self.pads.items()},
                "obj": self.obj, "goal": self.goal, "goal_kind": self.goal_kind,
                "instruction": self.instruction, "dynamics": self.dynamics,
                "randomized": self.randomized}


def _spot(rng: np.random.Generator, spawn: Dict, placed: List[Tuple[float, float]],
          obstacles: List[Dict]) -> Tuple[float, float]:
    for _ in range(60):
        r = rng.uniform(*spawn["r"])
        a = rng.uniform(*spawn["a"])
        p = (r * np.cos(a), r * np.sin(a))
        if any(np.hypot(p[0] - o[0], p[1] - o[1]) < spawn["min_sep"] for o in placed):
            continue
        if any(_inside_xy(p, ob, margin=0.04) for ob in obstacles if not ob.get("visual_only")):
            continue
        placed.append(p)
        return p
    raise RuntimeError("could not place objects; spawn region too crowded")


def _inside_xy(p, ob, margin=0.0) -> bool:
    return (abs(p[0] - ob["pos"][0]) < ob["size"][0] / 2 + margin and
            abs(p[1] - ob["pos"][1]) < ob["size"][1] / 2 + margin)


def sample_dynamics(spec: Dict, rng: np.random.Generator, randomize: bool) -> Dict[str, float]:
    d = dict(spec["dynamics"])
    d.setdefault("link_mass", 1.0)
    d.setdefault("cube_size", 1.0)
    if not randomize:
        return d
    for k, r in randomization()["dynamics"].items():
        base = d.get(k, 1.0)
        if "scale" in r:
            d[k] = float(base * rng.uniform(*r["scale"]))
        else:
            d[k] = float(base + rng.uniform(*r["add"]))
    d["latency_steps"] = int(round(d["latency_steps"]))
    return d


def sample_episode(env_id: str, seed: int, randomize: bool = True,
                   task: Optional[Tuple[str, str]] = None) -> EpisodeConfig:
    """Random scene + task for `env_id`. Deterministic in `seed`."""
    spec = load_env(env_id)
    rng = np.random.default_rng(seed)
    obstacles = spec["obstacles"]
    shapes_all = spec.get("shapes", {})
    lo, hi = spec.get("n_objects", [len(spec["cubes"])] * 2)
    n = int(rng.integers(lo, hi + 1))
    # which objects are on the table this episode (clutter is randomized too)
    required = [c for c in (task or ()) if c in spec["cubes"]]
    rest = [c for c in spec["cubes"] if c not in required]
    chosen = required + [str(c) for c in rng.permutation(rest)[:max(0, n - len(required))]]
    chosen = [c for c in spec["cubes"] if c in chosen]  # stable order
    for attempt in range(80):  # rejection sampling of the whole layout
        placed: List[Tuple[float, float]] = []
        try:
            cubes = {c: (*_spot(rng, spec["spawn"], placed, obstacles), float(rng.uniform(-np.pi / 4, np.pi / 4)))
                     for c in chosen}
            pads = {c: _spot(rng, spec["spawn"], placed, obstacles) for c in spec["pads"]}
            break
        except RuntimeError:
            if attempt % 20 == 19 and len(chosen) > max(2, len(required)):
                drop = [c for c in chosen if c not in required][-1]
                chosen = [c for c in chosen if c != drop]
            continue
    else:
        raise RuntimeError(f"could not lay out env {env_id!r}")
    if task is None:
        obj = str(rng.choice(chosen))
        stack_goals = [c for c in chosen if c != obj and shapes_all.get(c, "cube") == "cube"]
        if rng.random() < 0.7 or not stack_goals:
            goal = str(rng.choice(spec["pads"]))
        else:
            goal = str(rng.choice(stack_goals))
    else:
        obj, goal = task
    kind = "pad" if goal in spec["pads"] else "cube"
    templates = PLACE_TEMPLATES if kind == "pad" else STACK_TEMPLATES
    t = templates[int(rng.integers(len(templates)))]
    so_id, so_en = SHAPE_WORD[shapes_all.get(obj, "cube")]
    if t.split()[0] in ("ambil", "pindahkan", "tumpuk", "letakkan"):
        text = t.format(so=so_id, obj=COLOR_ID[obj], goal=COLOR_ID[goal])
    else:
        text = t.format(so=so_en, obj=obj, goal=goal)
    return EpisodeConfig(env_id=env_id, seed=seed, cubes=cubes, pads=pads, obj=obj, goal=goal,
                         goal_kind=kind, instruction=text,
                         dynamics=sample_dynamics(spec, rng, randomize), randomized=randomize,
                         shapes={c: shapes_all.get(c, "cube") for c in cubes})


ENV_STATE_DIM = 8 + 4 * len(GRASPABLE) + 2 * len(PADS)   # 32


def env_state_vector(obj, goal, goal_is_cube: bool, objects: Dict, pads: Dict) -> np.ndarray:
    """Fixed-layout environment state (dataset column observation.environment_state):
    target obj xyz yaw (4) | goal support xyz (3) | goal_is_cube (1) |
    GRASPABLE slots xyz yaw (5x4, zeros if absent) | PADS slots xy (2x2).
    Mirrored by app/src/data/recorder.js::envState."""
    out = [np.asarray(obj, dtype=np.float64)[:4], np.asarray(goal, dtype=np.float64)[:3],
           [1.0 if goal_is_cube else 0.0]]
    for c in GRASPABLE:
        out.append(np.asarray(objects[c], dtype=np.float64)[:4] if c in objects else np.zeros(4))
    for c in PADS:
        out.append(np.asarray(pads[c], dtype=np.float64)[:2] if c in pads else np.zeros(2))
    return np.concatenate(out).astype(np.float32)


def hard_dynamics(spec: Dict) -> Dict[str, float]:
    """A fixed 'worn real robot' parameter set OUTSIDE the training DR range of
    most envs (used as a sim-to-real stress test, not for training)."""
    d = dict(spec["dynamics"], link_mass=1.2, cube_size=1.0)
    d.update(servo_tau=0.22, latency_steps=3, backlash=0.008, q_noise=0.004, obj_noise=0.005,
             kp_scale=0.75, cube_mass=0.04, friction=0.6)
    return d
