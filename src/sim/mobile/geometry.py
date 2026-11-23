"""2-D world model of a mobile-robot map: boxes, lidar ray casting, occupancy
grid, A* planning, and randomized navigation episodes.

Maps live in `app/public/maps/*.json` (shared with the app's rover mode, which
re-implements `raycast`, `plan_path` and `nav_features` in
`app/src/rover/rover.js` — parity-tested). Coordinates: metres, origin at
the bottom-left corner, x right, y up, yaw counter-clockwise from +x.
"""

from __future__ import annotations

import heapq
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

MAP_DIR = Path(__file__).resolve().parents[3] / "app" / "public" / "maps"
GRID_RES = 0.05


def _load(name: str) -> Dict:
    return json.loads((MAP_DIR / name).read_text(encoding="utf-8"))


def map_index() -> Dict:
    return _load("index.json")


def robot_spec() -> Dict:
    return map_index()["robot"]


def load_map(map_id: str) -> Dict:
    m = _load(f"{map_id}.json")
    if m["id"] != map_id:
        raise ValueError(f"map file {map_id}.json has id {m['id']!r}")
    return m


def boxes(m: Dict) -> np.ndarray:
    """(N, 4) array of [xmin, ymin, xmax, ymax] for walls + obstacles."""
    out = []
    for b in m["walls"] + m["obstacles"]:
        (x, y), (sx, sy) = b["pos"], b["size"]
        out.append([x - sx / 2, y - sy / 2, x + sx / 2, y + sy / 2])
    return np.asarray(out, dtype=np.float64)


def beam_angles(n: int = 36) -> np.ndarray:
    return np.arange(n) * (2 * np.pi / n)


def raycast(B: np.ndarray, x: float, y: float, yaw: float, n: int = 36,
            rmax: float = 3.5) -> np.ndarray:
    """Distances along n beams (robot frame angles beam_angles(n)) to the
    nearest box face, capped at rmax. Slab method, vectorised over boxes."""
    ang = yaw + beam_angles(n)
    dx, dy = np.cos(ang)[:, None], np.sin(ang)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        tx1 = (B[None, :, 0] - x) / dx
        tx2 = (B[None, :, 2] - x) / dx
        ty1 = (B[None, :, 1] - y) / dy
        ty2 = (B[None, :, 3] - y) / dy
    tmin = np.maximum(np.minimum(tx1, tx2), np.minimum(ty1, ty2))
    tmax = np.minimum(np.maximum(tx1, tx2), np.maximum(ty1, ty2))
    hit = (tmax >= np.maximum(tmin, 0.0)) & np.isfinite(tmin)
    d = np.where(hit, np.where(tmin > 0, tmin, 0.0), np.inf)
    return np.minimum(d.min(axis=1), rmax)


def clearance(B: np.ndarray, x: float, y: float) -> float:
    """Distance from a point to the nearest box (0 if inside)."""
    dx = np.maximum(np.maximum(B[:, 0] - x, 0.0), x - B[:, 2])
    dy = np.maximum(np.maximum(B[:, 1] - y, 0.0), y - B[:, 3])
    return float(np.sqrt(dx ** 2 + dy ** 2).min())


@dataclass
class Grid:
    occ: np.ndarray          # (ny, nx) bool, inflated obstacles
    cost: np.ndarray         # (ny, nx) extra cost near obstacles
    res: float
    size: Tuple[float, float]

    def cell(self, x: float, y: float) -> Tuple[int, int]:
        return (int(np.clip(y / self.res, 0, self.occ.shape[0] - 1)),
                int(np.clip(x / self.res, 0, self.occ.shape[1] - 1)))

    def point(self, c: Tuple[int, int]) -> Tuple[float, float]:
        return ((c[1] + 0.5) * self.res, (c[0] + 0.5) * self.res)


def build_grid(m: Dict, inflate: float, res: float = GRID_RES) -> Grid:
    B = boxes(m)
    w, h = m["size"]
    xs = (np.arange(int(round(w / res))) + 0.5) * res
    ys = (np.arange(int(round(h / res))) + 0.5) * res
    X, Y = np.meshgrid(xs, ys)
    dx = np.maximum(np.maximum(B[:, 0, None, None] - X, 0), X - B[:, 2, None, None])
    dy = np.maximum(np.maximum(B[:, 1, None, None] - Y, 0), Y - B[:, 3, None, None])
    dist = np.sqrt(dx ** 2 + dy ** 2).min(axis=0)
    occ = dist < inflate
    cost = np.clip(0.35 - dist, 0, None) * 8.0   # prefer the middle of corridors
    return Grid(occ=occ, cost=cost, res=res, size=(w, h))


_GRID_CACHE: Dict[Tuple[str, float], Grid] = {}


def grid_for(map_id: str, inflate: Optional[float] = None) -> Grid:
    inflate = robot_spec()["radius"] + 0.06 if inflate is None else inflate
    key = (map_id, round(inflate, 4))
    if key not in _GRID_CACHE:
        _GRID_CACHE[key] = build_grid(load_map(map_id), inflate)
    return _GRID_CACHE[key]


def _nearest_free(g: Grid, c: Tuple[int, int]) -> Tuple[int, int]:
    if not g.occ[c]:
        return c
    free = np.argwhere(~g.occ)
    k = np.argmin(((free - np.asarray(c)) ** 2).sum(1))
    return tuple(free[k])


def plan_path(g: Grid, start: Sequence[float], goal: Sequence[float]) -> Optional[List[Tuple[float, float]]]:
    """8-connected A* with a clearance cost, then line-of-sight shortcutting.
    Returns waypoints (world coords) from start to goal, or None."""
    s, t = _nearest_free(g, g.cell(*start)), _nearest_free(g, g.cell(*goal))
    ny, nx = g.occ.shape
    moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]
    gs = {s: 0.0}
    came: Dict[Tuple[int, int], Tuple[int, int]] = {}
    hq = [(0.0, s)]
    closed = set()
    while hq:
        _, c = heapq.heappop(hq)
        if c == t:
            break
        if c in closed:
            continue
        closed.add(c)
        for dy, dx, w in moves:
            n = (c[0] + dy, c[1] + dx)
            if not (0 <= n[0] < ny and 0 <= n[1] < nx) or g.occ[n]:
                continue
            ng = gs[c] + w + g.cost[n]
            if ng < gs.get(n, np.inf):
                gs[n] = ng
                came[n] = c
                heapq.heappush(hq, (ng + np.hypot(n[0] - t[0], n[1] - t[1]), n))
    if t not in gs:
        return None
    cells = [t]
    while cells[-1] != s:
        cells.append(came[cells[-1]])
    cells.reverse()
    # line-of-sight shortcutting on the inflated grid
    out = [cells[0]]
    i = 0
    while i < len(cells) - 1:
        j = len(cells) - 1
        while j > i + 1 and not _los(g, cells[i], cells[j]):
            j -= 1
        out.append(cells[j])
        i = j
    pts = [g.point(c) for c in out]
    pts[0] = (float(start[0]), float(start[1]))
    pts[-1] = (float(goal[0]), float(goal[1]))
    return pts


def _los(g: Grid, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) * 2 + 1
    for k in range(n + 1):
        u = k / n
        c = (int(round(a[0] + (b[0] - a[0]) * u)), int(round(a[1] + (b[1] - a[1]) * u)))
        if g.occ[c] or g.cost[c] > 0.8:
            return False
    return True


def lookahead_point(path: List[Tuple[float, float]], x: float, y: float, L: float = 0.4) -> Tuple[np.ndarray, int]:
    """Pure-pursuit target: the point at arc distance L ahead of the closest
    point on the path. Returns (point, index of segment)."""
    P = np.asarray(path)
    best, bi, bu = np.inf, 0, 0.0
    for i in range(len(P) - 1):
        a, b = P[i], P[i + 1]
        ab = b - a
        u = float(np.clip(np.dot([x, y] - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1))
        d = np.linalg.norm(a + u * ab - [x, y])
        if d < best:
            best, bi, bu = d, i, u
    rem = L
    cur = P[bi] + bu * (P[bi + 1] - P[bi]) if len(P) > 1 else P[0]
    i = bi
    while i < len(P) - 1:
        seg = P[i + 1] - cur
        ln = np.linalg.norm(seg)
        if ln >= rem:
            return cur + seg / max(ln, 1e-9) * rem, i
        rem -= ln
        cur = P[i + 1]
        i += 1
    return P[-1].copy(), len(P) - 2


# ---------------------------------------------------------------- episodes
@dataclass
class NavEpisode:
    map_id: str
    seed: int
    start: Tuple[float, float, float]   # x, y, yaw
    goal_place: str
    goal: Tuple[float, float]
    instruction: str
    dynamics: Dict[str, float] = field(default_factory=dict)
    randomized: bool = False
    unknown: List[Dict] = field(default_factory=list)  # obstacles NOT in the map (people, boxes)

    def to_json(self) -> Dict:
        return {"map_id": self.map_id, "seed": self.seed, "start": list(self.start), "goal_place": self.goal_place,
                "goal": list(self.goal), "instruction": self.instruction, "dynamics": self.dynamics,
                "randomized": self.randomized, "unknown": self.unknown}


def episode_boxes(m: Dict, ep: "NavEpisode") -> np.ndarray:
    """Physical boxes = map walls/obstacles + this episode's unknown obstacles."""
    B = boxes(m)
    if not ep.unknown:
        return B
    U = np.asarray([[o["pos"][0] - o["size"][0] / 2, o["pos"][1] - o["size"][1] / 2,
                     o["pos"][0] + o["size"][0] / 2, o["pos"][1] + o["size"][1] / 2] for o in ep.unknown])
    return np.concatenate([B, U])


def sample_unknown(m: Dict, rng: np.random.Generator, start, goal, n: int) -> List[Dict]:
    """n boxes (0.15-0.35 m) placed near the planned path, away from start and goal,
    never fully blocking: the path must still exist with them added."""
    g = grid_for(m["id"])
    path = plan_path(g, start, goal)
    if not path or n <= 0:
        return []
    P = np.asarray(path)
    seg = np.cumsum(np.r_[0, np.hypot(*np.diff(P, axis=0).T)])
    out: List[Dict] = []
    for _ in range(40 * n):
        if len(out) >= n:
            break
        s = rng.uniform(0.25, 0.75) * seg[-1]
        i = int(np.searchsorted(seg, s) - 1)
        i = min(max(i, 0), len(P) - 2)
        u = (s - seg[i]) / max(seg[i + 1] - seg[i], 1e-9)
        c = P[i] + u * (P[i + 1] - P[i]) + rng.normal(0, 0.12, 2)
        size = rng.uniform(0.15, 0.35, 2)
        if min(np.hypot(*(c - np.asarray(start[:2]))), np.hypot(*(c - np.asarray(goal)))) < 0.6:
            continue
        cand = {"name": f"unknown{len(out)}", "pos": [float(c[0]), float(c[1])], "size": size.round(3).tolist(),
                "height": 0.5, "color": "#8a8f96"}
        if clearance(boxes(m), *c) < 0.25:
            continue
        gg = grid_with_boxes(g, out + [cand], robot_spec()["radius"] + 0.06)
        if plan_path(gg, start, goal) is None:
            continue
        out.append(cand)
    return out


def grid_with_boxes(g: Grid, extra: List[Dict], inflate: float) -> Grid:
    """Copy of grid g with extra boxes (inflated) marked occupied — local update only."""
    occ = g.occ.copy()
    for o in extra:
        (x, y), (sx, sy) = o["pos"], o["size"]
        x0, x1 = x - sx / 2 - inflate, x + sx / 2 + inflate
        y0, y1 = y - sy / 2 - inflate, y + sy / 2 + inflate
        i0, i1 = max(int(x0 / g.res), 0), min(int(x1 / g.res) + 1, occ.shape[1])
        j0, j1 = max(int(y0 / g.res), 0), min(int(y1 / g.res) + 1, occ.shape[0])
        for j in range(j0, j1):
            cy = (j + 0.5) * g.res
            dy = max(y - sy / 2 - cy, 0.0, cy - (y + sy / 2))
            for i in range(i0, i1):
                cx = (i + 0.5) * g.res
                dx = max(x - sx / 2 - cx, 0.0, cx - (x + sx / 2))
                if dx * dx + dy * dy < inflate * inflate:
                    occ[j, i] = True
    return Grid(occ=occ, cost=g.cost, res=g.res, size=g.size)


NAV_TEMPLATES_ID = ["pergi ke {p}", "ke {p}", "tolong ke {p}", "antar ke {p}", "menuju {p}"]
NAV_TEMPLATES_EN = ["go to the {p}", "navigate to the {p}", "drive to the {p}", "head to the {p}"]


def sample_nav_dynamics(m: Dict, rng: np.random.Generator, randomize: bool) -> Dict[str, float]:
    d = dict(m["dynamics"])
    d.setdefault("body_mass", 1.0)
    if not randomize:
        return d
    for k, r in _load("randomization.json")["dynamics"].items():
        base = d.get(k, 1.0)
        d[k] = float(base * rng.uniform(*r["scale"])) if "scale" in r else float(base + rng.uniform(*r["add"]))
    d["latency_steps"] = int(round(d["latency_steps"]))
    return d


def sample_nav_episode(map_id: str, seed: int, randomize: bool = True,
                       goal_place: Optional[str] = None, unknown: int = 0) -> NavEpisode:
    """unknown: number of obstacles that exist physically but are NOT in the map
    the planner sees (negative = random 0..|unknown|)."""
    m = load_map(map_id)
    rng = np.random.default_rng(seed)
    g = grid_for(map_id)
    B = boxes(m)
    free = np.argwhere(~g.occ)
    places = {p["id"]: p for p in m["places"]}
    for _ in range(200):
        c = tuple(free[rng.integers(len(free))])
        x, y = g.point(c)
        if clearance(B, x, y) < 0.25:
            continue
        gp = goal_place or str(rng.choice(list(places)))
        gx, gy = places[gp]["pos"]
        if np.hypot(gx - x, gy - y) < 1.0:
            continue
        if plan_path(g, (x, y), (gx, gy)) is None:
            continue
        p = places[gp]
        if rng.random() < 0.5:
            text = str(rng.choice(NAV_TEMPLATES_ID)).format(p=str(rng.choice(p["names_id"])))
        else:
            text = str(rng.choice(NAV_TEMPLATES_EN)).format(p=str(rng.choice(p["names_en"])))
        dyn = sample_nav_dynamics(m, rng, randomize)
        k = int(rng.integers(0, -unknown + 1)) if unknown < 0 else unknown
        unk = sample_unknown(m, rng, (x, y), (gx, gy), k)
        return NavEpisode(map_id, seed, (float(x), float(y), float(rng.uniform(-np.pi, np.pi))), gp,
                          (float(gx), float(gy)), text, dyn, randomize, unk)
    raise RuntimeError(f"could not sample a navigation episode in {map_id!r}")


def hard_nav_dynamics(m: Dict) -> Dict[str, float]:
    """Out-of-range 'worn robot on a slippery floor' parameters (stress test)."""
    d = dict(m["dynamics"], body_mass=1.5)
    d.update(motor_tau=0.25, latency_steps=3, floor_friction=0.45, lidar_noise=0.03, pose_noise=0.03,
             wheel_radius_scale=0.94)
    return d
