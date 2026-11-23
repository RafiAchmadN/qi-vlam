"""Navigation stack for the rover: planner-following expert, policy features,
lidar safety shield. Mirrored in `app/src/rover/rover.js` (parity-tested).

Architecture (the standard Nav2-style split):
  global planner  A* on the inflated occupancy grid (classical, geometry.py)
  local controller  pure-pursuit EXPERT  or  a learned policy (dense / TTNS)
                    trained by behaviour cloning on the expert
  safety shield  lidar control-barrier-style speed limit, applied last
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.sim.mobile.geometry import Grid, grid_for, lookahead_point, plan_path, robot_spec

NAV_DIM = 25
NAV_OUT = 2
LOOKAHEAD = 0.4


def wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def nav_features(pose, scan, target, goal, vw) -> np.ndarray:
    """18 lidar sectors (min of beam pairs, /range_max) | lookahead target in the
    robot frame: dist/1 (clipped), sin, cos of bearing | goal dist/3 (clipped) |
    v/v_max, w/w_max (last command) | 1   = 25"""
    R = robot_spec()
    x, y, yaw = pose
    s = np.asarray(scan, dtype=np.float64)
    sec = np.minimum(s[0::2], s[1::2]) / R["lidar"]["range_max"]
    dx, dy = target[0] - x, target[1] - y
    b = wrap(np.arctan2(dy, dx) - yaw)
    gd = np.hypot(goal[0] - x, goal[1] - y)
    return np.concatenate([np.clip(sec, 0, 1), [min(np.hypot(dx, dy), 1.0), np.sin(b), np.cos(b)],
                           [min(gd, 3.0) / 3.0], [vw[0] / R["v_max"], vw[1] / R["w_max"]], [1.0]]).astype(np.float32)


def shield_vw(v: float, w: float, scan, d_stop: float = 0.08, horizon: float = 0.6) -> Tuple[float, float, bool]:
    """Lidar barrier: with h = d_front - radius - d_stop, enforce v <= h / horizon
    (discrete-time CBF-style speed bound) for forward motion, and the mirrored
    bound for reversing. Turning in place is always allowed."""
    R = robot_spec()
    s = np.asarray(scan, dtype=np.float64)
    n = len(s)
    k = max(1, int(round(n * 45 / 360)))
    front = np.r_[s[:k + 1], s[-k:]].min()
    back = s[n // 2 - k: n // 2 + k + 1].min()
    vmax_f = max(0.0, (front - R["radius"] - d_stop) / horizon)
    vmax_b = max(0.0, (back - R["radius"] - d_stop) / horizon)
    vs = float(np.clip(v, -vmax_b, vmax_f))
    return vs, float(w), bool(abs(vs - v) > 1e-6)


class Costmap:
    """Static map grid + obstacles marked from lidar hits (persisting), the
    way Nav2's costmap layers fuse the map with live sensor data."""

    def __init__(self, static: Grid) -> None:
        self.static = static
        self.marks = np.zeros_like(static.occ)
        ri = int(np.ceil((robot_spec()["radius"] + 0.06) / static.res))
        yy, xx = np.mgrid[-ri:ri + 1, -ri:ri + 1]
        self.disc = np.argwhere(yy ** 2 + xx ** 2 <= ri ** 2) - ri

    def update(self, pose, scan, rmax: float = 2.5) -> bool:
        x, y, yaw = pose
        n = len(scan)
        new = False
        g = self.static
        for k in range(n):
            r = scan[k]
            if r >= rmax:
                continue
            a = yaw + 2 * np.pi * k / n
            px, py = x + (r + 0.03) * np.cos(a), y + (r + 0.03) * np.sin(a)
            c = np.asarray(g.cell(px, py))
            if g.occ[tuple(c)] and g.cost[tuple(c)] > 0:   # already known map obstacle
                continue
            cells = c + self.disc
            ok = (cells[:, 0] >= 0) & (cells[:, 0] < g.occ.shape[0]) & (cells[:, 1] >= 0) & (cells[:, 1] < g.occ.shape[1])
            cells = cells[ok]
            if not self.marks[cells[:, 0], cells[:, 1]].all():
                self.marks[cells[:, 0], cells[:, 1]] = True
                new = True
        return new

    def grid(self) -> Grid:
        g = self.static
        return Grid(occ=g.occ | self.marks, cost=g.cost, res=g.res, size=g.size)

    def blocked(self, path) -> bool:
        g = self.static
        return any(self.marks[g.cell(px, py)] for px, py in path[1:])


class NavExpert:
    """Pure pursuit on the A* path (re-planned on a lidar-updated costmap) with
    curvature- and clearance-aware speed."""

    def __init__(self, map_id: str, goal, replan_s: float = 4.0, dart: float = 0.0,
                 rng: Optional[np.random.Generator] = None) -> None:
        self.map_id, self.goal = map_id, np.asarray(goal, dtype=np.float64)
        self.grid = grid_for(map_id)
        self.costmap = Costmap(self.grid)
        self.path: Optional[List[Tuple[float, float]]] = None
        self.t_plan = -1e9
        self.replan_s = replan_s
        self.dart = dart
        self.rng = rng or np.random.default_rng(0)
        self.done = False

    def target(self, pose, t: float, scan=None) -> np.ndarray:
        x, y, _ = pose
        changed = scan is not None and self.costmap.update(pose, scan)
        since = t - self.t_plan
        need = self.path is None or since > self.replan_s
        if not need and since > 1.0:  # event-triggered replans are rate-limited to 1 Hz
            P = np.asarray(self.path)
            need = np.min(np.hypot(P[:, 0] - x, P[:, 1] - y)) > 0.5 or (changed and self.costmap.blocked(self.path))
        if need:
            self.t_plan = t  # also on failure: never retry A* every tick
            p = plan_path(self.costmap.grid(), (x, y), self.goal)
            if p is None:  # goal region swallowed by noisy marks: fall back to the static map
                p = plan_path(self.grid, (x, y), self.goal)
            if p is not None:
                self.path = p
        if self.path is None:
            return self.goal.copy()
        return lookahead_point(self.path, x, y, LOOKAHEAD)[0]

    def act(self, obs: Dict) -> Dict:
        R = robot_spec()
        pose = obs["pose"]
        tgt = self.target(pose, obs["t"], obs["scan"])
        x, y, yaw = pose
        gd = np.hypot(self.goal[0] - x, self.goal[1] - y)
        if gd < 0.12:
            self.done = True
            return {"v": 0.0, "w": 0.0, "target": tgt}
        a = wrap(np.arctan2(tgt[1] - y, tgt[0] - x) - yaw)
        w = float(np.clip(2.5 * a, -R["w_max"], R["w_max"]))
        v = R["v_max"] * max(np.cos(a), 0.0) ** 2 * min(1.0, gd / 0.5 + 0.2)
        s = np.asarray(obs["scan"])
        k = len(s) // 8
        front = np.r_[s[:k + 1], s[-k:]].min()
        v *= float(np.clip((front - R["radius"] - 0.05) / 0.35, 0.0, 1.0))
        return {"v": float(v), "w": w, "target": tgt}


def run_nav(sim, controller="expert", policy=None, shield: bool = True, max_t: float = 60.0,
            record: bool = False, dart: float = 0.0, rng=None) -> Dict:
    """Closed-loop navigation episode. controller: 'expert' or 'policy' (needs
    `policy` with .forward). Success = within 0.2 m of the goal place."""
    ep = sim.episode
    ex = NavExpert(ep.map_id, ep.goal)
    rng = rng or np.random.default_rng(ep.seed + 11)
    R = robot_spec()
    frames = []
    interventions = 0
    steps = 0
    stable = 0
    while sim.t < max_t:
        obs = sim.observe()
        a = ex.act(obs)
        if controller == "policy":
            y = policy.forward(nav_features(obs["pose"], obs["scan"], a["target"], ep.goal, obs["vw"]))
            v, w = float(np.clip(y[0], -1, 1)) * R["v_max"], float(np.clip(y[1], -1, 1)) * R["w_max"]
        else:
            v, w = a["v"], a["w"]
        if record:
            frames.append((obs, a, (a["v"], a["w"])))
        if dart > 0:
            v += rng.normal(0, dart) * R["v_max"]
            w += rng.normal(0, dart) * R["w_max"]
        if shield:
            v, w, hit = shield_vw(v, w, obs["scan"])
            interventions += int(hit)
        sim.step(v, w)
        steps += 1
        stable = stable + 1 if sim.at_goal() else 0
        if stable >= 5:
            break
    ok = stable >= 5
    return {"success": ok, "t": round(sim.t, 2), "collision": sim.collision_steps > 0,
            "collision_steps": sim.collision_steps, "path_len": round(sim.path_len, 3),
            "shield_pct": 100.0 * interventions / max(steps, 1), "frames": frames}
