"""Gymnasium environment for language-goal navigation with the MuJoCo rover.

    import gymnasium as gym, src.sim.mobile.gym_env
    env = gym.make("QiVla/Nav-v0", map_id="warehouse", unknown=2)
    obs, info = env.reset(seed=0); print(info["instruction"])   # e.g. "ke rak b"

Action Box[-1,1]^2 = (v/v_max, w/w_max). Observation (Dict): pose (noisy,
3), scan (36), goal (2), features (25 — nav.nav_features with the
pure-pursuit target of the A*/costmap planner, so an agent can learn only
the local controller, or ignore `features` and learn everything from
pose/scan/goal). Reward: 10 x progress toward the goal - 0.01 per step
- 1 per colliding step + 10 on success (within 0.2 m for 5 steps).
`backend="kinematic"` uses the fast unicycle model instead of MuJoCo.
"""

from __future__ import annotations

from typing import Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.sim.mobile.geometry import map_index, robot_spec, sample_nav_episode
from src.sim.mobile.nav import NAV_DIM, NavExpert, nav_features
from src.sim.mobile.physics import make_rover


class QiVlaNavEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, map_id: str = "home", randomize: bool = True, unknown: int = 0,
                 backend: str = "mujoco", max_steps: int = 1200, render_mode: Optional[str] = None) -> None:
        if map_id != "random" and map_id not in map_index()["maps"]:
            raise ValueError(f"unknown map {map_id!r}")
        self.map_id, self.randomize, self.unknown, self.backend = map_id, randomize, unknown, backend
        self.max_steps, self.render_mode = max_steps, render_mode
        n = robot_spec()["lidar"]["beams"]
        f = lambda k: spaces.Box(-np.inf, np.inf, (k,), np.float32)  # noqa: E731
        self.observation_space = spaces.Dict({"pose": f(3), "scan": f(n), "goal": f(2), "features": f(NAV_DIM)})
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.sim = None

    def _obs(self) -> Dict:
        o = self.sim.observe()
        tgt = self.planner.target(o["pose"], o["t"], o["scan"])
        self._last = o
        return {"pose": o["pose"].astype(np.float32), "scan": o["scan"].astype(np.float32),
                "goal": np.asarray(self.sim.episode.goal, np.float32),
                "features": nav_features(o["pose"], o["scan"], tgt, self.sim.episode.goal, o["vw"])}

    def _dist(self) -> float:
        x, y, _ = self.sim.pose()
        g = self.sim.episode.goal
        return float(np.hypot(x - g[0], y - g[1]))

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        options = options or {}
        mid = options.get("map_id", self.map_id)
        if mid == "random":
            mid = str(self.np_random.choice(map_index()["train_maps"]))
        ep = sample_nav_episode(mid, int(self.np_random.integers(2 ** 31 - 1)), randomize=self.randomize,
                                goal_place=options.get("goal_place"), unknown=options.get("unknown", self.unknown))
        self.sim = make_rover(ep, self.backend, options.get("dynamics"))
        self.planner = NavExpert(mid, ep.goal)
        self.steps, self.stable = 0, 0
        self.prev = self._dist()
        return self._obs(), {"instruction": ep.instruction, "map_id": mid, "episode": ep.to_json()}

    def step(self, action):
        R = robot_spec()
        a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        c0 = self.sim.collision_steps
        self.sim.step(a[0] * R["v_max"], a[1] * R["w_max"])
        self.steps += 1
        d = self._dist()
        reward = 10.0 * (self.prev - d) - 0.01 - 1.0 * (self.sim.collision_steps - c0)
        self.prev = d
        self.stable = self.stable + 1 if self.sim.at_goal() else 0
        terminated = self.stable >= 5
        if terminated:
            reward += 10.0
        obs = self._obs()
        return obs, float(reward), terminated, self.steps >= self.max_steps, {
            "success": terminated, "collision_steps": self.sim.collision_steps, "goal_dist": d}

    def render(self):
        if self.render_mode == "rgb_array" and hasattr(self.sim, "render"):
            return self.sim.render()
        return None


gym.register(id="QiVla/Nav-v0", entry_point="src.sim.mobile.gym_env:QiVlaNavEnv", max_episode_steps=1200)
