"""Gymnasium wrapper: the MuJoCo arm + environment presets as a standard
training environment (RL, imitation, or any library that speaks Gymnasium).

    import gymnasium as gym, src.sim.gym_env  # registers the ids
    env = gym.make("QiVla/PickPlace-v0", env_id="kitchen_counter", randomize=True)
    obs, info = env.reset(seed=0); print(info["instruction"])
    obs, r, terminated, truncated, info = env.step(env.action_space.sample())

Action (Box[-1, 1]^5): normalized joint velocities qd0..qd3 (x QD_MAX) and a
gripper command (> 0 open, <= 0 close). Integrated into joint targets exactly
like the app and `train_pickplace.policy_episode` (anti-windup, analytic
wrist roll), then sent through the servo model (lag, latency, backlash).

Observation (Dict):
  state               6   measured joints (noisy), gripper 0..1
  environment_state   32  same layout as the dataset (src/sim/collect.py)
  features            17  the policy feature vector (src/sim/features.py)
Reward: dense shaping (-distance tip->cube before grasp, -distance
cube->place after) + 10 on success; -1 per step with an arm/world collision.
"""

from __future__ import annotations

from typing import Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.robot import arm
from src.sim.envs import ENV_STATE_DIM, env_index, env_state_vector, sample_episode
from src.sim.expert import MAX_WINDUP, roll_for_yaw
from src.sim.features import PP_DIM, pickplace_features
from src.sim.physics import DT, ArmSim


class QiVlaPickPlaceEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, env_id: str = "lab_table", randomize: bool = True,
                 render_mode: Optional[str] = None, max_steps: int = 500,
                 camera: str = "side") -> None:
        if env_id != "random" and env_id not in env_index()["envs"]:
            raise ValueError(f"unknown env_id {env_id!r}")
        self.env_id, self.randomize, self.render_mode = env_id, randomize, render_mode
        self.max_steps, self.camera = max_steps, camera
        f = lambda n: spaces.Box(-np.inf, np.inf, (n,), np.float32)  # noqa: E731
        self.observation_space = spaces.Dict({"state": f(6), "environment_state": f(ENV_STATE_DIM), "features": f(PP_DIM)})
        self.action_space = spaces.Box(-1.0, 1.0, (5,), np.float32)
        self.sim: Optional[ArmSim] = None

    def _obs(self) -> Dict:
        o = self.sim.observe()
        ep = self.sim.episode
        env_state = env_state_vector(o["obj"], o["goal"], ep.goal_kind == "cube", o["objects"], ep.pads)
        self._last = o
        return {"state": o["q"].astype(np.float32), "environment_state": env_state,
                "features": pickplace_features(o["q"], o["obj"], o["goal"], ep.goal_kind == "cube")}

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        options = options or {}
        env_id = options.get("env_id", self.env_id)
        if env_id == "random":
            env_id = str(self.np_random.choice(env_index()["train_envs"]))
        ep_seed = int(self.np_random.integers(2 ** 31 - 1))
        ep = sample_episode(env_id, ep_seed, randomize=self.randomize, task=options.get("task"))
        self.sim = ArmSim(ep, dynamics=options.get("dynamics"))
        self.q_cmd = arm.Q_HOME.copy()
        self.roll, self.steps, self.stable = 0.0, 0, 0
        obs = self._obs()
        return obs, {"instruction": ep.instruction, "env_id": env_id, "episode": ep.to_json()}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        o = self._last
        q = o["q"]
        grip = 1.0 if a[4] > 0 else 0.0
        if grip > 0.5 and o["obj"][2] < self.sim.cube_half + 0.01:
            self.roll = roll_for_yaw(q[0], o["obj"][3])
        qc = self.q_cmd
        qc[:4] = np.clip(qc[:4] + a[:4] * arm.QD_MAX[:4] * DT, q[:4] - MAX_WINDUP, q[:4] + MAX_WINDUP)
        qc[4] += np.clip(self.roll - qc[4], -arm.QD_MAX[4] * DT, arm.QD_MAX[4] * DT)
        qc[5] = grip
        self.q_cmd = arm.clamp_q(qc)
        coll_before = self.sim.collision_steps
        self.sim.step(self.q_cmd)
        self.steps += 1
        obs = self._obs()
        tip, obj = self.sim.tip_pos(), self.sim.cube_pose(self.sim.episode.obj)[:3]
        place = self.sim.goal_pos() + np.array([0, 0, self.sim.cube_half])
        held = self.sim.held()
        reward = -np.linalg.norm(obj - place) - (0.0 if held else 0.5 * np.linalg.norm(tip - obj))
        reward -= 1.0 * (self.sim.collision_steps - coll_before)
        ok = self.sim.success()
        self.stable = self.stable + 1 if ok else 0
        terminated = self.stable >= 10
        if terminated:
            reward += 10.0
        truncated = self.steps >= self.max_steps
        info = {"success": bool(ok), "held": bool(held), "collision_steps": self.sim.collision_steps}
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array" and self.sim is not None:
            return self.sim.render(self.camera, 320, 240)
        return None


gym.register(id="QiVla/PickPlace-v0", entry_point="src.sim.gym_env:QiVlaPickPlaceEnv",
             max_episode_steps=500)
