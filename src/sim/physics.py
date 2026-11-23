"""MuJoCo physics simulation of the arm in an environment preset.

`ArmSim` is the "realistic" counterpart of the app's kinematic simulation:
rigid-body dynamics with gravity, contacts and friction grasping (the cube is
held only by finger friction — no attach/weld), plus a servo model with the
imperfections real hobby/STS servos have:

  * command latency (`latency_steps` control periods of delay)
  * first-order servo lag (`servo_tau`)
  * backlash / dead-band (`backlash`, rad, hysteresis on the commanded target)
  * limited actuator strength (`kp_scale`, force ranges in the MJCF)
  * joint-encoder noise (`q_noise`) and object-position noise (`obj_noise`,
    standing in for camera-detection error)

Interface (control period DT = 0.05 s, the same 20 Hz as the app):
    sim = ArmSim(episode)                 # or ArmSim.from_env("kitchen_counter")
    obs = sim.observe()                   # measured q, tip, perceived objects
    sim.step(q_cmd)                       # 6 joint targets (gripper 0..1)
    sim.success(), sim.collisions
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

import mujoco
import numpy as np

from src.robot import arm
from src.sim.envs import CUBE, EpisodeConfig, load_env, sample_episode
from src.sim.mjcf import FINGER_OPEN, build_mjcf

DT = 0.05
ARM_JOINTS = arm.JOINT_NAMES[:5]


def quat_yaw(q: np.ndarray) -> float:
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class ArmSim:
    def __init__(self, episode: EpisodeConfig, seed: Optional[int] = None,
                 dynamics: Optional[Dict] = None) -> None:
        self.episode = episode
        self.spec = load_env(episode.env_id)
        self.dyn = dict(self.spec["dynamics"], link_mass=1.0, cube_size=1.0)
        self.dyn.update(episode.dynamics)
        if dynamics:
            self.dyn.update(dynamics)
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(episode.env_id, episode, self.dyn))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(episode.seed if seed is None else seed)
        self.n_sub = int(round(DT / self.model.opt.timestep))
        self.cube_half = CUBE / 2 * float(self.dyn.get("cube_size", 1.0))
        m = self.model
        self.jid = [m.joint(n).id for n in ARM_JOINTS]
        self.qadr = [m.jnt_qposadr[j] for j in self.jid]
        self.fadr = [m.jnt_qposadr[m.joint(n).id] for n in ("finger_l", "finger_r")]
        self.tip = m.site("tip").id
        self.cube_body = {c: m.body(f"cube_{c}").id for c in episode.cubes}
        self.cube_geom = {m.geom(f"cube_{c}_geom").id: c for c in episode.cubes}
        self.finger_geoms = {m.geom("finger_l_geom").id, m.geom("finger_r_geom").id}
        arm_bodies = {m.body(n).id for n in ("base", "pan_link", "upper_arm", "forearm", "wrist", "hand",
                                              "finger_l", "finger_r")}
        self.arm_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in arm_bodies}
        self.world_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == 0}
        self.reset()

    @classmethod
    def from_env(cls, env_id: str, seed: int = 0, randomize: bool = False) -> "ArmSim":
        return cls(sample_episode(env_id, seed, randomize=randomize))

    # ------------------------------------------------------------------ state
    def reset(self, q0: Optional[np.ndarray] = None) -> None:
        mujoco.mj_resetData(self.model, self.data)
        q0 = arm.Q_HOME.copy() if q0 is None else np.asarray(q0, dtype=np.float64)
        for a, v in zip(self.qadr, q0[:5]):
            self.data.qpos[a] = v
        for a in self.fadr:
            self.data.qpos[a] = q0[5] * FINGER_OPEN
        self.target = q0.copy()                 # servo-filtered actuator target
        self.eff = q0.copy()                    # after backlash dead-band
        self.queue: deque = deque([q0.copy()] * int(self.dyn.get("latency_steps", 0)))
        self._apply_ctrl(self.eff)
        mujoco.mj_forward(self.model, self.data)
        for _ in range(100):                    # settle cubes onto the table
            mujoco.mj_step(self.model, self.data)
        self.t = 0.0
        self.collisions = 0
        self.collision_steps = 0
        self.last_cmd = q0.copy()

    def _apply_ctrl(self, q: np.ndarray) -> None:
        self.data.ctrl[:5] = q[:5]
        self.data.ctrl[5] = self.data.ctrl[6] = float(np.clip(q[5], 0, 1)) * FINGER_OPEN

    def q_true(self) -> np.ndarray:
        q = [self.data.qpos[a] for a in self.qadr]
        g = float(np.mean([self.data.qpos[a] for a in self.fadr]) / FINGER_OPEN)
        return np.asarray(q + [g])

    def tip_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.tip].copy()

    def cube_pose(self, c: str) -> np.ndarray:
        b = self.cube_body[c]
        p = self.data.xpos[b]
        return np.array([p[0], p[1], p[2], quat_yaw(self.data.xquat[b])])

    def goal_pos(self) -> np.ndarray:
        """Support point the object should end up on (pad centre on the table,
        or the top-centre of the goal cube), in world coordinates."""
        ep = self.episode
        if ep.goal_kind == "pad":
            x, y = ep.pads[ep.goal]
            return np.array([x, y, 0.0])
        p = self.cube_pose(ep.goal)
        return np.array([p[0], p[1], p[2] + self.cube_half])

    def observe(self) -> Dict:
        """What a real robot would measure: noisy joints, noisy object detections."""
        qn = float(self.dyn.get("q_noise", 0.0))
        on = float(self.dyn.get("obj_noise", 0.0))
        q = self.q_true() + np.r_[self.rng.normal(0, qn, 5), 0.0]
        objs = {}
        for c in self.episode.cubes:
            p = self.cube_pose(c)
            p[:2] += self.rng.normal(0, on, 2)
            objs[c] = p
        goal = self.goal_pos()
        goal[:2] += self.rng.normal(0, on, 2)
        return {"q": q, "tip": arm.fk(q), "objects": objs, "goal": goal,
                "obj": objs[self.episode.obj], "t": self.t}

    # ------------------------------------------------------------------ step
    def step(self, q_cmd: np.ndarray) -> None:
        q_cmd = arm.clamp_q(np.asarray(q_cmd, dtype=np.float64))
        self.last_cmd = q_cmd.copy()
        self.queue.append(q_cmd)
        cmd = self.queue.popleft()
        tau = max(float(self.dyn.get("servo_tau", 0.06)), 1e-3)
        self.target = self.target + (cmd - self.target) * min(DT / tau, 1.0)
        b = float(self.dyn.get("backlash", 0.0)) / 2
        self.eff[:5] = np.clip(self.eff[:5], self.target[:5] - b, self.target[:5] + b)
        self.eff[5] = self.target[5]
        self._apply_ctrl(self.eff)
        hit = False
        for _ in range(self.n_sub):
            mujoco.mj_step(self.model, self.data)
            hit |= self._arm_hits_world()
        self.collision_steps += int(hit)
        self.t += DT

    def _arm_hits_world(self) -> bool:
        d = self.data
        for i in range(d.ncon):
            c = d.contact[i]
            a, b = c.geom1, c.geom2
            if (a in self.arm_geoms and b in self.world_geoms) or (b in self.arm_geoms and a in self.world_geoms):
                if c.dist < -0.002:  # >2 mm penetration = a real collision, not a graze
                    return True
        return False

    def held(self, c: Optional[str] = None) -> bool:
        """Cube c is touched by both fingers and off the table."""
        c = c or self.episode.obj
        g = {k for k, v in self.cube_geom.items() if v == c}
        touch = set()
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            if con.geom1 in g and con.geom2 in self.finger_geoms:
                touch.add(con.geom2)
            elif con.geom2 in g and con.geom1 in self.finger_geoms:
                touch.add(con.geom1)
        return len(touch) == 2 and self.cube_pose(c)[2] > self.cube_half + 0.005

    def success(self) -> bool:
        ep = self.episode
        p = self.cube_pose(ep.obj)
        if self.held(ep.obj):
            return False
        if ep.goal_kind == "pad":
            x, y = ep.pads[ep.goal]
            return bool(np.hypot(p[0] - x, p[1] - y) < 0.03 and p[2] < self.cube_half + 0.01)
        g = self.cube_pose(ep.goal)
        return bool(np.hypot(p[0] - g[0], p[1] - g[1]) < 0.02 and
                    abs(p[2] - (g[2] + 2 * self.cube_half)) < 0.01)

    def snapshot(self) -> Dict:
        """JSON-friendly world state (used by the bridge to mirror the scene in the app)."""
        return {"q": self.q_true().round(5).tolist(),
                "objects": {c: self.cube_pose(c).round(5).tolist() for c in self.episode.cubes},
                "pads": {c: list(v) for c, v in self.episode.pads.items()},
                "shapes": dict(self.episode.shapes),
                "env": self.episode.env_id, "t": round(self.t, 3)}

    def render(self, camera: str = "top", width: int = 160, height: int = 128) -> np.ndarray:
        r = mujoco.Renderer(self.model, height=height, width=width)
        r.update_scene(self.data, camera=camera)
        img = r.render()
        r.close()
        return img


def rollout_success_rate(policy_fn, env_ids: List[str], n: int, seed0: int = 10_000,
                         randomize: bool = True, max_steps: int = 300) -> float:
    """Convenience: fraction of successful episodes for a callable
    policy_fn(sim, obs) -> q_cmd."""
    ok = 0
    for k in range(n):
        ep = sample_episode(env_ids[k % len(env_ids)], seed0 + k, randomize=randomize)
        sim = ArmSim(ep)
        for _ in range(max_steps):
            sim.step(policy_fn(sim, sim.observe()))
            if sim.success():
                ok += 1
                break
    return ok / max(n, 1)
