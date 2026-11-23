"""ROS-independent logic of the qi_vla_ros nodes (unit-tested without ROS).

The nodes reuse the repository's single-source arm model (`src/robot/arm.py`)
and simulators (`src/sim`), found via the QI_VLA_ROOT environment variable or
by walking up from this file (works with `colcon build --symlink-install`).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def ensure_repo_on_path() -> Path:
    cands = [os.environ.get("QI_VLA_ROOT")] + [str(p) for p in Path(__file__).resolve().parents]
    for c in cands:
        if c and (Path(c) / "src" / "robot" / "arm.py").exists():
            if c not in sys.path:
                sys.path.insert(0, c)
            return Path(c)
    raise RuntimeError("QI-VLA repository not found: export QI_VLA_ROOT=/path/to/qi-vlam")


ensure_repo_on_path()

import numpy as np  # noqa: E402

from src.robot import arm  # noqa: E402

FINGER_OPEN = 0.024
STATE_NAMES = list(arm.JOINT_NAMES) + ["finger_l_joint", "finger_r_joint"]


def trajectory_to_q(names: Sequence[str], positions: Sequence[float], current: Sequence[float]) -> np.ndarray:
    """Map a JointTrajectory point (any joint order / subset) onto the 6-joint
    vector; joints not mentioned keep their current value. Clamped to limits."""
    q = np.asarray(current, dtype=np.float64).copy()
    for n, v in zip(names, positions):
        if n in arm.JOINT_NAMES and np.isfinite(v):
            q[arm.JOINT_NAMES.index(n)] = float(v)
    return arm.clamp_q(q)


def joint_state_lists(q: Sequence[float]) -> List[float]:
    q = list(map(float, q))
    return q + [q[5] * FINGER_OPEN, q[5] * FINGER_OPEN]


class KinematicServo:
    """First-order servo lag + command latency + slew limit (no contacts)."""

    def __init__(self, tau: float = 0.08, latency_steps: int = 0) -> None:
        self.q = arm.Q_HOME.copy()
        self.cmd = arm.Q_HOME.copy()
        self.tau = tau
        self.queue: List[np.ndarray] = [self.cmd.copy()] * latency_steps

    def command(self, q: Sequence[float]) -> None:
        self.queue.append(arm.clamp_q(np.asarray(q, dtype=np.float64)))
        self.cmd = self.queue.pop(0)

    def step(self, dt: float) -> np.ndarray:
        dq = (self.cmd - self.q) * min(dt / self.tau, 1.0)
        dq = np.clip(dq, -arm.QD_MAX * dt * 1.2, arm.QD_MAX * dt * 1.2)
        self.q = arm.clamp_q(self.q + dq)
        return self.q


class Backend:
    """kinematic: KinematicServo (fast, no objects)  |  mujoco: src.sim.physics.ArmSim."""

    def __init__(self, kind: str = "kinematic", env_id: str = "lab_table", seed: int = 0,
                 randomize: bool = False) -> None:
        self.kind, self.env_id, self.seed, self.randomize = kind, env_id, seed, randomize
        self.cmd = arm.Q_HOME.copy()
        self.reset(seed)

    @property
    def dt(self) -> float:
        return 0.05 if self.kind == "mujoco" else 0.02

    def reset(self, seed: Optional[int] = None) -> None:
        self.seed = self.seed if seed is None else seed
        self.cmd = arm.Q_HOME.copy()
        if self.kind == "mujoco":
            from src.sim.envs import sample_episode
            from src.sim.physics import ArmSim
            self.sim = ArmSim(sample_episode(self.env_id, self.seed, randomize=self.randomize))
        elif self.kind == "kinematic":
            self.servo = KinematicServo()
        else:
            raise ValueError(f"unknown backend {self.kind!r} (kinematic | mujoco)")

    def command(self, names: Sequence[str], positions: Sequence[float]) -> np.ndarray:
        self.cmd = trajectory_to_q(names, positions, self.cmd)
        if self.kind == "kinematic":
            self.servo.command(self.cmd)
        return self.cmd

    def step(self) -> np.ndarray:
        if self.kind == "mujoco":
            self.sim.step(self.cmd)
            return self.sim.q_true()
        return self.servo.step(self.dt)

    def q(self) -> np.ndarray:
        return self.sim.q_true() if self.kind == "mujoco" else self.servo.q.copy()

    def objects(self) -> Dict[str, Dict]:
        """name -> {kind, pos[3], yaw, color} (empty for the kinematic backend)."""
        if self.kind != "mujoco":
            return {}
        out = {c: {"kind": "cube", "pos": self.sim.cube_pose(c)[:3].tolist(), "yaw": float(self.sim.cube_pose(c)[3]),
                   "size": self.sim.cube_half * 2} for c in self.sim.episode.cubes}
        out.update({c: {"kind": "pad", "pos": [x, y, 0.001], "yaw": 0.0, "size": 0.06}
                    for c, (x, y) in self.sim.episode.pads.items()})
        return out


COLORS = {"red": (0.90, 0.22, 0.21), "green": (0.26, 0.63, 0.28), "blue": (0.12, 0.53, 0.90),
          "yellow": (0.99, 0.85, 0.21), "purple": (0.56, 0.14, 0.67)}


def reach_command(policy, q_meas: Sequence[float], target: Sequence[float], dt: float = 0.05) -> np.ndarray:
    """One step of an exported reach policy -> next joint position target."""
    q = np.asarray(q_meas, dtype=np.float64)
    qd = np.asarray(policy.act(q, target))
    out = q.copy()
    out[:4] += qd * dt
    return arm.clamp_q(out)


class RoverBackend:
    """Diff-drive rover for the ROS node: src.sim.mobile (MuJoCo or kinematic)
    + a /cmd_vel watchdog (zero velocity after `watchdog_s` without commands)."""

    def __init__(self, kind: str = "kinematic", map_id: str = "home", seed: int = 0, unknown: int = 0,
                 watchdog_s: float = 0.5) -> None:
        from src.sim.mobile.geometry import robot_spec
        self.kind, self.map_id, self.seed, self.unknown, self.watchdog_s = kind, map_id, seed, unknown, watchdog_s
        self.R = robot_spec()
        self.reset(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        from src.sim.mobile.geometry import load_map, sample_nav_episode
        from src.sim.mobile.physics import make_rover
        self.seed = self.seed if seed is None else seed
        self.ep = sample_nav_episode(self.map_id, self.seed, randomize=False, unknown=self.unknown)
        self.sim = make_rover(self.ep, "mujoco" if self.kind == "mujoco" else "kinematic")
        self.map = load_map(self.map_id)
        self.cmd = (0.0, 0.0)
        self.t_cmd = -1e9

    def command(self, v: float, w: float, now: float) -> None:
        if np.isfinite(v) and np.isfinite(w):
            self.cmd = (float(np.clip(v, -self.R["v_max"], self.R["v_max"])),
                        float(np.clip(w, -self.R["w_max"], self.R["w_max"])))
            self.t_cmd = now

    def step(self, now: float) -> Dict:
        v, w = self.cmd if now - self.t_cmd <= self.watchdog_s else (0.0, 0.0)
        self.sim.step(v, w)
        o = self.sim.observe()
        return {"pose": list(self.sim.pose()), "vw": [float(self.sim.cmd[0]), float(self.sim.cmd[1])],
                "scan": o["scan"].tolist()}

    def map_boxes(self) -> List[Dict]:
        hexrgb = lambda h: tuple(int(h.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4))  # noqa: E731
        out = [{"pos": b["pos"], "size": b["size"], "h": 0.35, "rgb": hexrgb(self.map["wall_color"])} for b in self.map["walls"]]
        out += [{"pos": o["pos"], "size": o["size"], "h": o["height"], "rgb": hexrgb(o["color"])}
                for o in self.map["obstacles"] + self.ep.unknown]
        return out
