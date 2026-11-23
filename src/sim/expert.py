"""Scripted pick-and-place expert for the MuJoCo arm (data-collection teacher).

The same skill decomposition as the app's `app/src/vla/agent.js`
(open -> above -> descend -> close -> lift -> above goal -> lower -> release
-> retreat), but closed-loop on *measured* joints through the physics and the
servo model: each tick it computes a damped-least-squares IK velocity from the
measured pose (`arm.ik_step`) and integrates it into the joint command, so
gravity droop and servo lag are corrected the way a real position-controlled
arm would need (integral action), with an anti-windup clamp.

Wrist roll is set analytically so the fingers close on two parallel cube
faces: the closing direction is q0 + pi/2 - roll (verified numerically on the
MJCF model), hence roll = wrap_{pi/4}(q0 - yaw).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from src.robot import arm
from src.sim.physics import DT, ArmSim

PITCH = 2.8
ABOVE = 0.07          # approach height above the grasp/release point (m)
TOL = 0.006
GRIP_TIME = 0.6       # s to close/open
MAX_WINDUP = 0.25     # rad: |q_cmd - q_meas| clamp
PHASES = ("open", "above_obj", "descend", "close", "lift", "above_goal", "lower", "release", "retreat", "done")


def wrap_quarter(a: float) -> float:
    """Wrap an angle to [-pi/4, pi/4) (a cube is symmetric under 90 deg)."""
    return float((a + np.pi / 4) % (np.pi / 2) - np.pi / 4)


def roll_for_yaw(q0: float, yaw: float) -> float:
    return wrap_quarter(q0 - yaw)


@dataclass
class ExpertState:
    phase: str = "open"
    t: float = 0.0
    grasp_xy: Optional[np.ndarray] = None
    roll: float = 0.0
    settle: int = 0
    offset: Optional[np.ndarray] = None   # tip_xy - cube_xy while holding (in-hand slip)


class PickPlaceExpert:
    def __init__(self, travel_z: float = 0.12, gain: float = 1.0, dart_noise: float = 0.0,
                 rng: Optional[np.random.Generator] = None) -> None:
        self.travel_z = travel_z
        self.gain = gain
        self.dart = dart_noise
        self.rng = rng or np.random.default_rng(0)
        self.reset()

    def reset(self, q_cmd: Optional[np.ndarray] = None) -> None:
        self.s = ExpertState()
        self.q_cmd = arm.Q_HOME.copy() if q_cmd is None else np.asarray(q_cmd, dtype=np.float64).copy()

    # ------------------------------------------------------------------ goals
    def subgoal(self, obs: Dict, half: float) -> Tuple[Optional[np.ndarray], float]:
        """(tip target or None, gripper command) for the current phase."""
        s, obj, goal = self.s, obs["obj"], obs["goal"]
        ph = s.phase
        if ph in ("open",):
            return None, 1.0
        if ph == "above_obj":
            return np.array([obj[0], obj[1], obj[2] + ABOVE]), 1.0
        if ph == "descend":
            xy = s.grasp_xy if s.grasp_xy is not None else obj[:2]
            return np.array([xy[0], xy[1], half + 0.002]), 1.0
        if ph == "close":
            return None, 0.0
        if ph == "lift":
            tip = obs["tip"]
            return np.array([tip[0], tip[1], self.travel_z]), 0.0
        if ph in ("above_goal", "lower"):
            # visual servoing: place the CUBE (not the tip) over the goal, correcting in-hand slip
            off = s.offset if s.offset is not None else np.zeros(2)
            z = (max(goal[2] + half + ABOVE, self.travel_z - 0.02) if ph == "above_goal"
                 else goal[2] + half + 0.006)
            return np.array([goal[0] + off[0], goal[1] + off[1], z]), 0.0
        if ph == "release":
            return None, 1.0
        if ph == "retreat" or ph == "done":
            tip = obs["tip"]
            return np.array([tip[0], tip[1], self.travel_z]), 1.0
        raise ValueError(ph)

    def _advance(self, obs: Dict, target: Optional[np.ndarray], half: float = 0.015) -> None:
        s = self.s
        s.t += DT
        near = target is not None and np.linalg.norm(obs["tip"] - target) < TOL
        s.settle = s.settle + 1 if near else 0
        reached = s.settle >= 2          # two consecutive ticks: robust to latency overshoot
        timeout = s.t > 6.0
        if s.phase in ("lift", "above_goal", "lower") and obs["obj"][2] > half + 0.01:
            new = obs["tip"][:2] - obs["obj"][:2]
            s.offset = new if s.offset is None else 0.7 * s.offset + 0.3 * new  # low-pass noisy detections
        nxt = None
        if s.phase in ("open", "close", "release"):
            g = obs["q"][5]
            settled = (g < 0.5) if s.phase == "close" else (g > 0.85)
            if (s.t >= GRIP_TIME and settled) or s.t >= 2.5:
                nxt = PHASES[PHASES.index(s.phase) + 1]
        elif s.phase == "done":
            return
        elif reached or (timeout and s.phase in ("lift", "retreat")):
            nxt = PHASES[PHASES.index(s.phase) + 1]
            if s.phase == "above_obj":
                s.grasp_xy = obs["obj"][:2].copy()  # freeze: the arm occludes the cube on the way down
        if nxt:
            s.phase, s.t, s.settle = nxt, 0.0, 0

    # ------------------------------------------------------------------ act
    def act(self, obs: Dict, half: float = 0.015) -> Dict:
        """Returns {"q_cmd": 6 targets, "qd": 4 rad/s, "gripper": 0|1, "phase": str}."""
        target, grip = self.subgoal(obs, half)
        self._advance(obs, target, half)
        target, grip = self.subgoal(obs, half)
        q = obs["q"]
        qd = np.zeros(4)
        if target is not None:
            qd = arm.ik_step(q, target, pitch=PITCH, gain=self.gain)
        if self.s.phase in ("above_obj", "descend"):
            self.s.roll = roll_for_yaw(q[0], obs["obj"][3])
        label_qd = qd.copy()
        if self.dart > 0:  # DART: execute a perturbed action, keep the clean label
            qd = qd + self.rng.normal(0, self.dart, 4) * arm.QD_MAX[:4]
        qc = self.q_cmd.copy()
        qc[:4] = np.clip(qc[:4] + qd * DT, q[:4] - MAX_WINDUP, q[:4] + MAX_WINDUP)
        qc[4] += np.clip(self.s.roll - qc[4], -arm.QD_MAX[4] * DT, arm.QD_MAX[4] * DT)
        qc[5] = grip
        self.q_cmd = arm.clamp_q(qc)
        return {"q_cmd": self.q_cmd.copy(), "qd": label_qd, "gripper": grip, "roll": self.s.roll,
                "phase": self.s.phase}

    @property
    def done(self) -> bool:
        return self.s.phase == "done"


def run_expert(sim: ArmSim, expert: Optional[PickPlaceExpert] = None, max_steps: int = 400,
               record: bool = False) -> Dict:
    expert = expert or PickPlaceExpert(travel_z=sim.spec["travel_z"])
    frames = []
    success_t = None
    for k in range(max_steps):
        obs = sim.observe()
        a = expert.act(obs, sim.cube_half)
        if record:
            frames.append((obs, a))
        sim.step(a["q_cmd"])
        if expert.done:
            if sim.success():
                success_t = sim.t
            break
    return {"success": bool(sim.success()), "steps": k + 1, "t_success": success_t,
            "collision_steps": sim.collision_steps, "frames": frames, "phase": expert.s.phase}
