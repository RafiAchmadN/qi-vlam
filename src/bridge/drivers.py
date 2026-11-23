"""Robot drivers for the QI-VLA bridge.

A driver turns joint-space commands in the app's canonical convention
(radians, `src/robot/arm.py`; gripper normalized 0 closed .. 1 open) into
whatever the hardware speaks, and reports measured joint positions back in
the same convention.

  MockDriver    software robot with first-order servo lag (no hardware)
  SerialDriver  JSON-lines over USB serial to an Arduino/ESP32 servo arm
                (firmware: firmware/arduino_serial_arm). Per-joint
                calibration maps radians -> servo degrees.
  MujocoDriver  physics "digital twin": the MuJoCo arm in an environment
                preset (src/sim), with friction grasping and a servo model
                (lag, latency, backlash, noise). Also reports object poses so
                the app can mirror the simulated scene.

Adding a robot = subclass `Driver` and implement read/write/stop.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.robot import arm


class Driver:
    name = "base"

    def connect(self) -> None: ...
    def read(self) -> List[float]: raise NotImplementedError
    def write(self, q: List[float]) -> None: raise NotImplementedError
    def stop(self) -> None:
        """Hold the current measured pose."""
        self.write(self.read())
    def close(self) -> None: ...

    def info(self) -> Dict:
        return {"robot": self.name, "joints": arm.JOINT_NAMES,
                "q_min": arm.Q_MIN.tolist(), "q_max": arm.Q_MAX.tolist()}


class MockDriver(Driver):
    """First-order lag servo model: q' = (q_cmd - q) / tau, slew-limited."""

    name = "mock-arm"

    def __init__(self, tau: float = 0.12, noise: float = 0.002, seed: int = 0) -> None:
        self.tau, self.noise = tau, noise
        self.q = arm.Q_HOME.copy()
        self.cmd = arm.Q_HOME.copy()
        self.t = time.monotonic()
        self.rng = np.random.default_rng(seed)
        self.lock = threading.Lock()

    def _advance(self) -> None:
        now = time.monotonic()
        dt, self.t = min(now - self.t, 0.1), now
        dq = (self.cmd - self.q) * min(dt / self.tau, 1.0)
        dq = np.clip(dq, -arm.QD_MAX * dt * 1.2, arm.QD_MAX * dt * 1.2)
        self.q = arm.clamp_q(self.q + dq)

    def read(self) -> List[float]:
        with self.lock:
            self._advance()
            return (self.q + self.rng.normal(0, self.noise, 6) * (self.noise > 0)).tolist()

    def write(self, q: List[float]) -> None:
        with self.lock:
            self._advance()
            self.cmd = arm.clamp_q(np.asarray(q, dtype=np.float64))

    def stop(self) -> None:
        with self.lock:
            self._advance()
            self.cmd = self.q.copy()


@dataclass
class JointCal:
    """servo_deg = center_deg + sign * scale * degrees(q)   (gripper: lerp closed..open)."""
    center_deg: float = 90.0
    sign: float = 1.0
    scale: float = 1.0


@dataclass
class SerialConfig:
    port: str = "COM3"
    baud: int = 115200
    cal: List[JointCal] = field(default_factory=lambda: [JointCal() for _ in range(5)])
    gripper_closed_deg: float = 30.0
    gripper_open_deg: float = 90.0


class SerialDriver(Driver):
    """Protocol (one JSON object per line, both directions):
         host -> device: {"cmd":"q","deg":[d0..d5]}   |  {"cmd":"stop"}
         device -> host: {"deg":[d0..d5]}   (measured or last-commanded servo angles)
    Hobby servos usually have no position feedback; the firmware then echoes
    the last commanded angles, so the "measured" state is only as good as
    the servo's tracking — say so when reporting results.
    """

    name = "serial-servo-arm"

    def __init__(self, cfg: SerialConfig) -> None:
        self.cfg = cfg
        self.ser = None
        self.last_q = arm.Q_HOME.tolist()
        self.lock = threading.Lock()

    def connect(self) -> None:
        import serial  # pyserial
        self.ser = serial.Serial(self.cfg.port, self.cfg.baud, timeout=0.05)
        time.sleep(2.0)  # most Arduinos reset on open

    def to_deg(self, q: List[float]) -> List[float]:
        out = [c.center_deg + c.sign * c.scale * float(np.degrees(v)) for c, v in zip(self.cfg.cal, q[:5])]
        g = self.cfg
        out.append(g.gripper_closed_deg + (g.gripper_open_deg - g.gripper_closed_deg) * float(np.clip(q[5], 0, 1)))
        return [float(np.clip(d, 0, 180)) for d in out]

    def from_deg(self, deg: List[float]) -> List[float]:
        q = [float(np.radians((d - c.center_deg) / (c.sign * c.scale))) for c, d in zip(self.cfg.cal, deg[:5])]
        g = self.cfg
        q.append(float(np.clip((deg[5] - g.gripper_closed_deg) / (g.gripper_open_deg - g.gripper_closed_deg), 0, 1)))
        return q

    def read(self) -> List[float]:
        if self.ser is None:
            return self.last_q
        with self.lock:
            line = None
            while self.ser.in_waiting:
                line = self.ser.readline()
        if line:
            try:
                self.last_q = self.from_deg(json.loads(line)["deg"])
            except (ValueError, KeyError):
                pass
        return self.last_q

    def write(self, q: List[float]) -> None:
        q = arm.clamp_q(np.asarray(q, dtype=np.float64)).tolist()
        if self.ser is not None:
            with self.lock:
                self.ser.write((json.dumps({"cmd": "q", "deg": [round(d, 2) for d in self.to_deg(q)]}) + "\n").encode())

    def stop(self) -> None:
        if self.ser is not None:
            with self.lock:
                self.ser.write(b'{"cmd":"stop"}\n')

    def close(self) -> None:
        if self.ser is not None:
            self.ser.close()


class MujocoDriver(Driver):
    """Real-time MuJoCo simulation running in a background thread (20 Hz control)."""

    name = "mujoco-sim-arm"

    def __init__(self, env_id: str = "lab_table", seed: int = 0, randomize: bool = False,
                 realtime: bool = True) -> None:
        self.env_id, self.seed, self.randomize, self.realtime = env_id, seed, randomize, realtime
        self.lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.reset_scene(env_id, seed)

    def reset_scene(self, env_id: Optional[str] = None, seed: Optional[int] = None) -> Dict:
        from src.sim.envs import sample_episode
        from src.sim.physics import ArmSim
        with self.lock:
            self.env_id = env_id or self.env_id
            self.seed = self.seed + 1 if seed is None else int(seed)
            self.sim = ArmSim(sample_episode(self.env_id, self.seed, randomize=self.randomize))
            self.cmd = arm.Q_HOME.copy()
            return self.sim.snapshot()

    def connect(self) -> None:
        if self.realtime and not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def _loop(self) -> None:
        from src.sim.physics import DT
        nxt = time.monotonic()
        while self._running:
            self.advance()
            nxt += DT
            time.sleep(max(0.0, nxt - time.monotonic()))

    def advance(self) -> None:
        with self.lock:
            self.sim.step(self.cmd)

    def read(self) -> List[float]:
        with self.lock:
            return self.sim.observe()["q"].tolist()

    def write(self, q: List[float]) -> None:
        with self.lock:
            self.cmd = arm.clamp_q(np.asarray(q, dtype=np.float64))

    def stop(self) -> None:
        with self.lock:
            self.cmd = self.sim.q_true()

    def close(self) -> None:
        self._running = False

    def scene(self) -> Dict:
        with self.lock:
            snap = self.sim.snapshot()
            snap["success"] = bool(self.sim.success())
            snap["instruction"] = self.sim.episode.instruction
            return snap

    def info(self) -> Dict:
        from src.sim.envs import env_index
        return dict(super().info(), sim=True, env=self.env_id, envs=env_index()["envs"])


class RoverDriver:
    """Simulated differential-drive rover (src/sim/mobile) for the bridge.

    Speaks twist commands instead of joint targets. Safety: velocity clamp to
    the robot's limits, a watchdog that zeroes the command if the app stops
    sending (0.5 s), and the bridge's e-stop latch (zero velocity)."""

    kind = "rover"
    name = "mujoco-rover"

    def __init__(self, map_id: str = "home", seed: int = 0, randomize: bool = False,
                 backend: str = "mujoco", unknown: int = 0, realtime: bool = True,
                 watchdog_s: float = 0.5) -> None:
        self.map_id, self.seed, self.randomize, self.backend = map_id, seed, randomize, backend
        self.unknown, self.realtime, self.watchdog_s = unknown, realtime, watchdog_s
        self.lock = threading.Lock()
        self._running = False
        self.reset_scene(map_id, seed)

    def reset_scene(self, map_id: Optional[str] = None, seed: Optional[int] = None) -> Dict:
        from src.sim.mobile.geometry import sample_nav_episode
        from src.sim.mobile.physics import make_rover
        with self.lock:
            self.map_id = map_id or self.map_id
            self.seed = self.seed + 1 if seed is None else int(seed)
            self.ep = sample_nav_episode(self.map_id, self.seed, randomize=self.randomize, unknown=self.unknown)
            self.sim = make_rover(self.ep, self.backend)
            self.cmd = (0.0, 0.0)
            self.t_cmd = time.monotonic()
            self.watchdog_trips = 0
            return self.scene()

    def connect(self) -> None:
        if self.realtime and not self._running:
            self._running = True
            threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        from src.sim.mobile.physics import DT
        nxt = time.monotonic()
        while self._running:
            self.advance()
            nxt += DT
            time.sleep(max(0.0, nxt - time.monotonic()))

    def advance(self) -> None:
        with self.lock:
            if time.monotonic() - self.t_cmd > self.watchdog_s and self.cmd != (0.0, 0.0):
                self.cmd = (0.0, 0.0)
                self.watchdog_trips += 1
            self.sim.step(*self.cmd)

    def twist(self, v: float, w: float) -> Optional[tuple]:
        from src.sim.mobile.geometry import robot_spec
        R = robot_spec()
        if not (np.isfinite(v) and np.isfinite(w)):
            return None
        with self.lock:
            self.cmd = (float(np.clip(v, -R["v_max"], R["v_max"])), float(np.clip(w, -R["w_max"], R["w_max"])))
            self.t_cmd = time.monotonic()
            return self.cmd

    def read_state(self) -> Dict:
        with self.lock:
            o = self.sim.observe()
            return {"pose": o["pose"].round(4).tolist(), "vw": o["vw"].round(4).tolist(),
                    "scan": o["scan"].round(3).tolist(), "map": self.map_id}

    def read(self) -> List[float]:
        return []

    def write(self, q: List[float]) -> None:
        raise ValueError("rover: use twist_command")

    def stop(self) -> None:
        with self.lock:
            self.cmd = (0.0, 0.0)

    def close(self) -> None:
        self._running = False

    def scene(self) -> Dict:
        return {"map": self.map_id, "unknown": self.ep.unknown, "start": list(self.ep.start),
                "pose": list(self.sim.pose()), "collisions": self.sim.collision_steps}

    def info(self) -> Dict:
        from src.sim.mobile.geometry import map_index, robot_spec
        return {"robot": self.name, "kind": "rover", "sim": True, "map": self.map_id,
                "maps": map_index()["maps"], "spec": robot_spec()}


def make_driver(kind: str, port: Optional[str] = None, cal_path: Optional[str] = None,
                env_id: str = "lab_table", seed: int = 0, randomize: bool = False,
                map_id: str = "home", unknown: int = 0):
    if kind == "mock":
        return MockDriver()
    if kind == "mujoco":
        return MujocoDriver(env_id, seed, randomize)
    if kind in ("rover", "rover-kinematic"):
        return RoverDriver(map_id, seed, randomize, "mujoco" if kind == "rover" else "kinematic", unknown)
    if kind == "serial":
        cfg = SerialConfig(port=port or "COM3")
        if cal_path:
            raw = json.loads(open(cal_path, encoding="utf-8").read())
            cfg.cal = [JointCal(**c) for c in raw["joints"]]
            cfg.gripper_closed_deg = raw.get("gripper_closed_deg", cfg.gripper_closed_deg)
            cfg.gripper_open_deg = raw.get("gripper_open_deg", cfg.gripper_open_deg)
        return SerialDriver(cfg)
    raise ValueError(f"unknown driver {kind!r} (mock | serial | mujoco | rover | rover-kinematic)")
