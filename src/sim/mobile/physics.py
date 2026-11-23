"""Differential-drive rover simulation (MuJoCo physics, or a fast kinematic model).

`RoverSim` (MuJoCo): a TurtleBot3-Burger-sized rover — two driven wheels with
velocity-controlled motors, two low-friction casters, a 36-beam 360° lidar
(MuJoCo rangefinders) — in a map from `app/public/maps`. Real-world effects:
inertia, wheel/floor friction (slip), motor lag, command latency, wheel-radius
error (commanded vs real speed), lidar and localization noise, collisions.

`KinematicRover`: unicycle + first-order lag + 2-D ray casting
(`geometry.raycast`). Same interface; used by the ROS 2 node's kinematic
backend and as a fast reference.

    sim = RoverSim(sample_nav_episode("home", 0))
    obs = sim.observe()          # pose (noisy), scan (36), v/w commanded last
    sim.step(v, w)               # m/s, rad/s at 20 Hz
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Optional

import mujoco
import numpy as np

from src.sim.mjcf import hex_rgba
from src.sim.mobile.geometry import (NavEpisode, beam_angles, episode_boxes, load_map, raycast,
                                     robot_spec, sample_nav_episode)

DT = 0.05


def rover_mjcf(ep: NavEpisode, dyn: Dict) -> str:
    m = load_map(ep.map_id)
    R = robot_spec()
    w, h = m["size"]
    rw = R["wheel_radius"] * float(dyn.get("wheel_radius_scale", 1.0))
    bt = R["half_track"]
    fr = float(dyn.get("floor_friction", 0.9))
    mass = 0.8 * float(dyn.get("body_mass", 1.0))
    n = R["lidar"]["beams"]
    lh = R["lidar"]["height"]
    walls = "\n".join(
        f'    <geom name="wall{i}" type="box" pos="{b["pos"][0]} {b["pos"][1]} 0.175" '
        f'size="{b["size"][0] / 2} {b["size"][1] / 2} 0.175" rgba="{hex_rgba(m["wall_color"])}"/>'
        for i, b in enumerate(m["walls"]))
    obst = "\n".join(
        f'    <geom name="obs{i}" type="box" pos="{o["pos"][0]} {o["pos"][1]} {o["height"] / 2}" '
        f'size="{o["size"][0] / 2} {o["size"][1] / 2} {o["height"] / 2}" rgba="{hex_rgba(o["color"])}"/>'
        for i, o in enumerate(m["obstacles"] + ep.unknown))
    places = "\n".join(
        f'    <site name="place_{p["id"]}" pos="{p["pos"][0]} {p["pos"][1]} 0.002" size="0.12 0.001" '
        f'type="cylinder" rgba="0.2 0.8 1 0.35"/>' for p in m["places"])
    ang = beam_angles(n)
    sites = "\n".join(f'      <site name="lidar{k}" pos="0 0 {lh}" zaxis="{np.cos(a):.6f} {np.sin(a):.6f} 0" size="0.004"/>'
                      for k, a in enumerate(ang))
    sensors = "\n".join(f'    <rangefinder name="r{k}" site="lidar{k}" cutoff="{R["lidar"]["range_max"]}"/>' for k in range(n))
    x, y, yaw = ep.start
    return f"""<mujoco model="qi_vla_rover_{ep.map_id}">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic"/>
  <visual><global offwidth="640" offheight="480"/><headlight ambient="0.3 0.3 0.3" diffuse="0.4 0.4 0.4" specular="0 0 0"/></visual>
  <default><geom friction="{fr} 0.005 0.0001" solref="0.004 1"/></default>
  <worldbody>
    <light name="top" pos="{w / 2} {h / 2} 4" dir="0 0 -1" diffuse="0.5 0.5 0.5" castshadow="false"/>
    <camera name="top" pos="{w / 2} {h / 2} {max(w, h) * 1.25}" xyaxes="1 0 0 0 1 0" fovy="45"/>
    <geom name="floor" type="plane" pos="{w / 2} {h / 2} 0" size="{w / 2} {h / 2} 0.05" rgba="{hex_rgba(m['floor_color'])}"/>
{walls}
{obst}
{places}
    <body name="rover" pos="{x} {y} {rw + 0.001}" euler="0 0 {yaw}">
      <freejoint name="rover_free"/>
      <geom name="chassis" type="cylinder" size="{R['radius']} 0.03" pos="0 0 0.045" mass="{mass}" rgba="0.22 0.28 0.31 1"/>
      <geom name="lidar_head" type="cylinder" size="0.035 0.02" pos="0 0 {lh - 0.02}" mass="0.1" rgba="0.1 0.1 0.12 1"/>
      <geom name="caster_f" type="sphere" size="0.012" pos="0.075 0 {0.015 - rw}" friction="0.001 0.0001 0.0001" condim="1" mass="0.01" rgba="0.5 0.5 0.5 1"/>
      <geom name="caster_b" type="sphere" size="0.012" pos="-0.075 0 {0.015 - rw}" friction="0.001 0.0001 0.0001" condim="1" mass="0.01" rgba="0.5 0.5 0.5 1"/>
      <body name="wheel_l" pos="0 {bt} 0">
        <joint name="wheel_l" type="hinge" axis="0 1 0" damping="0.0005"/>
        <geom name="wheel_l_geom" type="sphere" size="{rw}" mass="0.05" condim="4" rgba="0 0 0 0"/>
        <geom name="wheel_l_vis" type="cylinder" size="{rw} 0.009" euler="1.5708 0 0" mass="0.001" contype="0" conaffinity="0" rgba="0.1 0.1 0.1 1"/>
      </body>
      <body name="wheel_r" pos="0 {-bt} 0">
        <joint name="wheel_r" type="hinge" axis="0 1 0" damping="0.0005"/>
        <geom name="wheel_r_geom" type="sphere" size="{rw}" mass="0.05" condim="4" rgba="0 0 0 0"/>
        <geom name="wheel_r_vis" type="cylinder" size="{rw} 0.009" euler="1.5708 0 0" mass="0.001" contype="0" conaffinity="0" rgba="0.1 0.1 0.1 1"/>
      </body>
{sites}
    </body>
  </worldbody>
  <actuator>
    <velocity name="m_l" joint="wheel_l" kv="0.06" forcerange="-0.4 0.4"/>
    <velocity name="m_r" joint="wheel_r" kv="0.06" forcerange="-0.4 0.4"/>
  </actuator>
  <sensor>
{sensors}
  </sensor>
</mujoco>
"""


class _Base:
    def _init_common(self, ep: NavEpisode, dynamics: Optional[Dict], seed: Optional[int]) -> None:
        self.episode = ep
        self.map = load_map(ep.map_id)
        self.B = episode_boxes(self.map, ep)
        self.R = robot_spec()
        self.dyn = dict(self.map["dynamics"], body_mass=1.0)
        self.dyn.update(ep.dynamics)
        if dynamics:
            self.dyn.update(dynamics)
        self.rng = np.random.default_rng(ep.seed if seed is None else seed)
        self.queue: deque = deque([(0.0, 0.0)] * int(self.dyn.get("latency_steps", 0)))
        self.cmd = np.zeros(2)       # (v, w) after latency + motor lag
        self.last = np.zeros(2)      # last commanded (what the controller knows)
        self.t = 0.0
        self.collision_steps = 0
        self.path_len = 0.0

    def _filter_cmd(self, v: float, w: float) -> np.ndarray:
        R = self.R
        v = float(np.clip(v, -R["v_max"], R["v_max"]))
        w = float(np.clip(w, -R["w_max"], R["w_max"]))
        self.last = np.array([v, w])
        self.queue.append((v, w))
        target = np.asarray(self.queue.popleft())
        tau = max(float(self.dyn.get("motor_tau", 0.08)), 1e-3)
        self.cmd = self.cmd + (target - self.cmd) * min(DT / tau, 1.0)
        return self.cmd

    def observe(self) -> Dict:
        x, y, yaw = self.pose()
        pn = float(self.dyn.get("pose_noise", 0.0))
        ln = float(self.dyn.get("lidar_noise", 0.0))
        scan = np.clip(self.scan() + self.rng.normal(0, ln, self.R["lidar"]["beams"]),
                       self.R["lidar"]["range_min"], self.R["lidar"]["range_max"])
        pose = np.array([x + self.rng.normal(0, pn), y + self.rng.normal(0, pn), yaw + self.rng.normal(0, pn)])
        return {"pose": pose, "scan": scan, "vw": self.last.copy(), "goal": np.asarray(self.episode.goal), "t": self.t}

    def at_goal(self, tol: float = 0.2) -> bool:
        x, y, _ = self.pose()
        return bool(np.hypot(x - self.episode.goal[0], y - self.episode.goal[1]) < tol)

    def snapshot(self) -> Dict:
        x, y, yaw = self.pose()
        return {"pose": [round(x, 4), round(y, 4), round(yaw, 4)], "vw": self.last.round(4).tolist(),
                "map": self.episode.map_id, "goal": list(self.episode.goal), "t": round(self.t, 3)}


class RoverSim(_Base):
    def __init__(self, ep: NavEpisode, dynamics: Optional[Dict] = None, seed: Optional[int] = None) -> None:
        self._init_common(ep, dynamics, seed)
        self.model = mujoco.MjModel.from_xml_string(rover_mjcf(ep, self.dyn))
        self.data = mujoco.MjData(self.model)
        self.n_sub = int(round(DT / self.model.opt.timestep))
        m = self.model
        self.body = m.body("rover").id
        rbodies = {m.body(n).id for n in ("rover", "wheel_l", "wheel_r")}
        self.robot_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in rbodies}
        self.floor = m.geom("floor").id
        self.obstacle_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == 0 and g != self.floor}
        mujoco.mj_forward(m, self.data)
        for _ in range(50):
            mujoco.mj_step(m, self.data)

    def pose(self):
        p = self.data.xpos[self.body]
        R = self.data.xmat[self.body].reshape(3, 3)
        return float(p[0]), float(p[1]), float(np.arctan2(R[1, 0], R[0, 0]))

    def scan(self) -> np.ndarray:
        s = self.data.sensordata[:self.R["lidar"]["beams"]].copy()
        s[s < 0] = self.R["lidar"]["range_max"]  # -1 = no hit within cutoff
        return s

    def step(self, v: float, w: float) -> None:
        v, w = self._filter_cmd(v, w)
        r, b = self.R["wheel_radius"], self.R["half_track"]   # controller uses NOMINAL geometry
        self.data.ctrl[0] = (v - w * b) / r
        self.data.ctrl[1] = (v + w * b) / r
        x0, y0, _ = self.pose()
        hit = False
        for _ in range(self.n_sub):
            mujoco.mj_step(self.model, self.data)
            hit |= self._collides()
        x1, y1, _ = self.pose()
        self.path_len += float(np.hypot(x1 - x0, y1 - y0))
        self.collision_steps += int(hit)
        self.t += DT

    def _collides(self) -> bool:
        d = self.data
        for i in range(d.ncon):
            c = d.contact[i]
            a, b = c.geom1, c.geom2
            if (a in self.robot_geoms and b in self.obstacle_geoms) or (b in self.robot_geoms and a in self.obstacle_geoms):
                return True
        return False

    def render(self, width: int = 320, height: int = 240) -> np.ndarray:
        r = mujoco.Renderer(self.model, height=height, width=width)
        r.update_scene(self.data, camera="top")
        img = r.render()
        r.close()
        return img


class KinematicRover(_Base):
    """Unicycle with motor lag/latency, 2-D ray-cast lidar, box collisions (no slip)."""

    def __init__(self, ep: NavEpisode, dynamics: Optional[Dict] = None, seed: Optional[int] = None) -> None:
        self._init_common(ep, dynamics, seed)
        self.x, self.y, self.yaw = ep.start

    def pose(self):
        return self.x, self.y, self.yaw

    def scan(self) -> np.ndarray:
        L = self.R["lidar"]
        return raycast(self.B, self.x, self.y, self.yaw, L["beams"], L["range_max"])

    def step(self, v: float, w: float) -> None:
        v, w = self._filter_cmd(v, w)
        v *= float(self.dyn.get("wheel_radius_scale", 1.0))
        nx = self.x + v * np.cos(self.yaw) * DT
        ny = self.y + v * np.sin(self.yaw) * DT
        from src.sim.mobile.geometry import clearance
        if clearance(self.B, nx, ny) > self.R["radius"]:
            self.path_len += float(np.hypot(nx - self.x, ny - self.y))
            self.x, self.y = nx, ny
        else:
            self.collision_steps += 1
        self.yaw = float((self.yaw + w * DT + np.pi) % (2 * np.pi) - np.pi)
        self.t += DT


def make_rover(ep: NavEpisode, backend: str = "mujoco", dynamics: Optional[Dict] = None):
    return RoverSim(ep, dynamics) if backend == "mujoco" else KinematicRover(ep, dynamics)


__all__ = ["RoverSim", "KinematicRover", "make_rover", "sample_nav_episode", "DT"]
