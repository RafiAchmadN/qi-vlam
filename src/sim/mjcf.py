"""MJCF (MuJoCo XML) generator for the arm + an environment preset.

The kinematic chain is built from the constants in `src/robot/arm.py`, so the
MuJoCo arm's gripper-tip site coincides with `arm.fk(q)` (asserted in
tests/test_sim.py). Joint convention matches arm.py: q1/q2/q3 are rotations
about the link-local +y axis, which maps the link's +z direction to
(sin a, 0, cos a) — exactly `r = L sin(a)`, `z = L cos(a)`.

    python -m src.sim.mjcf --env kitchen_counter --out sim/kitchen_counter.xml
    python -m mujoco.viewer --mjcf=sim/kitchen_counter.xml      # interactive view

Contact classes: arm geoms (contype 2, conaffinity 1) collide with the
world and objects but never with each other; everything else is (1, 1).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

from src.robot import arm
from src.sim.envs import CUBE, EpisodeConfig, load_env

RGBA = {"red": "0.90 0.22 0.21 1", "green": "0.26 0.63 0.28 1", "blue": "0.12 0.53 0.90 1",
        "yellow": "0.99 0.85 0.21 1", "purple": "0.56 0.14 0.67 1",
        "orange": "0.98 0.45 0.09 1", "cyan": "0.00 0.75 0.65 1"}
FINGER_OPEN = 0.024   # finger travel per side (m) at gripper = 1 (fully open)
CUBE_GAP = 0.0        # fingers close to the centre line; the cube stops them


def hex_rgba(h: str, a: float = 1.0) -> str:
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return f"{r:.3f} {g:.3f} {b:.3f} {a}"


def arm_xml(link_mass: float = 1.0) -> str:
    H0, L1, L2, L3 = arm.LINKS
    m = lambda v: f"{v * link_mass:.4f}"  # noqa: E731
    lo, hi = arm.Q_MIN, arm.Q_MAX
    return f"""
    <body name="base" pos="0 0 0">
      <geom name="base_geom" class="arm" type="cylinder" size="0.05 0.015" pos="0 0 0.015" rgba="0.22 0.28 0.31 1" mass="{m(0.3)}"/>
      <body name="pan_link" pos="0 0 0">
        <joint name="shoulder_pan" axis="0 0 1" range="{lo[0]} {hi[0]}"/>
        <geom class="arm" type="cylinder" fromto="0 0 0.03 0 0 {H0 - 0.02}" size="0.022" rgba="0.22 0.28 0.31 1" mass="{m(0.12)}"/>
        <body name="upper_arm" pos="0 0 {H0}">
          <joint name="shoulder_lift" axis="0 1 0" range="{lo[1]} {hi[1]}"/>
          <geom class="arm" type="capsule" fromto="0 0 0 0 0 {L1}" size="0.018" rgba="0.69 0.75 0.77 1" mass="{m(0.10)}"/>
          <body name="forearm" pos="0 0 {L1}">
            <joint name="elbow_flex" axis="0 1 0" range="{lo[2]} {hi[2]}"/>
            <geom class="arm" type="capsule" fromto="0 0 0 0 0 {L2}" size="0.015" rgba="0.69 0.75 0.77 1" mass="{m(0.08)}"/>
            <body name="wrist" pos="0 0 {L2}">
              <joint name="wrist_flex" axis="0 1 0" range="{lo[3]} {hi[3]}"/>
              <geom class="arm" type="capsule" fromto="0 0 0 0 0 {L3 - 0.05}" size="0.012" rgba="0.69 0.75 0.77 1" mass="{m(0.03)}"/>
              <body name="hand" pos="0 0 0">
                <joint name="wrist_roll" axis="0 0 1" range="{lo[4]} {hi[4]}"/>
                <geom name="palm" class="arm" type="box" pos="0 0 {L3 - 0.042}" size="0.012 0.03 0.006" rgba="0.22 0.28 0.31 1" mass="{m(0.03)}"/>
                <site name="tip" pos="0 0 {L3}" size="0.004" rgba="1 1 1 0.6"/>
                <body name="finger_l" pos="0 0 {L3 - 0.016}">
                  <joint name="finger_l" type="slide" axis="0 1 0" range="0 {FINGER_OPEN}"/>
                  <geom name="finger_l_geom" class="finger" pos="0 0.004 0" size="0.009 0.004 0.026"/>
                </body>
                <body name="finger_r" pos="0 0 {L3 - 0.016}">
                  <joint name="finger_r" type="slide" axis="0 -1 0" range="0 {FINGER_OPEN}"/>
                  <geom name="finger_r_geom" class="finger" pos="0 -0.004 0" size="0.009 0.004 0.026"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>"""


def build_mjcf(env_id: str = "lab_table", episode: Optional[EpisodeConfig] = None,
               dynamics: Optional[Dict] = None) -> str:
    spec = load_env(env_id)
    dyn = dict(spec["dynamics"], link_mass=1.0, cube_size=1.0)
    if episode is not None:
        dyn.update(episode.dynamics)
    if dynamics:
        dyn.update(dynamics)
    t = spec["table"]
    lt = spec["lighting"]
    fr = float(t.get("friction", 0.9)) * float(dyn.get("friction", 1.0))
    half = CUBE / 2 * float(dyn.get("cube_size", 1.0))
    kp = 40.0 * float(dyn.get("kp_scale", 1.0))
    obstacles = "\n".join(
        f'    <geom name="obs_{o["name"]}" type="box" pos="{o["pos"][0]} {o["pos"][1]} {o["pos"][2]}" '
        f'size="{o["size"][0] / 2} {o["size"][1] / 2} {o["size"][2] / 2}" rgba="{hex_rgba(o["color"])}"'
        + (' contype="0" conaffinity="0"' if o.get("visual_only") else "") + "/>"
        for o in spec["obstacles"])
    cubes_xy = episode.cubes if episode else {c: (0.2, -0.1 + 0.1 * i, 0.0) for i, c in enumerate(spec["cubes"])}
    pads_xy = episode.pads if episode else {c: (0.25, -0.1 + 0.2 * i) for i, c in enumerate(spec["pads"])}
    shapes = episode.shapes if episode else spec.get("shapes", {})

    def gsize(c):
        if shapes.get(c, "cube") == "cylinder":
            return f'type="cylinder" size="{half} {half}"'
        return f'type="box" size="{half} {half} {half}"'
    cubes = "\n".join(
        f'    <body name="cube_{c}" pos="{x} {y} {half + 0.0005}" euler="0 0 {yaw}">'
        f'<freejoint name="cube_{c}_free"/>'
        f'<geom name="cube_{c}_geom" {gsize(c)} rgba="{RGBA[c]}" '
        f'mass="{float(dyn.get("cube_mass", 0.02))}" friction="{1.0 * float(dyn.get("friction", 1.0))} 0.01 0.0005" '
        f'condim="4"/></body>'
        for c, (x, y, yaw) in cubes_xy.items())
    pads = "\n".join(
        f'    <geom name="pad_{c}" type="cylinder" pos="{x} {y} 0.001" size="0.03 0.0005" '
        f'rgba="{RGBA[c]}" contype="0" conaffinity="0"/>'
        for c, (x, y) in pads_xy.items())
    tint = hex_rgba(lt.get("tint", "#ffffff"))
    s = max(lt["sun"] * 0.32, 0.15)
    sp = lt["sun_pos"]
    return f"""<mujoco model="qi_vla_{env_id}">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <visual><global offwidth="640" offheight="480"/><quality shadowsize="2048"/><headlight ambient="0.18 0.18 0.18" diffuse="0.2 0.2 0.2" specular="0 0 0"/></visual>
  <default>
    <joint damping="0.6" armature="0.005"/>
    <geom friction="{fr} 0.005 0.0001" solref="0.004 1" solimp="0.95 0.99 0.001"/>
    <default class="arm"><geom contype="2" conaffinity="1" condim="3"/></default>
    <default class="finger"><geom type="box" contype="2" conaffinity="1" condim="4" friction="1.2 0.01 0.0005" rgba="0.22 0.28 0.31 1" mass="0.01"/></default>
    <position ctrllimited="true"/>
  </default>
  <worldbody>
    <light name="sun" pos="{sp[0]} {sp[1]} {sp[2]}" dir="{-sp[0]} {-sp[1]} {-sp[2]}" diffuse="{s} {s} {s}" castshadow="true"/>
    <light name="fill" pos="0.2 0 1.5" dir="0 0 -1" diffuse="{0.12 * lt['ambient']:.3f} {0.12 * lt['ambient']:.3f} {0.12 * lt['ambient']:.3f}" castshadow="false"/>
    <camera name="top" pos="0.18 0 1.0" xyaxes="0 -1 0 1 0 0" fovy="30"/>
    <camera name="side" pos="0.17 -0.62 0.42" xyaxes="1 0 0 0 0.6 0.8" fovy="50"/>
    <geom name="table" type="box" pos="{t['center'][0]} {t['center'][1]} -0.01" size="{t['size'][0] / 2} {t['size'][1] / 2} 0.01" rgba="{hex_rgba(t['color'])}"/>
{obstacles}
{pads}
{arm_xml(float(dyn.get('link_mass', 1.0)))}
{cubes}
  </worldbody>
  <actuator>
    <position name="a_pan" joint="shoulder_pan" kp="{kp}" ctrlrange="{arm.Q_MIN[0]} {arm.Q_MAX[0]}" forcerange="-2.5 2.5"/>
    <position name="a_lift" joint="shoulder_lift" kp="{kp * 1.5}" ctrlrange="{arm.Q_MIN[1]} {arm.Q_MAX[1]}" forcerange="-3.5 3.5"/>
    <position name="a_elbow" joint="elbow_flex" kp="{kp}" ctrlrange="{arm.Q_MIN[2]} {arm.Q_MAX[2]}" forcerange="-2.5 2.5"/>
    <position name="a_wflex" joint="wrist_flex" kp="{kp * 0.5}" ctrlrange="{arm.Q_MIN[3]} {arm.Q_MAX[3]}" forcerange="-1.5 1.5"/>
    <position name="a_roll" joint="wrist_roll" kp="{kp * 0.25}" ctrlrange="{arm.Q_MIN[4]} {arm.Q_MAX[4]}" forcerange="-1 1"/>
    <position name="a_finger_l" joint="finger_l" kp="300" ctrlrange="0 {FINGER_OPEN}" forcerange="-4 4"/>
    <position name="a_finger_r" joint="finger_r" kp="300" ctrlrange="0 {FINGER_OPEN}" forcerange="-4 4"/>
  </actuator>
</mujoco>
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="lab_table")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    xml = build_mjcf(a.env)
    out = Path(a.out or f"sim/{a.env}.xml")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(xml, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
