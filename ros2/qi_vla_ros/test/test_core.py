"""ROS-free tests of qi_vla_ros.core (run with plain pytest)."""

import numpy as np

from qi_vla_ros.core import Backend, KinematicServo, joint_state_lists, trajectory_to_q
from src.robot import arm


def test_trajectory_mapping_any_order_and_subset():
    q = trajectory_to_q(["elbow_flex", "shoulder_pan", "unknown"], [1.0, 0.3, 9.0], arm.Q_HOME)
    assert q[0] == 0.3 and q[2] == 1.0 and q[1] == arm.Q_HOME[1]
    assert trajectory_to_q(["shoulder_pan"], [99.0], arm.Q_HOME)[0] == arm.Q_MAX[0]


def test_kinematic_servo_converges_with_latency():
    s = KinematicServo(tau=0.05, latency_steps=2)
    target = arm.Q_HOME.copy()
    target[0] = 0.4
    for _ in range(100):
        s.command(target)
        s.step(0.02)
    assert abs(s.q[0] - 0.4) < 1e-3


def test_joint_state_has_finger_joints():
    v = joint_state_lists(arm.Q_HOME)
    assert len(v) == 8 and abs(v[6] - 0.024) < 1e-9


def test_mujoco_backend_tracks_and_reports_objects():
    b = Backend("mujoco", "lab_table", seed=0)
    names = list(arm.JOINT_NAMES)
    tgt = arm.Q_HOME.copy()
    tgt[0] = 0.3
    for _ in range(40):
        b.command(names, tgt)
        b.step()
    assert abs(b.q()[0] - 0.3) < 0.02
    objs = b.objects()
    assert {"red", "green", "blue", "yellow", "purple"} <= set(objs)


def test_rover_backend_cmd_vel_watchdog_and_scan():
    from qi_vla_ros.core import RoverBackend
    for kind in ("kinematic", "mujoco"):
        b = RoverBackend(kind, "home", seed=1)
        x0 = b.sim.pose()[0:2]
        t = 0.0
        for _ in range(40):
            b.command(0.2, 0.0, t)
            st = b.step(t)
            t += 0.05
        moved = np.hypot(st["pose"][0] - x0[0], st["pose"][1] - x0[1])
        assert moved > 0.2 or b.sim.collision_steps > 0, kind
        assert len(st["scan"]) == 36
        p1 = st["pose"]
        for _ in range(30):  # no new commands: watchdog stops the robot
            st = b.step(t)
            t += 0.05
        assert abs(st["vw"][0]) < 0.02
    assert len(b.map_boxes()) >= len(b.map["walls"])
