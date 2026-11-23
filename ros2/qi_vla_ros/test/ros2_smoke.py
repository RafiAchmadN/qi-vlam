"""End-to-end ROS 2 smoke test (run by CI inside a ros:humble container).

Precondition: `ros2 launch qi_vla_ros sim.launch.py backend:=mujoco policy:=true`
is running (sim node + robot_state_publisher + rosbridge + policy node).

  1. rclpy:     JointTrajectory on /arm_controller/joint_trajectory -> /joint_states follows
  2. rosbridge: the APP's exact rosbridge messages (advertise/subscribe/publish,
                copied from app/src/connect/transports.js) drive the arm over ws://:9090
  3. policy:    /qi_vla/target -> qi_vla_policy (TTNS reach policy) brings the tip there
"""

import json
import sys
import threading
import time

import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from qi_vla_ros.core import ensure_repo_on_path

ensure_repo_on_path()
from src.robot import arm  # noqa: E402

CMD = "/arm_controller/joint_trajectory"
NAMES = list(arm.JOINT_NAMES)


class Probe(Node):
    def __init__(self):
        super().__init__("qi_vla_smoke")
        self.q = None
        self.create_subscription(JointState, "/joint_states", self.on_js, 10)
        self.cmd = self.create_publisher(JointTrajectory, CMD, 10)
        self.tgt = self.create_publisher(PointStamped, "/qi_vla/target", 10)

    def on_js(self, m):
        self.q = [m.position[m.name.index(n)] for n in NAMES]


def spin_for(node, s):
    t = time.time()
    while time.time() - t < s:
        rclpy.spin_once(node, timeout_sec=0.02)


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        sys.exit(1)


def main():
    rclpy.init()
    p = Probe()
    spin_for(p, 3.0)
    check(p.q is not None, "receiving /joint_states")

    # 1) plain ROS 2 command
    tgt = list(arm.Q_HOME)
    tgt[0] = 0.5
    for _ in range(60):
        p.cmd.publish(JointTrajectory(joint_names=NAMES, points=[JointTrajectoryPoint(
            positions=tgt, time_from_start=Duration(sec=0, nanosec=100_000_000))]))
        spin_for(p, 0.05)
    spin_for(p, 1.0)
    check(abs(p.q[0] - 0.5) < 0.05, f"rclpy JointTrajectory tracked (pan={p.q[0]:.3f})")

    # 2) the app's rosbridge protocol
    import websocket  # websocket-client
    ws = websocket.create_connection("ws://localhost:9090", timeout=5)
    ws.send(json.dumps({"op": "advertise", "topic": CMD, "type": "trajectory_msgs/msg/JointTrajectory"}))
    ws.send(json.dumps({"op": "subscribe", "topic": "/joint_states", "type": "sensor_msgs/msg/JointState",
                        "throttle_rate": 50}))
    seen = {"n": 0, "q": None}

    def reader():
        while True:
            try:
                m = json.loads(ws.recv())
            except Exception:
                return
            if m.get("op") == "publish" and m.get("topic") == "/joint_states":
                name, pos = m["msg"]["name"], m["msg"]["position"]
                seen["q"] = [pos[name.index(n)] for n in NAMES]
                seen["n"] += 1
    threading.Thread(target=reader, daemon=True).start()
    tgt = list(arm.Q_HOME)
    tgt[0] = -0.4
    for _ in range(80):
        ws.send(json.dumps({"op": "publish", "topic": CMD, "msg": {
            "joint_names": NAMES, "points": [{"positions": tgt, "velocities": [], "accelerations": [],
                                              "effort": [], "time_from_start": {"sec": 0, "nanosec": 100000000}}]}}))
        time.sleep(0.05)
    time.sleep(1.0)
    check(seen["n"] > 10, f"app-protocol subscription receives joint_states ({seen['n']} msgs)")
    check(seen["q"] is not None and abs(seen["q"][0] + 0.4) < 0.05,
          f"app-protocol command tracked over rosbridge (pan={seen['q'][0] if seen['q'] else None})")
    ws.close()

    # 3) policy node
    m = PointStamped()
    m.header.frame_id = "base_link"
    m.point.x, m.point.y, m.point.z = 0.2, 0.05, 0.08
    for _ in range(3):
        p.tgt.publish(m)
        spin_for(p, 0.2)
    spin_for(p, 8.0)
    err = sum((a - b) ** 2 for a, b in zip(arm.fk(p.q), (0.2, 0.05, 0.08))) ** 0.5
    check(err < 0.02, f"qi_vla_policy reached target (err={1000 * err:.1f} mm)")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
