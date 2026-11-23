"""End-to-end ROS 2 smoke test of the rover (run by CI in a ros:humble container).

Precondition: `ros2 launch qi_vla_ros rover.launch.py backend:=mujoco` is running.
  1. /scan has 36 beams, /odom is published
  2. rclpy Twist on /cmd_vel moves the robot; silence -> watchdog stops it
  3. the APP's rosbridge messages (copied from app/src/connect/transports.js,
     rover mode) drive the robot and receive /odom over ws://:9090
"""

import json
import math
import sys
import threading
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class Probe(Node):
    def __init__(self):
        super().__init__("qi_vla_rover_smoke")
        self.odom = None
        self.scan = None
        self.create_subscription(Odometry, "/odom", lambda m: setattr(self, "odom", m), 10)
        self.create_subscription(LaserScan, "/scan", lambda m: setattr(self, "scan", m), 10)
        self.cmd = self.create_publisher(Twist, "/cmd_vel", 10)

    def pose(self):
        p = self.odom.pose.pose
        q = p.orientation
        return p.position.x, p.position.y, math.atan2(2 * (q.w * q.z), 1 - 2 * q.z * q.z)


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
    check(p.odom is not None, "receiving /odom")
    check(p.scan is not None and len(p.scan.ranges) == 36, "receiving /scan with 36 beams")
    x0, y0, yaw0 = p.pose()
    tw = Twist()
    tw.angular.z = 1.0
    for _ in range(30):
        p.cmd.publish(tw)
        spin_for(p, 0.05)
    spin_for(p, 0.5)
    check(abs(math.atan2(math.sin(p.pose()[2] - yaw0), math.cos(p.pose()[2] - yaw0))) > 0.5, "Twist on /cmd_vel turns the robot")
    spin_for(p, 1.5)
    check(abs(p.odom.twist.twist.angular.z) < 0.05, "watchdog stops the robot after /cmd_vel silence")

    import websocket
    ws = websocket.create_connection("ws://localhost:9090", timeout=5)
    ws.send(json.dumps({"op": "advertise", "topic": "/cmd_vel", "type": "geometry_msgs/msg/Twist"}))
    ws.send(json.dumps({"op": "subscribe", "topic": "/odom", "type": "nav_msgs/msg/Odometry", "throttle_rate": 50}))
    seen = {"n": 0, "w": None}

    def reader():
        while True:
            try:
                m = json.loads(ws.recv())
            except Exception:
                return
            if m.get("op") == "publish" and m.get("topic") == "/odom":
                seen["n"] += 1
                seen["w"] = m["msg"]["twist"]["twist"]["angular"]["z"]
    threading.Thread(target=reader, daemon=True).start()
    for _ in range(40):
        ws.send(json.dumps({"op": "publish", "topic": "/cmd_vel",
                            "msg": {"linear": {"x": 0, "y": 0, "z": 0}, "angular": {"x": 0, "y": 0, "z": -1.0}}}))
        time.sleep(0.05)
    check(seen["n"] > 10, f"app-protocol /odom subscription ({seen['n']} msgs)")
    check(seen["w"] is not None and seen["w"] < -0.3, f"app-protocol Twist drives the robot (w={seen['w']})")
    ws.close()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
