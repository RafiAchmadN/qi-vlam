"""qi_vla_rover_sim — simulated QI-VLA rover (diff-drive) for ROS 2.

TurtleBot3/Nav2-compatible interface:
  sub  /cmd_vel (geometry_msgs/Twist)      — watchdog: zero after 0.5 s silence
  pub  /odom (nav_msgs/Odometry, odom -> base_footprint), TF odom -> base_footprint
  pub  /scan (sensor_msgs/LaserScan, 36 beams, frame base_scan)
  pub  /qi_vla/map_markers (MarkerArray: walls, furniture, unknown obstacles, places)
  srv  /qi_vla/reset (std_srvs/Trigger)

The odom frame coincides with the map frame of app/public/maps/<map>.json
(the simulator knows the true pose), so RViz shows the map and robot aligned.

    ros2 run qi_vla_ros qi_vla_rover_sim --ros-args -p map:=warehouse -p backend:=mujoco -p unknown:=2
"""

from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from qi_vla_ros.core import RoverBackend


def _quat(yaw: float):
    return 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)


class RoverNode(Node):
    def __init__(self) -> None:
        super().__init__("qi_vla_rover_sim")
        self.b = RoverBackend(self.declare_parameter("backend", "kinematic").value,
                              self.declare_parameter("map", "home").value,
                              int(self.declare_parameter("seed", 0).value),
                              int(self.declare_parameter("unknown", 0).value))
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.scan_pub = self.create_publisher(LaserScan, "/scan", 10)
        self.map_pub = self.create_publisher(MarkerArray, "/qi_vla/map_markers", 1)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(Twist, "/cmd_vel", self.on_cmd, 10)
        self.create_service(Trigger, "/qi_vla/reset", self.on_reset)
        self.create_timer(0.05, self.tick)
        self.k = 0
        self.get_logger().info(f"qi_vla_rover_sim: map={self.b.map_id} backend={self.b.kind} unknown={len(self.b.ep.unknown)}")

    def on_cmd(self, m: Twist) -> None:
        self.b.command(m.linear.x, m.angular.z, self.get_clock().now().nanoseconds * 1e-9)

    def on_reset(self, _req, res):
        self.b.reset(self.b.seed + 1)
        res.success, res.message = True, f"rover reset (seed {self.b.seed})"
        return res

    def tick(self) -> None:
        st = self.b.step(self.get_clock().now().nanoseconds * 1e-9)
        now = self.get_clock().now().to_msg()
        x, y, yaw = st["pose"]
        qx, qy, qz, qw = (float(v) for v in _quat(yaw))
        o = Odometry()
        o.header.stamp, o.header.frame_id, o.child_frame_id = now, "odom", "base_footprint"
        o.pose.pose.position.x, o.pose.pose.position.y = float(x), float(y)
        o.pose.pose.orientation.z, o.pose.pose.orientation.w = qz, qw
        o.twist.twist.linear.x, o.twist.twist.angular.z = float(st["vw"][0]), float(st["vw"][1])
        self.odom_pub.publish(o)
        t = TransformStamped()
        t.header.stamp, t.header.frame_id, t.child_frame_id = now, "odom", "base_footprint"
        t.transform.translation.x, t.transform.translation.y = float(x), float(y)
        t.transform.rotation.z, t.transform.rotation.w = qz, qw
        self.tf.sendTransform(t)
        s = LaserScan()
        s.header.stamp, s.header.frame_id = now, "base_scan"
        n = len(st["scan"])
        s.angle_min, s.angle_max, s.angle_increment = 0.0, 2 * math.pi * (n - 1) / n, 2 * math.pi / n
        s.scan_time, s.range_min, s.range_max = 0.05, self.b.R["lidar"]["range_min"], self.b.R["lidar"]["range_max"]
        s.ranges = [float(r) for r in st["scan"]]
        self.scan_pub.publish(s)
        if self.k % 40 == 0:
            self.publish_map(now)
        self.k += 1

    def publish_map(self, now) -> None:
        arr = MarkerArray()
        for i, b in enumerate(self.b.map_boxes()):
            mk = Marker()
            mk.header.frame_id, mk.header.stamp = "odom", now
            mk.ns, mk.id, mk.type = "qi_vla_map", i, Marker.CUBE
            # ROS 2 message fields are strictly typed: map JSON holds ints like 8 -> cast to float
            mk.pose.position.x, mk.pose.position.y, mk.pose.position.z = float(b["pos"][0]), float(b["pos"][1]), float(b["h"]) / 2
            mk.pose.orientation.w = 1.0
            mk.scale.x, mk.scale.y, mk.scale.z = float(b["size"][0]), float(b["size"][1]), float(b["h"])
            mk.color.r, mk.color.g, mk.color.b, mk.color.a = (*(float(c) for c in b["rgb"]), 1.0)
            arr.markers.append(mk)
        self.map_pub.publish(arr)


def main() -> None:
    rclpy.init()
    node = RoverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
