"""qi_vla_sim — simulated QI-VLA arm for ROS 2.

Subscribes  <cmd_topic> (trajectory_msgs/JointTrajectory, default
            /arm_controller/joint_trajectory — the app's rosbridge default)
Publishes   /joint_states (sensor_msgs/JointState), and with backend:=mujoco
            /qi_vla/objects (visualization_msgs/MarkerArray, frame base_link)
Service     /qi_vla/reset (std_srvs/Trigger): new random scene (mujoco)

    ros2 run qi_vla_ros qi_vla_sim --ros-args -p backend:=mujoco -p env:=kitchen_counter
"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory
from visualization_msgs.msg import Marker, MarkerArray

from qi_vla_ros.core import COLORS, STATE_NAMES, Backend, joint_state_lists


class SimNode(Node):
    def __init__(self) -> None:
        super().__init__("qi_vla_sim")
        backend = self.declare_parameter("backend", "kinematic").value
        env = self.declare_parameter("env", "lab_table").value
        seed = int(self.declare_parameter("seed", 0).value)
        cmd_topic = self.declare_parameter("cmd_topic", "/arm_controller/joint_trajectory").value
        self.backend = Backend(backend, env, seed, bool(self.declare_parameter("randomize", False).value))
        self.pub = self.create_publisher(JointState, "/joint_states", 10)
        self.obj_pub = self.create_publisher(MarkerArray, "/qi_vla/objects", 10)
        self.create_subscription(JointTrajectory, cmd_topic, self.on_cmd, 10)
        self.create_service(Trigger, "/qi_vla/reset", self.on_reset)
        self.create_timer(self.backend.dt, self.tick)
        self.k = 0
        self.get_logger().info(f"qi_vla_sim: backend={backend} env={env} cmd={cmd_topic}")

    def on_cmd(self, msg: JointTrajectory) -> None:
        if msg.points:
            self.backend.command(list(msg.joint_names), list(msg.points[-1].positions))

    def on_reset(self, _req, res):
        self.backend.reset(self.backend.seed + 1)
        res.success, res.message = True, f"scene reset (seed {self.backend.seed})"
        return res

    def tick(self) -> None:
        q = self.backend.step()
        m = JointState()
        m.header.stamp = self.get_clock().now().to_msg()
        m.name = STATE_NAMES
        m.position = joint_state_lists(q)
        self.pub.publish(m)
        self.k += 1
        if self.backend.kind == "mujoco" and self.k % 4 == 0:
            self.publish_objects()

    def publish_objects(self) -> None:
        arr = MarkerArray()
        for i, (name, o) in enumerate(self.backend.objects().items()):
            mk = Marker()
            mk.header.frame_id = "base_link"
            mk.header.stamp = self.get_clock().now().to_msg()
            mk.ns, mk.id = "qi_vla", i
            mk.type = Marker.CUBE if o["kind"] == "cube" else Marker.CYLINDER
            mk.pose.position = Point(x=o["pos"][0], y=o["pos"][1], z=o["pos"][2])
            import math
            mk.pose.orientation.z, mk.pose.orientation.w = math.sin(o["yaw"] / 2), math.cos(o["yaw"] / 2)
            s = o["size"]
            mk.scale.x = mk.scale.y = s
            mk.scale.z = s if o["kind"] == "cube" else 0.002
            mk.color.r, mk.color.g, mk.color.b = COLORS.get(name, (0.8, 0.8, 0.8))
            mk.color.a = 1.0
            arr.markers.append(mk)
        self.obj_pub.publish(arr)


def main() -> None:
    rclpy.init()
    node = SimNode()
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
