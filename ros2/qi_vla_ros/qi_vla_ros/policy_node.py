"""qi_vla_policy — runs an exported reach policy (dense or TTNS-compressed)
as a ROS 2 controller, so ROS users get the same model the app runs.

Subscribes  /joint_states, /qi_vla/target (geometry_msgs/PointStamped, base_link)
Publishes   <cmd_topic> (trajectory_msgs/JointTrajectory) at 20 Hz while a
            target is set; stops (stays silent) within 1 cm of the target.

    ros2 run qi_vla_ros qi_vla_policy --ros-args -p policy:=/path/policy_ttns.json
    ros2 topic pub --once /qi_vla/target geometry_msgs/msg/PointStamped \
        "{header: {frame_id: base_link}, point: {x: 0.2, y: 0.05, z: 0.08}}"
"""

from __future__ import annotations

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from qi_vla_ros.core import ensure_repo_on_path, reach_command

ROOT = ensure_repo_on_path()

from src.robot import arm  # noqa: E402
from src.robot.policy_runtime import PolicyRuntime  # noqa: E402


class PolicyNode(Node):
    def __init__(self) -> None:
        super().__init__("qi_vla_policy")
        path = self.declare_parameter("policy", str(ROOT / "app/public/models/policy_ttns.json")).value
        cmd_topic = self.declare_parameter("cmd_topic", "/arm_controller/joint_trajectory").value
        self.policy = PolicyRuntime.load(path)
        if self.policy.spec.get("kind", "reach") != "reach":
            raise ValueError("qi_vla_policy serves reach policies (policy_dense/ttns.json)")
        self.q = None
        self.target = None
        self.pub = self.create_publisher(JointTrajectory, cmd_topic, 10)
        self.create_subscription(JointState, "/joint_states", self.on_state, 10)
        self.create_subscription(PointStamped, "/qi_vla/target", self.on_target, 10)
        self.create_timer(0.05, self.tick)
        self.get_logger().info(f"policy {self.policy.spec['method']} ({self.policy.spec['params']} params) -> {cmd_topic}")

    def on_state(self, m: JointState) -> None:
        q = arm.Q_HOME.copy()
        for i, n in enumerate(arm.JOINT_NAMES):
            if n in m.name:
                q[i] = m.position[m.name.index(n)]
        self.q = q

    def on_target(self, m: PointStamped) -> None:
        self.target = np.array([m.point.x, m.point.y, m.point.z])
        self.get_logger().info(f"target {self.target.round(3).tolist()}")

    def tick(self) -> None:
        if self.q is None or self.target is None:
            return
        if np.linalg.norm(arm.fk(self.q) - self.target) < 0.01:
            return
        q_next = reach_command(self.policy, self.q, self.target)
        msg = JointTrajectory()
        msg.joint_names = list(arm.JOINT_NAMES)
        msg.points = [JointTrajectoryPoint(positions=q_next.tolist(), time_from_start=Duration(sec=0, nanosec=100_000_000))]
        self.pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = PolicyNode()
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
