"""View the URDF in RViz with joint sliders (no simulation)."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("qi_vla_ros")
    with open(os.path.join(share, "urdf", "qi_vla_arm.urdf"), encoding="utf-8") as f:
        urdf = f.read()
    return LaunchDescription([
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": urdf}]),
        Node(package="joint_state_publisher_gui", executable="joint_state_publisher_gui"),
        Node(package="rviz2", executable="rviz2", arguments=["-d", os.path.join(share, "rviz", "qi_vla.rviz")]),
    ])
