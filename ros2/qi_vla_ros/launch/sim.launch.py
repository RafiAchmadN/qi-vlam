"""Simulated QI-VLA arm + robot_state_publisher (+ rosbridge for the app, RViz, policy).

    ros2 launch qi_vla_ros sim.launch.py backend:=mujoco env:=kitchen_counter rviz:=true
    # then in the app: Koneksi -> ROS (rosbridge) -> ws://<this PC>:9090 -> Hubungkan

Needs QI_VLA_ROOT=/path/to/qi-vlam in the environment (the nodes import the
repo's arm model and simulators).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("qi_vla_ros")
    with open(os.path.join(share, "urdf", "qi_vla_arm.urdf"), encoding="utf-8") as f:
        urdf = f.read()
    arg = lambda n, d, h: DeclareLaunchArgument(n, default_value=d, description=h)  # noqa: E731
    return LaunchDescription([
        arg("backend", "kinematic", "kinematic (servo lag only) | mujoco (physics + objects)"),
        arg("env", "lab_table", "environment preset (app/public/envs/*.json)"),
        arg("rosbridge", "true", "start rosbridge_websocket on :9090 for the app"),
        arg("rviz", "false", "start RViz"),
        arg("policy", "false", "start qi_vla_policy (send targets on /qi_vla/target)"),
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": urdf}]),
        Node(package="qi_vla_ros", executable="qi_vla_sim", output="screen",
             parameters=[{"backend": LaunchConfiguration("backend"), "env": LaunchConfiguration("env")}]),
        Node(package="qi_vla_ros", executable="qi_vla_policy", output="screen",
             condition=IfCondition(LaunchConfiguration("policy"))),
        Node(package="rosbridge_server", executable="rosbridge_websocket", output="screen",
             condition=IfCondition(LaunchConfiguration("rosbridge"))),
        Node(package="rviz2", executable="rviz2", arguments=["-d", os.path.join(share, "rviz", "qi_vla.rviz")],
             condition=IfCondition(LaunchConfiguration("rviz"))),
    ])
