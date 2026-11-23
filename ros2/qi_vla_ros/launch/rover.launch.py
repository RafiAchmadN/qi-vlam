"""Simulated QI-VLA rover (diff-drive) + robot_state_publisher (+ rosbridge for the app, RViz).

    ros2 launch qi_vla_ros rover.launch.py map:=warehouse backend:=mujoco unknown:=2 rviz:=true
    # app: Mobile robot -> Koneksi -> rosbridge -> ws://<this PC>:9090 -> Hubungkan -> Armed

Same topics as a TurtleBot3 (/cmd_vel, /odom, /scan), so the app can drive a
real TurtleBot3 / Nav2 robot the same way. Needs QI_VLA_ROOT=/path/to/qi-vlam.
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
    with open(os.path.join(share, "urdf", "qi_vla_rover.urdf"), encoding="utf-8") as f:
        urdf = f.read()
    arg = lambda n, d, h: DeclareLaunchArgument(n, default_value=d, description=h)  # noqa: E731
    return LaunchDescription([
        arg("backend", "kinematic", "kinematic (unicycle + lag) | mujoco (physics, slip)"),
        arg("map", "home", "map (app/public/maps/*.json)"),
        arg("unknown", "0", "obstacles not in the map"),
        arg("rosbridge", "true", "start rosbridge_websocket on :9090 for the app"),
        arg("rviz", "false", "start RViz"),
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": urdf}]),
        Node(package="qi_vla_ros", executable="qi_vla_rover_sim", output="screen",
             parameters=[{"backend": LaunchConfiguration("backend"), "map": LaunchConfiguration("map"),
                          "unknown": LaunchConfiguration("unknown")}]),
        Node(package="rosbridge_server", executable="rosbridge_websocket", output="screen",
             condition=IfCondition(LaunchConfiguration("rosbridge"))),
        Node(package="rviz2", executable="rviz2", arguments=["-d", os.path.join(share, "rviz", "qi_vla.rviz")],
             condition=IfCondition(LaunchConfiguration("rviz"))),
    ])
