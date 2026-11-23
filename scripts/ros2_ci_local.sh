#!/usr/bin/env bash
# Reproduce the CI `ros2` job locally in the same container:
#   docker run --rm -v "$PWD":/ws -w /ws ros:humble-ros-base bash scripts/ros2_ci_local.sh [rover|arm|all]
set -e
WHAT=${1:-all}
apt-get update -qq
apt-get install -y -qq --no-install-recommends python3-pip python3-colcon-common-extensions \
  ros-humble-rosbridge-server ros-humble-robot-state-publisher ros-humble-tf2-ros ros-humble-nav-msgs \
  liburdfdom-tools libgl1 libegl1 > /dev/null
pip3 install -q mujoco websocket-client pytest
source /opt/ros/humble/setup.bash
rm -rf /tmp/ros2 && cp -r ros2 /tmp/ros2
(cd /tmp/ros2 && colcon build --packages-select qi_vla_ros > /dev/null)
source /tmp/ros2/install/setup.bash
export QI_VLA_ROOT=/ws
if [ "$WHAT" = "arm" ] || [ "$WHAT" = "all" ]; then
  ros2 launch qi_vla_ros sim.launch.py backend:=mujoco policy:=true > /tmp/launch.log 2>&1 &
  LPID=$!; sleep 15
  python3 ros2/qi_vla_ros/test/ros2_smoke.py || { cat /tmp/launch.log; kill $LPID; exit 1; }
  kill $LPID || true; sleep 3
fi
if [ "$WHAT" = "rover" ] || [ "$WHAT" = "all" ]; then
  ros2 launch qi_vla_ros rover.launch.py backend:=mujoco map:=warehouse unknown:=1 > /tmp/rover.log 2>&1 &
  LPID=$!; sleep 15
  python3 ros2/qi_vla_ros/test/ros2_rover_smoke.py || { echo "----- launch log"; cat /tmp/rover.log; kill $LPID; exit 1; }
  kill $LPID || true
fi
echo "ROS2 LOCAL CI OK"
