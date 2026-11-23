# qi_vla_ros — QI-VLA arm for ROS 2

ROS 2 (Humble or newer) package for the QI-VLA 6-servo arm:

| | |
|---|---|
| `urdf/qi_vla_arm.urdf` | generated from `src/robot/arm.py` (`python -m src.sim.urdf`); tip pose == `arm.fk` (tested) |
| `qi_vla_sim` | simulated arm: `backend:=kinematic` (servo lag + latency) or `backend:=mujoco` (physics, friction grasp, objects of an environment preset). Subscribes `/arm_controller/joint_trajectory`, publishes `/joint_states` and `/qi_vla/objects` (MarkerArray), service `/qi_vla/reset` |
| `qi_vla_policy` | runs an exported reach policy (TTNS-compressed by default): `/joint_states` + `/qi_vla/target` (PointStamped) → JointTrajectory |
| `sim.launch.py` | robot_state_publisher + sim + rosbridge (:9090, for the app) [+ RViz, + policy] |
| `display.launch.py` | URDF in RViz with joint sliders |

The topic names match the app's rosbridge defaults, so the phone app's
**Koneksi → rosbridge** mode drives this simulation exactly like a real
`ros2_control` arm (`/arm_controller/joint_trajectory`, `/joint_states`).

## Build & run

```bash
cd qi-vlam/ros2
colcon build --packages-select qi_vla_ros
source install/setup.bash
export QI_VLA_ROOT=$(cd .. && pwd)          # nodes import src/robot and src/sim from the repo
pip install mujoco                          # only for backend:=mujoco
ros2 launch qi_vla_ros sim.launch.py backend:=mujoco env:=kitchen_counter rviz:=true
```

App: *Koneksi → rosbridge (ROS 1 / ROS 2)* → `ws://<PC>:9090` → *Hubungkan* → *Armed*.

Policy without the app:

```bash
ros2 launch qi_vla_ros sim.launch.py policy:=true
ros2 topic pub --once /qi_vla/target geometry_msgs/msg/PointStamped \
  "{header: {frame_id: base_link}, point: {x: 0.2, y: 0.05, z: 0.08}}"
```

## Real robot

Replace `qi_vla_sim` with your arm's driver (e.g. a `ros2_control` joint
trajectory controller for an SO-100/SO-101 or any servo arm) that accepts
`trajectory_msgs/JointTrajectory` with the joint names in the URDF and
publishes `/joint_states`. The gripper joint uses the app's normalized value
(0 closed … 1 open).

## Verification

CI (`.github/workflows/ci.yml`, job `ros2`, container `ros:humble-ros-base`)
runs `check_urdf`, `colcon build`, the ROS-free unit tests
(`test/test_core.py`) and `test/ros2_smoke.py`, which launches the MuJoCo
backend + rosbridge + policy node and checks: (1) a JointTrajectory is
tracked, (2) the app's exact rosbridge messages drive the arm over
WebSocket, (3) the TTNS policy node reaches a target. Not tested on a
physical robot.
