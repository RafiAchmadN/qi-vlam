# Changelog

## v1.1.0 — 2026-11-23

Real robot data, physics simulation, mobile robot, ROS 2 and app redesign.
All numbers below are 3 seeds; details and scope in `docs/technical_report.md`
(§6.8–6.10), `docs/SIMULATION.md` and `docs/REAL_ROBOT.md`.

### Policies trained on REAL robot data (open-loop, held-out episodes)

Motion nMSE (1.0 = predicting "don't move"; lower is better).

| Model | SO-100/101 (208 ep, 4 tasks) | Koch v1.1 (50 ep) | Params |
|---|---|---|---|
| CNN dense | 0.255 | 0.091 | 379k |
| CNN INT8 (807 KB, −46%) | 0.255 | 0.091 | 379k |
| ACT-lite Transformer | 0.256 | 0.088 | 520k |
| **CNN + TTNS χ16 + healing** | 0.268 | 0.094 | **175k (2.2×)** |
| CNN + MPS χ16 + healing | 0.374 | 0.110 | 195–210k |
| no camera | 0.351 | 0.150 | 149k |
| TTNS without healing | 0.771 | 0.255 | 175k |
| 6-unit tanh bottleneck | 0.754 | 0.267 | 381k |
| 6-qubit PQC bottleneck | 1.245 | 0.356 | 381k |
| PQC surrogate | 1.267 | 0.389 | 381k |

* Same ranking on both robots: the camera matters, TTNS and INT8 are the
  deployable compressions, MPS trails TTNS, healing is mandatory.
* PQC head: no quantum advantage on either robot (loses to the classical tanh
  bottleneck, 5–6× slower in simulation). Reported as a negative result.
* The compressed TTNS policy runs on-device in the app (JS CNN runtime, parity
  5e-6) and in a 23 ms replay control loop (`src/real/deploy.py`). The
  physical-robot path (`lerobot` SO-100/101 follower) is **not hardware-tested**.

### Simulation (MuJoCo)

* 5 arm environments (`app/public/envs`, factory held out), 3–5 objects
  (cube/cylinder, 5 colors), domain randomization, servo model (latency, lag,
  backlash, noise), friction-only grasp, LeRobot-style datasets.
* Pick-place policy success (train envs / held-out factory / hard dynamics):
  dense+DR 93.9 / 77.8 / 80.0 %, TTNS+DR 92.8 / 68.9 / 70.0 % with 23,301
  params (5.9×), no-DR dense 87.8 / 42.2 / 52.5 %, TTNS without healing ≈0 %.
* Mobile robot (TurtleBot3-sized diff-drive, 36-beam lidar, 4 maps): dense and
  TTNS (12,546 params) controllers 100 % success / 0 % collisions — the nav
  benchmark is saturated and cannot rank controllers.
* Gymnasium envs `QiVla/PickPlace-v0`, `QiVla/Nav-v0`.

### ROS 2, bridge, app

* ROS 2 Humble package `ros2/qi_vla_ros`: arm + rover sim nodes, policy node,
  URDFs, RViz, rosbridge; end-to-end smoke tests in CI.
* Bridge drivers `mujoco` (arm physics twin) and `rover`; app mirrors the twin
  scene and uploads demos (`/episodes`).
* App redesign: arm / rover / real-data modes, environment and map tabs,
  demo recorder, Research tab with charts from `results/*.json`.
* Android APK attached to this release (debug build from CI).

### Fixes and tooling

* ROS 2 nodes cast message fields to float; rover launch passes `unknown` as int.
* GitHub Actions moved to Node 24 action majors; web/APK jobs on Node 22.
* `train_real --out/--models-dir`, `scripts/merge_real_results.py`,
  `scripts/ros2_ci_local.sh` (reproduce the ROS CI job in Docker).

## v1.0.0 — 2026-11-23

First public release: MPS/TTNS compression with healing, PQC policy heads,
Lyapunov/SDP safety certificate, LIBERO offline benchmark, on-device robot app
(web + Android), bridge and firmware.
