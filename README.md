# QI-VLA — Quantum-Inspired Vision-Language-Action for Real Robots

Open research + open-source app: **quantum-inspired tensor-network compression**,
**quantum (PQC) policy heads** and **Lyapunov/SDP runtime assurance** for
Vision-Language-Action (VLA) robot policies — evaluated on the open **LIBERO**
dataset and shipped as a phone app that drives a robot in **simulation** or
**live** (ESP32 Wi-Fi, USB-serial, ROS 2 via rosbridge, or a MuJoCo physics
twin). Two robot types: a **6-DOF arm** (five place-based environments — lab,
kitchen, warehouse, classroom, factory — with 3–5 cubes/cylinders per scene)
and a **mobile robot** (TurtleBot3-sized diff-drive rover with 360° lidar on
four maps — home, warehouse, office, hospital — with obstacles that are not on
the map). Both are shared by the app, MuJoCo physics with realistic actuator
imperfections, Gymnasium envs, a ROS 2 package and dataset generators that
train new policies — see [`docs/SIMULATION.md`](docs/SIMULATION.md).

📊 Auto-generated technical report: [`docs/technical_report.md`](docs/technical_report.md) ·
🧪 Simulation, environments & ROS: [`docs/SIMULATION.md`](docs/SIMULATION.md) ·
🦾 Real robot data → model → physical arm: [`docs/REAL_ROBOT.md`](docs/REAL_ROBOT.md)

## Key results (all reproducible, numbers from `results/`)

| Question | Result | Evidence |
|---|---|---|
| Policies trained on **REAL robot data** (SO-100/SO-101: 208 teleop episodes, 4 tasks, camera + joints; Koch v1.1) | camera policy motion error **0.255** vs 0.351 without camera; **INT8 807 KB (−46%)** and **TTNS 175k params (2.2×)** keep quality (0.255 / 0.268); MPS 0.374; TTNS without healing 0.771; quantum PQC head 1.245 (no advantage, 5× slower). Same ranking on a second robot (Koch v1.1, 50 episodes): camera 0.091 vs 0.150, TTNS 0.094, no-heal 0.255. Runs on-device in the app and in a 23 ms replay control loop | 3 seeds, held-out real episodes, `results/real_benchmark*.json` |
| Tensor-network compression on real VLA data (LIBERO, held-out episodes) | TTNS χ=64: **2.95× fewer params, nMSE 0.0845 vs 0.0831 dense**, faster on CPU | 3 seeds, `results/libero_benchmark.json` |
| MPS vs TTNS at equal compression | TTNS wins: 0.0845 vs 0.0985 (≈3×); 0.106 vs 0.143 (≈11×) | same |
| Is healing needed? | Yes — MPS without healing nMSE 0.87 → 0.099 with it | same |
| Quantum (PQC) head vs classical, equal width | **No advantage**: PQC 0.444, cosine surrogate 0.380, tanh 0.215 | same (negative result) |
| On-device compressed reach policy | 3.19× smaller, 100% reach (TTNS no-heal: 74%) | `results/app_policy.json` |
| Lyapunov SDP monitor + IK fallback (Simplex) under action noise | reach **8% → 78%**, table collisions **14% → 0%** | 300 paired episodes, `results/app_shield_eval.json` |
| End-to-end app (vision → language → policy → shield) | 30/30 instructions (ID/EN, pick-place, stacking) | browser e2e |
| New pick-place policy trained only on simulated data (MuJoCo, friction grasp, DR) | PICKPLACE_ROW | 3 seeds, `results/sim_pickplace.json` |
| App ↔ MuJoCo physics twin (live mode through the bridge) | pick-place 1.5 mm, stacking 6 mm placement error; app demos → dataset → training | browser e2e |
| Mobile robot: language goal → A* + lidar costmap → learned local controller (dense/TTNS) + lidar barrier shield | dense 36,610 vs TTNS 12,546 params (2.9×): **100% success, 0% collisions** on training maps, held-out hospital, out-of-range dynamics and 3 unknown obstacles (3 seeds) — this navigation benchmark is **saturated**: it shows compression is free here, not which controller is better | 3 seeds, `results/sim_nav.json` |
| App ↔ MuJoCo rover through the bridge (live) | reached the loading dock in 39 s around 2 unknown obstacles, shield active 1% of steps | browser e2e |

Honest scope: LIBERO numbers are offline action prediction **without images**,
not closed-loop simulator success. Literature success rates (OpenVLA 76.5%,
π0 86–94%, SmolVLA 87.3%, OpenVLA-OFT 97.1%, …) are reported in the
literature and were not re-run here. The live mode was verified against the bridge's mock
robot, not yet on physical hardware.

## The app — `app/`

| | |
|---|---|
| **Simulasi** | 3D 6-servo arm (SO-100/SO-101-like geometry), coloured cubes & pads, overhead camera with HSV perception, bilingual instructions (`ambil kubus merah taruh di pad kuning`, `pick up the blue cube and place it on the purple pad`, `tumpuk kubus hijau di atas kubus biru`) |
| **Robot Nyata** | streams joint targets at 20 Hz to: the Python bridge (`ws://PC:8765/ws`), an ESP32 with `firmware/esp32_wifi_arm` (no PC), or any ROS 1/2 arm via rosbridge (`ws://robot:9090`). Measured state shown as a cyan "ghost" arm |
| **Policies** | TTNS-compressed (default), dense, classical IK only, or remote server `/act` |
| **Safety** | joint/velocity/workspace limits, Lyapunov V(e)=eᵀPe monitor with IK fallback, latched E-STOP, pose sync before arming; the bridge/firmware repeat limits + watchdog + e-stop independently |

```bash
cd app
npm ci
npm run dev          # http://localhost:5173 (works in any phone browser on the same Wi-Fi)
npm test             # unit tests: kinematics/policy parity with Python, grounding, shield, vision, envs, recorder
node test/shield_eval.mjs   # regenerates results/app_shield_eval.json
```

### Android APK (open source)

The APK is built by GitHub Actions (`.github/workflows/android.yml`, Capacitor
6 + Gradle, JDK 17): download it from the workflow run's **artifacts**
(`qi-vla-robot-apk`), or from a **Release** when a `v*` tag is pushed. Local
build (needs Android SDK + JDK 17):

```bash
cd app && npm ci && npm run android:init && cd android && ./gradlew assembleDebug
```

The app is a debug-signed APK for research use. It talks plain `ws://`/`http://`
on your LAN — run the bridge/robot only on a trusted network.

### Connecting a real robot

```bash
# PC bridge (mock robot — no hardware)
python -m src.bridge.server --driver mock
# MuJoCo physics twin of an environment (objects mirrored into the app)
python -m src.bridge.server --driver mujoco --env kitchen_counter --randomize
# MuJoCo mobile robot (app: header -> Mobile robot)
python -m src.bridge.server --driver rover --map warehouse --unknown 2
# ROS 2 (simulated arm / rover + rosbridge for the app; see ros2/qi_vla_ros/README.md)
ros2 launch qi_vla_ros sim.launch.py backend:=mujoco env:=warehouse_shelf
ros2 launch qi_vla_ros rover.launch.py backend:=mujoco map:=office      # /cmd_vel /odom /scan like a TurtleBot3
# USB-serial servo arm (Arduino/ESP32 with firmware/arduino_serial_arm)
python -m src.bridge.server --driver serial --port COM3 --cal calib.json
```

The bridge also serves the built web app at `http://PC:8765/` and a policy
endpoint `POST /act`. Calibration JSON: `{"joints":[{"center_deg":90,"sign":1,"scale":1}, ×5], "gripper_closed_deg":30, "gripper_open_deg":90}`.
**First power-up:** remove the horn/linkage or hold the arm, test each joint
from the Manual tab at low speed, keep a hand on the E-STOP. The firmware has
not been compiled in CI — verify on your board.

## Research code

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
bash scripts/run_tests.sh                          # Python tests + coverage gate

python scripts/download_libero.py                  # 21 MB parquet (videos optional: --videos)
python -m src.evaluation.libero_benchmark --seeds 3   # dense/INT8/MPS/TTNS/PQC on LIBERO (~20 min CPU)
python -m src.robot.train_policy                   # train + compress + certify + export app policy
python -m src.sim.collect --out data/qivla_sim --episodes 1600   # MuJoCo dataset (LeRobot-style)
python -m src.sim.train_pickplace --seeds 3        # new pick-place policy: dense/TTNS, DR ablation, held-out tests
python -m src.sim.mobile.train_nav collect && python -m src.sim.mobile.train_nav train --seeds 3   # mobile robot
python -m src.real.download && python -m src.real.train_real --seeds 3   # REAL SO-100/101 data: 11 models, compression
python -m src.real.export_web --model models/real/cnn_ttns.pt          # -> app "Data robot nyata" mode
python -m src.real.deploy robot --robot so101 --port COM5 --dry-run    # physical arm via lerobot (see docs/REAL_ROBOT.md)
python scripts/generate_report.py                  # rebuild docs/technical_report.md
```

Earlier experiments (still in the repo, documented in the technical report):
proxy-tower compression & training-FLOPs study, Pareto sweep, PQC vs surrogate
GRPO, PQC noise sensitivity, heuristic-clipping vs SDP safety comparison,
measured wall-clock, partial LLaVA-1.5-7B compression (1 layer, Colab T4),
RLBench/CoppeliaSim behaviour cloning and RL (`scripts/rlbench_*.py`).

## Repository map

```
app/                         Web + Android app (Vite, Three.js, Capacitor)
  src/robot/                 kinematics (port of src/robot/arm.py), 3D scene
  src/vla/                   vision (HSV), language grounding, policy runtime, skill agent
  src/safety/shield.js       limits + Lyapunov monitor + IK fallback (Simplex)
  src/connect/transports.js  qi-vla-bridge/1 WebSocket + rosbridge
  src/vla/pickplace.js       end-to-end pick-place policy trained in MuJoCo
  src/data/recorder.js       demonstration recorder (qi-vla-demos/1)
  src/rover/                 mobile robot: planner, costmap, policy, shield, 3D map view
  src/real/                  real-data vision policy runtime (JS CNN) + replay of real episodes
  src/ui/research.js         "Riset" tab: charts of all results
  public/models/             exported dense / TTNS policies (+ test vectors)
firmware/                    ESP32 Wi-Fi and Arduino USB-serial servo firmware
src/compression/             MPS / TTNS layers, compress_model(), INT8 baseline
src/training/                trainer, Lyapunov validator, SDP, sysid, GRPO + PQC heads
src/vla_data/                LIBERO loader, hashed instruction encoder
src/robot/                   arm model, policy training/export, NumPy policy runtime
src/sim/                     environments, MuJoCo physics, expert, dataset generator, Gymnasium env,
                             pick-place training, MJCF/URDF export
src/real/                    REAL robot data: download, AV1 decode/alignment, vision policies, training,
                             NumPy runtime, web export, physical deployment (lerobot)
src/sim/mobile/              rover: maps, ray casting, A*, costmap, MuJoCo diff-drive, nav expert,
                             lidar shield, Gymnasium env, dataset + nav policy training
ros2/qi_vla_ros/             ROS 2 package: arm + rover URDF, sim nodes (kinematic/MuJoCo), policy node, launch
sim/mjcf/                    MJCF scenes of the five environments (MuJoCo viewer)
app/public/envs/             environment presets + domain-randomization ranges (shared app/Python)
app/public/maps/             mobile-robot maps, robot spec, DR ranges (shared app/Python)
src/bridge/                  robot bridge server + drivers (mock, serial)
src/evaluation/              benchmarks (LIBERO, proxy, ablation, safety, RL, noise)
src/inference/               FastAPI inference server
scripts/                     dataset download, report, LLaVA (Colab), RLBench
results/                     JSON outputs of the new experiments
```

## Citation

```bibtex
@misc{qivla2026,
  title  = {QI-VLA: Quantum-Inspired Compression, Quantum Policy Heads and Lyapunov
            Runtime Assurance for Vision-Language-Action Robot Policies},
  author = {Rafi Achmad N.},
  year   = {2026},
  url    = {https://github.com/RafiAchmadN/qi-vlam}
}
```

Dataset credit: LIBERO (Liu et al., NeurIPS 2023 D&B), LeRobot conversion `lerobot/libero`.
License: MIT (code, app, firmware).
