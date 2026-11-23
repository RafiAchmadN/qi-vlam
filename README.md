# QI-VLA — Quantum-Inspired Vision-Language-Action for Real Robots

Open research + open-source app: **quantum-inspired tensor-network compression**,
**quantum (PQC) policy heads** and **Lyapunov/SDP runtime assurance** for
Vision-Language-Action (VLA) robot policies — evaluated on the open **LIBERO**
dataset and shipped as a phone app that drives a robot in **simulation** or
**live** (ESP32 Wi-Fi, USB-serial, or ROS via rosbridge).

📊 Auto-generated technical report: [`docs/technical_report.md`](docs/technical_report.md)

## Key results (all reproducible, numbers from `results/`)

| Question | Result | Evidence |
|---|---|---|
| Tensor-network compression on real VLA data (LIBERO, held-out episodes) | TTNS χ=64: **2.95× fewer params, nMSE 0.0845 vs 0.0831 dense**, faster on CPU | 3 seeds, `results/libero_benchmark.json` |
| MPS vs TTNS at equal compression | TTNS wins: 0.0845 vs 0.0985 (≈3×); 0.106 vs 0.143 (≈11×) | same |
| Is healing needed? | Yes — MPS without healing nMSE 0.87 → 0.099 with it | same |
| Quantum (PQC) head vs classical, equal width | **No advantage**: PQC 0.444, cosine surrogate 0.380, tanh 0.215 | same (negative result) |
| On-device compressed reach policy | 3.19× smaller, 100% reach (TTNS no-heal: 74%) | `results/app_policy.json` |
| Lyapunov SDP monitor + IK fallback (Simplex) under action noise | reach **8% → 78%**, table collisions **14% → 0%** | 300 paired episodes, `results/app_shield_eval.json` |
| End-to-end app (vision → language → policy → shield) | 30/30 instructions (ID/EN, pick-place, stacking) | browser e2e |

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
npm test             # 11 unit tests: kinematics/policy parity with Python, grounding, shield, vision
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
  public/models/             exported dense / TTNS policies (+ test vectors)
firmware/                    ESP32 Wi-Fi and Arduino USB-serial servo firmware
src/compression/             MPS / TTNS layers, compress_model(), INT8 baseline
src/training/                trainer, Lyapunov validator, SDP, sysid, GRPO + PQC heads
src/vla_data/                LIBERO loader, hashed instruction encoder
src/robot/                   arm model, policy training/export, NumPy policy runtime
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
