# USAGE — Step-by-Step Guide

## 1. Environment setup
```bash
python3 --version            # 3.10+
python3 -m venv venv && source venv/bin/activate
```

## 2. Install dependencies
```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
# GPU: omit the extra index and install the CUDA build of torch
```

## 3. Download datasets (full-scale runs only)
- Autonomous Driving: nuScenes (https://www.nuscenes.org), Waymo Open Dataset, CARLA
- Robotics: Open X-Embodiment, RLBench, LIBERO
- Comply with each dataset license. The CPU demo pipeline needs **no** downloads.

## 4. Configure model
Default: CPU proxy tower (`make_proxy_vlam`). For LLaVA-1.5-7B:
```python
from transformers import LlavaForConditionalGeneration
from src.compression.tensor_network import compress_model

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
model, report = compress_model(model.language_model, bond_dimension=64,
                               skip_names=("lm_head", "embed_tokens"))
print(f"{report.overall_ratio:.2f}x on {len(report.layer_stats)} layers")
```
Then heal: fine-tune the compressed model briefly on your task data
(`VLAMTrainer` or your existing HF Trainer — cores are ordinary Parameters).

## 5. Run compression pipeline
```bash
python -m src.evaluation.benchmark --seeds 3 --bond-dimension 24 --out results.json
```
Steps performed per seed: (a) train baseline, (b) MPS-decompose, (c) heal,
(d) measure accuracy + ratio, (e) INT8 baseline comparison.

## 6. Train with QI enhancements
```python
from src.training.vlam_trainer import VLAMTrainer, QuantumInspiredRegularization
trainer = VLAMTrainer(model, qi_regularizer=QuantumInspiredRegularization(weight=2e-4),
                      seed=0)  # deterministic
```
Run 3+ seeds; loss histories are returned in `TrainingResult.loss_history`
(export to wandb/tensorboard as needed).

## 7. Benchmark vs baseline
```bash
bash scripts/run_benchmark.sh   # benchmark + ablation Pareto + safety suite + report
```
Outputs: `results.json`, `pareto.json`, `safety.json`.

## 8. Deploy (Docker)
```bash
docker build -t qi-vla .
docker run -p 8000:8000 qi-vla
curl localhost:8000/health          # {"status":"ok"}
curl localhost:8000/metrics         # rolling latency stats
```
Smoke test: POST 128 floats to `/infer`; response includes `latency_ms`,
`within_budget`, `safety_ok`.

## 9. Generate report
```bash
python scripts/generate_report.py   # docs/technical_report.md (+ PDF if pandoc)
```

## 10. Simulate in RLBench (real robot physics, optional)

`safety.json` certifies a *linearized analytic* closed loop. To also get
empirical safety evidence on a real nonlinear simulator, run the (possibly
MPS-compressed) VLAM policy in RLBench/CoppeliaSim as a joint-velocity
controller and check closed-loop tracking error with the same
`LyapunovStabilityValidator` used elsewhere in this repo.

Needs CoppeliaSim v4.1.0 (the exact version RLBench/PyRep are pinned to) +
PyRep + RLBench — not installed in the CPU-only dev sandbox by default.
Verified working on Ubuntu 24.04 desktop (headless mode, real X session not
required):
```bash
export COPPELIASIM_ROOT=$HOME/CoppeliaSim
mkdir -p "$COPPELIASIM_ROOT"
curl -fL -o /tmp/coppeliasim.tar.xz \
    https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
tar -xf /tmp/coppeliasim.tar.xz -C "$COPPELIASIM_ROOT" --strip-components 1

git clone --depth 1 https://github.com/stepjam/PyRep.git /tmp/PyRep
source scripts/rlbench_env.sh   # sets COPPELIASIM_ROOT/LD_LIBRARY_PATH/QT_QPA_PLATFORM_PLUGIN_PATH
pip install -r /tmp/PyRep/requirements.txt && pip install /tmp/PyRep
pip install "git+https://github.com/stepjam/RLBench.git" gymnasium  # gymnasium: undeclared rlbench dep

PYTHONPATH=. python scripts/rlbench_eval.py --task ReachTarget --episodes 20 \
    --bond-dimension 24 --out rlbench_results.json
```
If CoppeliaSim's Qt GUI fails to start (`could not load the Qt platform
plugin "xcb"`), `sudo apt-get install -y libxcb-cursor0 libxcb-xinerama0
libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libgl1
libglu1-mesa` — not needed for the headless path above, only if you drop
`headless=True`.

The policy is untrained by default (random init) — this checks that
compressed policies still produce bounded, non-diverging actuator commands
under real physics, not task success. See the script docstring for details
and how to point it at healed weights instead.
