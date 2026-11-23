#!/usr/bin/env python3
"""VLAM closed-loop safety evaluation in RLBench (CoppeliaSim) — real robot
physics, not the analytic toy dynamics used by src/evaluation/safety_suite.py.

Run this on a machine with CoppeliaSim installed (Ubuntu desktop, or a cloud
VM with Xvfb for headless) — NOT in this repo's CPU-only dev sandbox, which
has neither CoppeliaSim nor PyRep. Setup:
    1. Download CoppeliaSim 4.1 (Player/EDU edition):
       https://www.coppeliarobotics.com/downloads
    2. export COPPELIASIM_ROOT=/path/to/CoppeliaSim
       export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
       export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
    3. pip install pyrep rlbench   (github.com/stepjam/PyRep, /RLBench)
    4. No display available: xvfb-run -a python scripts/rlbench_eval.py

Example (PYTHONPATH=. is required for direct script invocation to see
src/ — same reason the Dockerfile sets ENV PYTHONPATH=/app):
    source scripts/rlbench_env.sh
    PYTHONPATH=. python scripts/rlbench_eval.py --task ReachTarget \\
        --episodes 20 --bond-dimension 24 --out rlbench_results.json

Design: the arm is driven by joint-velocity control. At each step the error
state fed to the existing LyapunovStabilityValidator is the *tracking error*
between the commanded joint velocity and the velocity the simulator actually
realizes one control step later — the same "closed-loop tracking error"
framing safety_suite.py uses on its analytic system, now measured against a
real nonlinear rigid-body simulator. Because the dynamics here are a
black-box simulator (no known closed-form model), the EXACT finite-difference
linearization safety_suite.py uses (linearize_closed_loop) does not apply —
instead (--sysid, default on) a linear model is FIT from the real collected
trajectories (identify_linear_dynamics) and fed to the same solve_lyapunov_sdp.
This is a formal certificate for the identified model, bounded by its
held-out r_squared, NOT a proof about the true nonlinear simulator — always
read r_squared before trusting `certified=True`. Per CLAUDE.md's
honest-claims policy, never describe this as "formal proof of the RLBench
physics"; it proves stability of the fitted linear approximation only.

The policy is untrained (random init, or MPS-compressed random init via
--bond-dimension) unless loaded via --policy-path (see
scripts/rlbench_train_bc.py, which trains one via behavior cloning on
RLBench's own live demonstrations). With an untrained policy, expect
near-zero task success — that's expected, not a bug; what this harness
checks in that case is whether the (possibly compressed) policy produces
bounded, non-diverging actuator commands under real physics.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))


def get_augmented_obs(obs, task_name: str, use_relative: bool = True):
    """Standard low_dim_data, PLUS an explicit relative target-to-gripper
    position vector when the task exposes a meaningful target position (see
    REACH_COMPATIBLE_TASKS in rlbench_gui_demo.py).

    Why: get_low_dim_data() already includes task_low_dim_state (target
    xyz) and gripper_pose (gripper xyz) — the network has to learn to
    subtract them itself to find the direction that matters. Empirically
    this failed: a trained policy's action was measured near-IDENTICAL
    across 8 resets with wildly different target positions (target xyz
    varying by up to 0.4m) — at reset, ~37 of 40 raw dims are constant
    (fixed home pose) and only ~3 vary (the target), so a plain feedforward
    net learned to weight the dominant constant portion and produce a
    generic "toward workspace" action rather than a target-specific one.
    Feeding the relative vector directly (not just the two absolute
    positions buried among 37 near-constant dims) is standard practice in
    goal-conditioned robot learning for exactly this reason.

    obs_dim changes (+3) when task_name is reach-compatible — build_policy
    must be called with the correct obs_dim (see collect_transitions /
    run_episode / run_policy for the len(get_augmented_obs(...)) pattern),
    same requirement as --hidden-dim/--depth matching between train and eval.
    """
    import numpy as np
    from rlbench_gui_demo import REACH_COMPATIBLE_TASKS  # sibling script
    low_dim = np.asarray(obs.get_low_dim_data(), dtype=np.float32)
    if use_relative and task_name in REACH_COMPATIBLE_TASKS:
        rel = (np.asarray(obs.task_low_dim_state[:3], dtype=np.float32)
              - np.asarray(obs.gripper_pose[:3], dtype=np.float32))
        return np.concatenate([low_dim, rel])
    return low_dim


def build_policy(obs_dim: int, action_dim: int, bond_dimension: int = 0,
                 seed: int = 0, policy_path: Optional[str] = None,
                 hidden_dim: int = 256, depth: int = 4) -> nn.Module:
    from src.evaluation.benchmark import make_proxy_vlam
    torch.manual_seed(seed)
    tower = make_proxy_vlam(dim=hidden_dim, depth=depth, out_dim=action_dim, seed=seed)
    model = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.GELU(), tower)
    if bond_dimension > 0:
        from src.compression.tensor_network import compress_model
        model, report = compress_model(model, bond_dimension=bond_dimension,
                                       skip_names=())
        if report.layer_stats:
            print(f"MPS-compressed policy: {report.overall_ratio:.2f}x")
    if policy_path:
        # Weights from scripts/rlbench_train_bc.py — same build_policy() call
        # (same obs_dim/action_dim/bond_dimension) produces an architecture
        # with matching state_dict keys, so this is a plain load, not a
        # partial/strict=False patch-up. weights_only=True: we only ever load
        # our own checkpoints (a plain state_dict), no need for full unsafe
        # unpickling — also silences the FutureWarning torch.load emits by
        # default.
        state = torch.load(policy_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        print(f"Loaded policy weights from {policy_path}")
    return model


def run_episode(task, policy: nn.Module, n_joints: int, max_joint_vel: float,
                max_steps: int, task_name: str, track_target: bool = False,
                use_relative: bool = True
                ) -> Tuple[torch.Tensor, bool, Optional[dict]]:
    """Returns (tracking-error trajectory (T, n_joints), task success, target_info).

    target_info (only if track_target=True and the task exposes a target xyz
    via task_low_dim_state — see REACH_COMPATIBLE_TASKS in
    rlbench_gui_demo.py) reports gripper-to-target Euclidean distance at
    episode start and the minimum reached — a distance-to-target signal that
    is INDEPENDENT of task_success_pct (RLBench's own binary proximity-sensor
    check) and of the tracking-error trajectory this function also returns
    (that one measures commanded-vs-achieved joint velocity, a safety/control
    signal, not task progress). Useful to tell "policy moves toward the goal
    but doesn't reach the success threshold" apart from "policy doesn't move
    toward the goal at all" — both look identical if you only read
    task_success_pct.
    """
    import numpy as np
    _, obs = task.reset()
    traj = []
    success = False
    target_info = None
    if track_target:
        target_xyz = np.array(obs.task_low_dim_state[:3])
        init_dist = float(np.linalg.norm(np.array(obs.gripper_pose[:3]) - target_xyz))
        min_dist = init_dist
    for _ in range(max_steps):
        low_dim = torch.tensor(get_augmented_obs(obs, task_name, use_relative), dtype=torch.float32)
        with torch.no_grad():
            raw = policy(low_dim.unsqueeze(0)).squeeze(0)
        arm_cmd = torch.tanh(raw[:n_joints]) * max_joint_vel
        gripper_cmd = float(torch.sigmoid(raw[n_joints]).round().item())
        action = torch.cat([arm_cmd, torch.tensor([gripper_cmd])]).numpy()
        try:
            obs, reward, terminate = task.step(action)
        except Exception as e:  # PyRep/IK can raise on unreachable configs
            print(f"    step failed ({e}); ending episode early")
            break
        achieved = torch.tensor(obs.joint_velocities[:n_joints], dtype=torch.float32)
        traj.append(achieved - arm_cmd)
        if track_target:
            d = float(np.linalg.norm(np.array(obs.gripper_pose[:3]) - target_xyz))
            min_dist = min(min_dist, d)
        if terminate:
            success = bool(reward > 0.5)
            break
    if track_target:
        target_info = {"init_dist_to_target": init_dist, "min_dist_to_target": min_dist}
    if not traj:
        traj = [torch.zeros(n_joints)]
    return torch.stack(traj), success, target_info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="ReachTarget", help="class name in rlbench.tasks")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--max-joint-vel", type=float, default=0.3, help="rad/s, action bound")
    ap.add_argument("--bond-dimension", type=int, default=0, help="0 = no MPS compression")
    ap.add_argument("--policy-path", default=None,
                    help="load weights from scripts/rlbench_train_bc.py "
                         "instead of evaluating a random-init policy")
    ap.add_argument("--hidden-dim", type=int, default=256,
                    help="policy tower width — MUST match the value used to "
                         "train --policy-path")
    ap.add_argument("--depth", type=int, default=4,
                    help="policy tower hidden-layer count — MUST match "
                         "--policy-path's training value")
    ap.add_argument("--sysid", dest="sysid", action="store_true", default=True,
                    help="fit a linear model from collected trajectories and "
                         "run solve_lyapunov_sdp on it (default on)")
    ap.add_argument("--no-sysid", dest="sysid", action="store_false")
    ap.add_argument("--rel-obs", dest="rel_obs", action="store_true", default=True,
                    help="append relative target-gripper position to the "
                         "observation for reach-compatible tasks (default on) "
                         "— MUST match --policy-path's training value")
    ap.add_argument("--no-rel-obs", dest="rel_obs", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--out", default="rlbench_results.json")
    args = ap.parse_args()

    try:
        from rlbench.environment import Environment
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import JointVelocity
        from rlbench.action_modes.gripper_action_modes import Discrete
        from rlbench.observation_config import ObservationConfig
        import rlbench.tasks as rlbench_tasks
    except ImportError as e:
        raise SystemExit(
            "RLBench/PyRep/CoppeliaSim not available in this environment. This "
            "script targets a desktop or Xvfb-headless machine with CoppeliaSim "
            "installed, not the CPU-only dev sandbox — see the module docstring "
            f"for setup steps.\nOriginal import error: {e}") from e

    from src.training.vlam_trainer import LyapunovStabilityValidator
    from src.inference.model_server import LatencyProfiler
    from rlbench_gui_demo import REACH_COMPATIBLE_TASKS

    track_target = args.task in REACH_COMPATIBLE_TASKS
    task_cls = getattr(rlbench_tasks, args.task)
    obs_config = ObservationConfig()
    obs_config.set_all_low_dim(True)
    obs_config.set_all_high_dim(False)  # policy here only consumes low-dim state
    action_mode = MoveArmThenGripper(arm_action_mode=JointVelocity(),
                                     gripper_action_mode=Discrete())
    env = Environment(action_mode, obs_config=obs_config, headless=args.headless)
    env.launch()
    task = env.get_task(task_cls)

    torch.manual_seed(args.seed)
    episodes = []
    trajs: List[torch.Tensor] = []
    try:
        _, obs0 = task.reset()
        n_joints = len(obs0.joint_velocities)
        # Re-seed right before the reproducible part starts: the obs0 probe
        # reset above intentionally runs on whatever RNG state existed
        # first, but every episode reset from here on must be deterministic
        # per --seed. torch.manual_seed alone doesn't cover this — RLBench's
        # task placement (init_episode, SpawnBoundary) samples via the
        # GLOBAL np.random/random state, not anything torch touches.
        np.random.seed(args.seed)
        random.seed(args.seed)
        obs_dim = len(get_augmented_obs(obs0, args.task, args.rel_obs))
        action_dim = env.action_shape[0]
        policy = build_policy(obs_dim, action_dim, args.bond_dimension, args.seed,
                              policy_path=args.policy_path,
                              hidden_dim=args.hidden_dim, depth=args.depth)
        validator = LyapunovStabilityValidator(tolerance=1e-3)
        profiler = LatencyProfiler(budget_ms=100.0)

        for ep in range(args.episodes):
            t0 = time.perf_counter()
            traj, success, target_info = run_episode(
                task, policy, n_joints, args.max_joint_vel, args.max_steps,
                args.task, track_target=track_target, use_relative=args.rel_obs)
            profiler.record((time.perf_counter() - t0) * 1000 / max(len(traj), 1))
            trajs.append(traj)
            r = validator.validate_trajectory(traj)
            terminal_norm = traj[-1].norm().item()
            episode_safe = terminal_norm < 0.5 and r.safe_fraction > 0.9
            episodes.append({
                "success": success,
                "target_info": target_info,
                "terminal_tracking_error_norm": terminal_norm,
                "lyapunov_safe_fraction": r.safe_fraction,
                "safe": episode_safe,
                "steps": len(traj),
            })
            dist_str = ""
            if target_info:
                dist_str = (f" dist={target_info['init_dist_to_target']:.3f}->"
                           f"{target_info['min_dist_to_target']:.3f}")
            print(f"episode {ep}: success={success} safe={episode_safe} "
                  f"terminal_err={terminal_norm:.3f}{dist_str}")
    finally:
        env.shutdown()

    # --- system-identified linear model + formal SDP certificate ---
    # No closed-form dynamics function exists for CoppeliaSim (it's a
    # black-box physics simulator), so linearize_closed_loop (exact
    # finite-difference Jacobian) does not apply here. identify_linear_dynamics
    # instead FITS an approximate linear model from the real trajectories
    # collected above, then feeds it to the same solve_lyapunov_sdp used
    # elsewhere. The certificate this produces is formal *for the identified
    # model*, not a proof about the true nonlinear simulator — validity is
    # bounded by sysid_r2 (fit quality on held-out transitions).
    sysid_certificate = None
    sysid_r2 = None
    if args.sysid and len(trajs) >= 2:
        from src.training.vlam_trainer import identify_linear_dynamics, solve_lyapunov_sdp
        try:
            A_hat, sysid_r2 = identify_linear_dynamics(trajs)
            cert = solve_lyapunov_sdp(A_hat)
            sysid_certificate = {
                "certified": cert.certified,
                "spectral_radius": float(torch.linalg.eigvals(A_hat).abs().max()),
                "min_eig_P": cert.min_eig_P,
                "max_eig_ATPA_minus_P": cert.max_eig_decrease,
                "r_squared": sysid_r2,
                "meaning": ("SDP-feasible P proves local asymptotic stability of the "
                           "SYSTEM-IDENTIFIED linear approximation of the tracking-"
                           "error dynamics (fit on real RLBench trajectories above), "
                           "not a proof about the true nonlinear CoppeliaSim physics. "
                           f"Validity is bounded by r_squared={sysid_r2:.3f} on held-out "
                           "transitions — treat as unreliable below ~0.5."),
            }
            print(f"System-ID: r_squared={sysid_r2:.3f} certified={cert.certified}")
        except Exception as e:  # numerically ill-conditioned data, etc.
            print(f"System-ID failed ({e}); skipping formal certificate")

    n = len(episodes)
    dist_reduction_pct = None
    if track_target:
        inits = [e["target_info"]["init_dist_to_target"] for e in episodes]
        mins = [e["target_info"]["min_dist_to_target"] for e in episodes]
        mean_init, mean_min = sum(inits) / n, sum(mins) / n
        dist_reduction_pct = 100.0 * (1.0 - mean_min / mean_init) if mean_init > 0 else None

    summary = {
        "task": args.task,
        "episodes": n,
        "bond_dimension": args.bond_dimension or None,
        "hidden_dim": args.hidden_dim,
        "depth": args.depth,
        "policy_path": args.policy_path,
        "task_success_pct": 100.0 * sum(e["success"] for e in episodes) / n,
        "safe_pct": 100.0 * sum(e["safe"] for e in episodes) / n,
        "mean_terminal_tracking_error_norm":
            sum(e["terminal_tracking_error_norm"] for e in episodes) / n,
        "mean_dist_to_target_reduction_pct": dist_reduction_pct,
        "latency": profiler.summary(),
        "formal_sdp_certificate": sysid_certificate,
        "note": ("Empirical closed-loop tracking-error monitoring on a real "
                 "nonlinear rigid-body simulator (CoppeliaSim via RLBench), "
                 "plus (if --sysid) a formal SDP certificate for a linear model "
                 "IDENTIFIED FROM the collected trajectories — see "
                 "formal_sdp_certificate.meaning and r_squared before trusting "
                 "it. This complements, not replaces, the exact-linearization "
                 "SDP certificate in safety.json (that one linearizes a known "
                 "closed-form toy system; this one fits an approximation to a "
                 "black-box simulator). Policy is untrained unless loaded via "
                 "--policy-path (see scripts/rlbench_train_bc.py). "
                 "mean_dist_to_target_reduction_pct (only for tasks in "
                 "REACH_COMPATIBLE_TASKS, rlbench_gui_demo.py) is a distance-"
                 "to-target signal INDEPENDENT of task_success_pct (RLBench's "
                 "binary proximity-sensor check, a tight threshold) and of "
                 "mean_terminal_tracking_error_norm (a velocity-tracking safety "
                 "signal, not task progress) — a policy can show large "
                 "distance reduction while task_success_pct stays 0%, which "
                 "means it learned real directional signal without precision "
                 "enough to trigger success; read both numbers, not just one."),
    }
    Path(args.out).write_text(json.dumps({"summary": summary, "episodes": episodes}, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
