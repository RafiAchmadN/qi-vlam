#!/usr/bin/env python3
"""Open a live CoppeliaSim GUI window with an RLBench task loaded and moving.

Standalone visual demo — not part of the evaluation pipeline
(scripts/rlbench_eval.py, which evaluates the project's own untrained/healed
VLAM policy). Needs the same CoppeliaSim + PyRep + RLBench setup (see
docs/USAGE.md section 10).

Run:
    source scripts/rlbench_env.sh
    PYTHONPATH=. python scripts/rlbench_gui_demo.py --task ReachTarget

Three controllers (--controller):
  reach (default): scripted proportional controller that drives the
    end-effector straight to a target position via motion planning
    (EndEffectorPoseViaPlanning) — actually completes the task, so you see
    real progress. Reads the target from `task_low_dim_state`, which only
    a handful of RLBench tasks populate meaningfully (checked directly
    against the installed rlbench package, not assumed): ReachTarget
    (xyz of the target sphere), PickAndLift and SlideBlockToTarget (xyz of
    the block to pick/slide — reach-only, does not attempt the grasp/lift/
    slide itself). For every other task, the base `Task.get_low_dim_state()`
    returns a concatenation of ALL scene objects' poses in scene-graph
    order — its first 3 elements are NOT a target position, so this
    controller refuses to run on unlisted tasks rather than silently
    driving toward a meaningless point (see `REACH_COMPATIBLE_TASKS` below).
  sine: blind oscillating joint-velocity wiggle, no target awareness — only
    useful to confirm the simulator itself is rendering/stepping (e.g. to
    rule out a frozen viewport), not to see the task solved. Works on any
    task since it reads no task-specific state.
  policy: loads an actual trained checkpoint from scripts/rlbench_train_bc.py
    (--policy-path) and drives the arm via JointVelocity through the SAME
    build_policy() architecture and action transform rlbench_eval.py uses —
    this is what the project's own pipeline produces, not a hand-scripted
    controller. With rlbench_bc_policy.pt (60 demos, no DAgger) expect real
    but modest behavior: measured 6.7% task_success_pct and ~55-63% mean
    distance-to-target reduction over 15 held-out episodes (see
    docs/learning/03-rlbench-simulation.md §8) — it visibly moves toward
    the target most episodes, occasionally reaches it, not a guaranteed
    solve like the reach controller.

Ctrl+C to stop and close the window cleanly. Camera: click the viewport
first, then right-drag to orbit, scroll to zoom, middle-drag to pan —
independent of whether the simulation is stepping.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

# Tasks whose Task.get_low_dim_state() is overridden to return a meaningful
# xyz in its first 3 elements — verified by reading each task's source in
# the installed rlbench package (grep for "def get_low_dim_state" under
# rlbench/tasks/), not assumed. Every other RLBench task falls back to the
# base class implementation, which concatenates ALL scene objects' 7D poses
# in scene-graph order — first 3 elements there are arbitrary, not a target.
REACH_COMPATIBLE_TASKS = {
    "ReachTarget": "target sphere position (task fully solvable by reach alone)",
    "PickAndLift": "block-to-pick position (reach-only; does not grasp/lift)",
    "SlideBlockToTarget": "block position (reach-only; does not push/slide)",
}


def safe_reset(task, retries: int = 5, delay: float = 0.3):
    """task.reset() can transiently raise RuntimeError('No value available
    yet.') from PyRep's simGetJointForce right after a reset — CoppeliaSim's
    physics engine hasn't finished settling/reporting joint state yet. This
    is a race condition (empirically hit at episode 16 of a live run, not a
    RLBench task-specific failure — the resets before and after it, on the
    same task, succeeded fine), so a short retry is the correct fix, not a
    task-level workaround.
    """
    last_err = None
    for attempt in range(retries):
        try:
            return task.reset()
        except RuntimeError as e:
            last_err = e
            time.sleep(delay)
    raise RuntimeError(f"task.reset() still failing after {retries} retries") from last_err


def run_sine(task, duration: float, amplitude: float) -> None:
    from rlbench.action_modes.arm_action_modes import JointVelocity  # noqa: F401
    _, obs = safe_reset(task)
    n_joints = len(obs.joint_velocities)
    t0 = time.time()
    step = 0
    while time.time() - t0 < duration:
        t = time.time() - t0
        arm = amplitude * np.sin(2 * np.pi * 0.1 * t + np.arange(n_joints))
        action = np.concatenate([arm, [1.0]]).astype(np.float32)
        _, reward, terminate = task.step(action)
        step += 1
        if step % 40 == 0:
            print(f"step {step}  t={t:.1f}s")
        if terminate:
            safe_reset(task)
            print("episode reset")
        time.sleep(0.03)


def run_reach(task, duration: float) -> None:
    """Drive straight to the target pose each episode via motion planning."""
    t0 = time.time()
    episode = 0
    while time.time() - t0 < duration:
        _, obs = safe_reset(task)
        target_xyz = obs.task_low_dim_state
        action = np.concatenate([target_xyz, obs.gripper_pose[3:7], [1.0]]).astype(np.float32)
        try:
            _, reward, terminate = task.step(action)
        except Exception as e:  # unreachable pose / planning failure
            print(f"episode {episode}: planning failed ({e})")
            continue
        episode += 1
        print(f"episode {episode}: reward={reward:.3f} terminate={terminate}")
        time.sleep(0.5)  # pause so the reach is visible before resetting


def run_policy(task, policy, n_joints: int, max_joint_vel: float, max_steps: int,
              duration: float, task_name: str, max_episodes: int = 20,
              use_relative: bool = True) -> None:
    """Same control loop as rlbench_eval.py::run_episode, GUI-visible,
    looped across episodes for live viewing instead of returning metrics.

    max_episodes bounds the run independently of --duration: long-running
    PyRep/CoppeliaSim sessions have been observed (empirically, not just in
    theory) to degrade after dozens of episodes — RuntimeError('The call
    failed on the V-REP side') from simGetModelProperty, distinct from the
    transient single-call race safe_reset() retries. That's a session-level
    CoppeliaSim/PyRep issue, not something a retry fixes; bounding episode
    count keeps default usage well clear of where it was observed (27
    episodes, ~11 min continuous stepping) rather than chasing it further —
    this is a visualization convenience script, not the evaluation pipeline
    (rlbench_eval.py, which always runs a bounded --episodes count and has
    never hit this).
    """
    from rlbench_eval import get_augmented_obs  # sibling script, see sys.path.insert above
    t0 = time.time()
    episode = 0
    while time.time() - t0 < duration and episode < max_episodes:
        episode += 1
        _, obs = safe_reset(task)
        target_xyz = obs.task_low_dim_state[:3] if hasattr(obs, "task_low_dim_state") else None
        success = False
        for step in range(max_steps):
            x = torch.tensor(get_augmented_obs(obs, task_name, use_relative), dtype=torch.float32)
            with torch.no_grad():
                raw = policy(x.unsqueeze(0)).squeeze(0)
            arm_cmd = torch.tanh(raw[:n_joints]) * max_joint_vel
            gripper_cmd = float(torch.sigmoid(raw[n_joints]).round().item())
            action = torch.cat([arm_cmd, torch.tensor([gripper_cmd])]).numpy()
            try:
                obs, reward, terminate = task.step(action)
            except Exception as e:
                print(f"episode {episode}: step failed ({e}); resetting")
                break
            if terminate:
                success = bool(reward > 0.5)
                break
        dist_str = ""
        if target_xyz is not None:
            d = float(np.linalg.norm(np.array(obs.gripper_pose[:3]) - target_xyz))
            dist_str = f" final_dist_to_target={d:.3f}"
        print(f"episode {episode}: success={success} steps={step + 1}{dist_str}")
        time.sleep(0.4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="ReachTarget", help="class name in rlbench.tasks")
    ap.add_argument("--controller", choices=["reach", "sine", "policy"], default="reach")
    ap.add_argument("--duration", type=float, default=600.0, help="seconds to run")
    ap.add_argument("--amplitude", type=float, default=0.25,
                    help="sine controller only: rad/s joint motion size")
    ap.add_argument("--policy-path", default=None,
                    help="policy controller: checkpoint from rlbench_train_bc.py")
    ap.add_argument("--bond-dimension", type=int, default=0,
                    help="policy controller: must match the value used to train "
                         "--policy-path (0 = uncompressed)")
    ap.add_argument("--hidden-dim", type=int, default=256,
                    help="policy controller: must match --policy-path's training value")
    ap.add_argument("--depth", type=int, default=4,
                    help="policy controller: must match --policy-path's training value")
    ap.add_argument("--rel-obs", dest="rel_obs", action="store_true", default=True,
                    help="policy controller: must match --policy-path's training "
                         "value (default on; older checkpoints trained before this "
                         "feature existed need --no-rel-obs)")
    ap.add_argument("--no-rel-obs", dest="rel_obs", action="store_false")
    ap.add_argument("--max-joint-vel", type=float, default=1.0,
                    help="policy controller: rad/s action bound, should match training")
    ap.add_argument("--max-steps", type=int, default=60, help="policy controller: steps/episode")
    ap.add_argument("--episodes", type=int, default=20,
                    help="policy controller: episode cap, independent of --duration "
                         "(see run_policy docstring — long CoppeliaSim sessions can "
                         "degrade; keep this well under ~25 unless you've confirmed "
                         "your machine handles longer sessions)")
    args = ap.parse_args()

    if args.controller == "policy" and not args.policy_path:
        raise SystemExit("--controller policy requires --policy-path "
                         "(train one with scripts/rlbench_train_bc.py)")

    if args.controller == "reach" and args.task not in REACH_COMPATIBLE_TASKS:
        raise SystemExit(
            f"--controller reach does not support --task {args.task}: its "
            "task_low_dim_state is not a target position (see module "
            f"docstring). Supported tasks: {sorted(REACH_COMPATIBLE_TASKS)}. "
            "Use --controller sine instead, or extend REACH_COMPATIBLE_TASKS "
            "after checking that task's get_low_dim_state() override.")

    try:
        from rlbench.environment import Environment
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import JointVelocity, EndEffectorPoseViaPlanning
        from rlbench.action_modes.gripper_action_modes import Discrete
        from rlbench.observation_config import ObservationConfig
        import rlbench.tasks as rlbench_tasks
    except ImportError as e:
        raise SystemExit(
            "RLBench/PyRep/CoppeliaSim not available — see docs/USAGE.md "
            f"section 10 for setup.\nOriginal import error: {e}") from e

    task_cls = getattr(rlbench_tasks, args.task)
    arm_mode = EndEffectorPoseViaPlanning() if args.controller == "reach" else JointVelocity()
    action_mode = MoveArmThenGripper(arm_action_mode=arm_mode, gripper_action_mode=Discrete())
    obs_config = ObservationConfig()
    obs_config.set_all_low_dim(True)
    obs_config.set_all_high_dim(False)
    env = Environment(action_mode, obs_config=obs_config, headless=False)
    env.launch()
    task = env.get_task(task_cls)
    print(f"Loaded {args.task} ({args.controller} controller): "
          f"window should be visible now. Ctrl+C to stop.")

    try:
        if args.controller == "reach":
            run_reach(task, args.duration)
        elif args.controller == "sine":
            run_sine(task, args.duration, args.amplitude)
        else:
            from rlbench_eval import build_policy, get_augmented_obs  # sibling script
            _, obs0 = safe_reset(task)
            n_joints = len(obs0.joint_velocities)
            obs_dim = len(get_augmented_obs(obs0, args.task, args.rel_obs))
            action_dim = env.action_shape[0]
            policy = build_policy(obs_dim, action_dim, args.bond_dimension, seed=0,
                                  policy_path=args.policy_path,
                                  hidden_dim=args.hidden_dim, depth=args.depth)
            run_policy(task, policy, n_joints, args.max_joint_vel, args.max_steps,
                      args.duration, args.task, max_episodes=args.episodes,
                      use_relative=args.rel_obs)
    except KeyboardInterrupt:
        print("stopping...")
    finally:
        env.shutdown()


if __name__ == "__main__":
    main()
