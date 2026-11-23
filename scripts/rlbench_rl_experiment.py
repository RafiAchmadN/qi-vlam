#!/usr/bin/env python3
"""RL alignment connected to REAL RLBench reward (not the synthetic bandit
task in src/evaluation/rl_experiment.py) — evidence on a standard open
robot-learning benchmark (RLBench, James et al. 2020).

Needs the same CoppeliaSim + PyRep + RLBench setup as scripts/rlbench_eval.py
(see docs/USAGE.md section 10) — NOT the CPU-only dev sandbox.

Design (single-step contextual interaction, not full episodic credit
assignment): GRPOAligner.align() (src/training/rl_alignment.py) samples a
group of states, has the policy act once, and computes advantage from the
resulting rewards — it was never built for multi-step temporal credit
assignment, and rewriting it into a full episodic policy-gradient method is
out of scope here. Instead each "rollout" is ONE real RLBench interaction
from a FRESH task.reset(): the policy observes the reset state, acts once,
and receives ReachTarget's real dense reward (-||gripper - target||,
env(..., shaped_rewards=True) — see rlbench/tasks/reach_target.py, verified
by reading the installed package). This is genuine physical-simulator
feedback, not a closed-form reward function of (state, action) like the
CPU-proxy experiment in rl_results.json — but it is single-step, not
full-trajectory RL. Report it as that: evidence the same PQC/surrogate/MLP
heads and the same GRPO update rule connect to real robot-simulator reward,
not a claim that the policy solves ReachTarget end-to-end.

Run:
    source scripts/rlbench_env.sh
    PYTHONPATH=. python scripts/rlbench_rl_experiment.py \\
        --seeds 2 --max-rollouts 160 --out rlbench_rl_results.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

from src.training.rl_alignment import GRPOAligner, PQCPolicyHead  # noqa: E402

try:
    from src.training.rl_alignment import PennyLanePQCHead
    _HAS_PQC = True
except Exception:  # pragma: no cover
    _HAS_PQC = False


def make_head(kind: str, state_dim: int, action_dim: int, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    if kind == "mlp":
        return nn.Sequential(nn.Linear(state_dim, 16), nn.Tanh(), nn.Linear(16, action_dim))
    if kind == "surrogate":
        return PQCPolicyHead(state_dim, action_dim, n_layers=2)
    if kind == "pqc":
        return PennyLanePQCHead(state_dim, action_dim, n_layers=2)
    raise ValueError(kind)


def make_rlbench_rollout_fn(task, n_joints: int, task_name: str,
                            max_joint_vel: float, use_relative: bool):
    """Builds rollout_fn(policy, group_size) -> (states, actions, rewards)
    for GRPOAligner.align() — group_size independent single-step
    interactions, each from a fresh task.reset(), executed against real
    CoppeliaSim physics with shaped (dense) reward."""
    from rlbench_eval import get_augmented_obs  # sibling script
    from rlbench_gui_demo import safe_reset  # sibling script, handles transient reset races

    def rollout_fn(policy: nn.Module, group_size: int
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        states, actions, rewards = [], [], []
        for _ in range(group_size):
            _, obs = safe_reset(task)
            state = torch.tensor(get_augmented_obs(obs, task_name, use_relative), dtype=torch.float32)
            with torch.no_grad():
                mean = policy(state.unsqueeze(0)).squeeze(0)
            noise = 0.1 * torch.randn_like(mean)
            raw_action = mean + noise
            arm_cmd = torch.tanh(raw_action[:n_joints]) * max_joint_vel
            gripper_cmd = float(torch.sigmoid(raw_action[n_joints]).round().item())
            action_np = torch.cat([arm_cmd, torch.tensor([gripper_cmd])]).numpy()
            try:
                _, reward, _ = task.step(action_np)
            except Exception as e:  # PyRep/IK can raise on unreachable configs
                print(f"    step failed ({e}); penalizing")
                reward = -1.0
            states.append(state)
            actions.append(raw_action.detach())
            rewards.append(float(reward))
        return torch.stack(states), torch.stack(actions), torch.tensor(rewards, dtype=torch.float32)

    return rollout_fn


def run_rlbench_rl_experiment(task_name: str = "ReachTarget", n_seeds: int = 2,
                              group_size: int = 4, max_rollouts: int = 160,
                              max_joint_vel: float = 0.3, target_reward: float = -0.05,
                              rel_obs: bool = True, headless: bool = True,
                              out_path: str = "rlbench_rl_results.json") -> Dict:
    try:
        from rlbench.environment import Environment
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import JointVelocity
        from rlbench.action_modes.gripper_action_modes import Discrete
        from rlbench.observation_config import ObservationConfig
        import rlbench.tasks as rlbench_tasks
    except ImportError as e:
        raise SystemExit(
            "RLBench/PyRep/CoppeliaSim not available in this environment — "
            "this script targets a desktop or Xvfb-headless machine with "
            "CoppeliaSim installed, not the CPU-only dev sandbox. See the "
            f"module docstring for setup steps.\nOriginal import error: {e}") from e

    from rlbench_eval import get_augmented_obs  # sibling script

    task_cls = getattr(rlbench_tasks, task_name)
    obs_config = ObservationConfig()
    obs_config.set_all_low_dim(True)
    obs_config.set_all_high_dim(False)
    action_mode = MoveArmThenGripper(arm_action_mode=JointVelocity(), gripper_action_mode=Discrete())
    env = Environment(action_mode, obs_config=obs_config, headless=headless, shaped_rewards=True)
    env.launch()
    kinds = ["mlp", "surrogate"] + (["pqc"] if _HAS_PQC else [])
    results: Dict[str, Dict] = {}
    try:
        task = env.get_task(task_cls)
        _, obs0 = task.reset()
        n_joints = len(obs0.joint_velocities)
        obs_dim = len(get_augmented_obs(obs0, task_name, rel_obs))
        action_dim = env.action_shape[0]
        print(f"task={task_name} obs_dim={obs_dim} action_dim={action_dim} n_joints={n_joints}")

        for kind in kinds:
            rollouts_used, finals = [], []
            for seed in range(n_seeds):
                head = make_head(kind, obs_dim, action_dim, seed)
                rollout_fn = make_rlbench_rollout_fn(task, n_joints, task_name,
                                                     max_joint_vel, rel_obs)
                res = GRPOAligner(head, group_size=group_size, seed=seed, lr=1e-2).align(
                    rollout_fn=rollout_fn, max_rollouts=max_rollouts, target_reward=target_reward)
                rollouts_used.append(float(res.rollouts_used))
                h = res.reward_history
                finals.append(sum(h[-10:]) / min(10, len(h)))
                print(f"{kind} seed={seed}: rollouts={res.rollouts_used} "
                      f"final_reward_ma10={finals[-1]:.4f}")
            results[kind] = {
                "rollouts_to_target": {"mean": statistics.mean(rollouts_used),
                                       "std": statistics.stdev(rollouts_used) if n_seeds > 1 else 0.0,
                                       "runs": rollouts_used},
                "final_reward_ma10": {"mean": statistics.mean(finals),
                                      "std": statistics.stdev(finals) if n_seeds > 1 else 0.0,
                                      "runs": finals},
                "params": sum(p.numel() for p in make_head(kind, obs_dim, action_dim, 0).parameters()),
            }
    finally:
        env.shutdown()

    summary = {
        "task": task_name,
        "reward_type": "shaped (dense, -distance-to-target), real CoppeliaSim physics",
        "interaction": "single-step per rollout (fresh reset each time), NOT full-episode credit assignment",
        "group_size": group_size,
        "max_rollouts_per_run": max_rollouts,
        "n_seeds": n_seeds,
        "results": results,
        "note": ("RL alignment evidence on the RLBench benchmark — complements, not replaces, "
                 "src/evaluation/rl_experiment.py's CPU-proxy bandit-task comparison "
                 "(rl_results.json), which remains the primary 3-seed statistically "
                 "rigorous comparison. This RLBench version trades seed count and "
                 "full-episode structure for genuine physical-simulator reward "
                 "feedback — read both together, not this one in isolation."),
    }
    Path(out_path).write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="ReachTarget")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--max-rollouts", type=int, default=160)
    ap.add_argument("--max-joint-vel", type=float, default=0.3)
    ap.add_argument("--target-reward", type=float, default=-0.05)
    ap.add_argument("--rel-obs", dest="rel_obs", action="store_true", default=True)
    ap.add_argument("--no-rel-obs", dest="rel_obs", action="store_false")
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--out", default="rlbench_rl_results.json")
    args = ap.parse_args()
    run_rlbench_rl_experiment(task_name=args.task, n_seeds=args.seeds, group_size=args.group_size,
                              max_rollouts=args.max_rollouts, max_joint_vel=args.max_joint_vel,
                              target_reward=args.target_reward, rel_obs=args.rel_obs,
                              headless=args.headless, out_path=args.out)


if __name__ == "__main__":
    main()
