#!/usr/bin/env python3
"""Behavior cloning on RLBench live demonstrations — closes the "policy
belum dilatih" limitation documented in docs/learning/03-rlbench-simulation.md
§8 (rlbench_eval.py otherwise only ever evaluates a random-init policy).

Needs the same CoppeliaSim + PyRep + RLBench setup as rlbench_eval.py (see
docs/USAGE.md section 10). No external dataset: RLBench's own IK-based
demonstrator (`task.get_demos(..., live_demos=True)`) generates expert
trajectories on the fly.

Label choice: each recorded demo `Observation` already carries
`joint_velocities` — the actual telemetry of the arm's motion during expert
execution, not a signal we derive ourselves — so behavior cloning here is
plain supervised regression: (obs.get_low_dim_data() -> obs.joint_velocities),
clipped to --max-joint-vel and trained through the SAME tanh-and-scale
transform `rlbench_eval.py::run_episode` applies at deployment (so training
and evaluation see the same action space, not a mismatched one). This reuses
`VLAMTrainer` from src/training/vlam_trainer.py completely unchanged — its
`train(batch_fn, loss_fn)` interface already fits a (obs, action) regression
task, we only supply a custom loss_fn that applies the deployment transform
before computing MSE.

Run:
    source scripts/rlbench_env.sh
    PYTHONPATH=. python scripts/rlbench_train_bc.py --task ReachTarget \\
        --demos 60 --train-steps 800 --out rlbench_bc_policy.pt
    # then evaluate the trained policy for real:
    PYTHONPATH=. python scripts/rlbench_eval.py --policy-path rlbench_bc_policy.pt \\
        --max-joint-vel 1.0

Honesty notes:
  - ~50-60 demos of a single task/scene layout is enough to show the pipeline
    can learn (task_success_pct > 0, materially above the random-init
    baseline), not enough to claim general manipulation competence — report
    it as a proof point, not a solved benchmark.
  - `--bond-dimension > 0` trains the MPS-compressed architecture directly
    (same policy structure `rlbench_eval.py` evaluates) — this is the
    evidence that a compressed policy can still learn from demonstrations,
    not just that it stays bounded under a random init.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from rlbench_eval import build_policy, get_augmented_obs  # noqa: E402  (sibling script)


def collect_transitions(task, n_joints: int, max_joint_vel: float, n_demos: int,
                        seed: int, task_name: str, use_relative: bool = True
                        ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Returns (X obs, Y clipped joint-velocity targets, stats dict)."""
    # torch.manual_seed alone does NOT make demo composition reproducible:
    # RLBench's own placement code (task init_episode, SpawnBoundary) samples
    # via the GLOBAL np.random / random state, which we never seeded before.
    # That's the root cause docs/learning/03-rlbench-simulation.md §8.1a
    # diagnosed as "task.get_demos(live_demos=True) uses RLBench's own
    # internal RNG, not tied to our --seed" — this fixes it.
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    ok, failed = 0, 0
    for i in range(n_demos):
        try:
            demos = task.get_demos(1, live_demos=True)
        except Exception as e:  # RLBench/PyRep planning failure on this seed
            failed += 1
            print(f"  demo {i}: failed ({e})")
            continue
        d = demos[0]
        ok += 1
        for obs in d:
            x = torch.tensor(get_augmented_obs(obs, task_name, use_relative), dtype=torch.float32)
            y = torch.tensor(obs.joint_velocities[:n_joints], dtype=torch.float32)
            y = y.clamp(-max_joint_vel, max_joint_vel)
            xs.append(x)
            ys.append(y)
        if (i + 1) % 10 == 0:
            print(f"  collected {i + 1}/{n_demos} demos "
                  f"({len(xs)} transitions so far)")
    stats = {"demos_requested": n_demos, "demos_ok": ok, "demos_failed": failed,
             "transitions": len(xs)}
    print(f"collect_transitions: {stats}")
    if not xs:
        raise RuntimeError("No demos succeeded — cannot train. Check task name "
                           "and CoppeliaSim setup.")
    return torch.stack(xs), torch.stack(ys), stats


def make_batch_fn(X: torch.Tensor, Y: torch.Tensor, batch_size: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    n = X.shape[0]

    def batch_fn():
        idx = torch.randint(0, n, (min(batch_size, n),), generator=g)
        return X[idx], Y[idx]

    return batch_fn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="ReachTarget", help="class name in rlbench.tasks")
    ap.add_argument("--demos", type=int, default=60)
    ap.add_argument("--max-joint-vel", type=float, default=1.0,
                    help="rad/s — demo velocities on ReachTarget reach ~0.85 "
                         "rad/s (p95 ~0.80); default is higher than "
                         "rlbench_eval.py's conservative 0.3 default used for "
                         "untrained-policy safety runs")
    ap.add_argument("--bond-dimension", type=int, default=0, help="0 = no MPS compression")
    ap.add_argument("--hidden-dim", type=int, default=256,
                    help="policy tower width — pass the same value to "
                         "rlbench_eval.py --hidden-dim when evaluating")
    ap.add_argument("--depth", type=int, default=4, help="policy tower hidden-layer count")
    ap.add_argument("--train-steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--convergence-loss", type=float, default=0.005)
    ap.add_argument("--rel-obs", dest="rel_obs", action="store_true", default=True,
                    help="append relative target-gripper position to the "
                         "observation (default on)")
    ap.add_argument("--no-rel-obs", dest="rel_obs", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="rlbench_bc_policy.pt")
    ap.add_argument("--out-json", default="rlbench_bc_training.json")
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
            "RLBench/PyRep/CoppeliaSim not available — see docs/USAGE.md "
            f"section 10 for setup.\nOriginal import error: {e}") from e

    from src.training.vlam_trainer import VLAMTrainer

    task_cls = getattr(rlbench_tasks, args.task)
    obs_config = ObservationConfig()
    obs_config.set_all_low_dim(True)
    obs_config.set_all_high_dim(False)
    action_mode = MoveArmThenGripper(arm_action_mode=JointVelocity(),
                                     gripper_action_mode=Discrete())
    env = Environment(action_mode, obs_config=obs_config, headless=True)
    env.launch()

    t0 = time.time()
    try:
        task = env.get_task(task_cls)
        _, obs0 = task.reset()
        n_joints = len(obs0.joint_velocities)
        obs_dim = len(get_augmented_obs(obs0, args.task, args.rel_obs))
        action_dim = env.action_shape[0]

        print(f"Collecting {args.demos} live demos for {args.task}...")
        X, Y, demo_stats = collect_transitions(task, n_joints, args.max_joint_vel,
                                               args.demos, args.seed, args.task,
                                               args.rel_obs)
    finally:
        env.shutdown()

    collect_time = time.time() - t0

    policy = build_policy(obs_dim, action_dim, args.bond_dimension, args.seed,
                          hidden_dim=args.hidden_dim, depth=args.depth)

    def action_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Same tanh-and-scale transform rlbench_eval.py::run_episode applies
        # at deployment — train and eval must see the same action space.
        arm_pred = torch.tanh(pred[:, :n_joints]) * args.max_joint_vel
        return F.mse_loss(arm_pred, target)

    batch_fn = make_batch_fn(X, Y, args.batch_size, args.seed)
    trainer = VLAMTrainer(policy, lr=args.lr, max_steps=args.train_steps,
                          convergence_loss=args.convergence_loss, seed=args.seed)
    print(f"Training {args.train_steps} steps max "
          f"(bond_dimension={args.bond_dimension or 'none'})...")
    t1 = time.time()
    result = trainer.train(batch_fn, loss_fn=action_loss)
    train_time = time.time() - t1
    print(f"Trained: steps={result.steps_to_convergence} "
          f"final_loss={result.final_loss:.5f} wall_clock={train_time:.1f}s")

    torch.save(policy.state_dict(), args.out)
    print(f"Wrote {args.out}")

    summary = {
        "task": args.task,
        "bond_dimension": args.bond_dimension or None,
        "hidden_dim": args.hidden_dim,
        "depth": args.depth,
        "max_joint_vel": args.max_joint_vel,
        "demo_collection": demo_stats,
        "collect_wall_clock_s": collect_time,
        "train_steps_to_convergence": result.steps_to_convergence,
        "train_final_loss": result.final_loss,
        "train_wall_clock_s": train_time,
        "train_loss_history": result.loss_history,
        "policy_checkpoint": args.out,
        "note": ("Behavior cloning on RLBench's own live IK demonstrator "
                 f"({demo_stats['demos_ok']} demos, {demo_stats['transitions']} "
                 "transitions, single task/scene layout) — a proof that the "
                 "(possibly MPS-compressed) policy architecture can learn from "
                 "demonstrations, not a general manipulation benchmark result. "
                 "Evaluate with: rlbench_eval.py --policy-path "
                 f"{args.out} --max-joint-vel {args.max_joint_vel}"),
    }
    Path(args.out_json).write_text(json.dumps(summary, indent=2))
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
