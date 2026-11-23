#!/usr/bin/env bash
# Re-run of the 3-seed x {rel-obs, no-rel-obs} ReachTarget BC ablation
# (docs/learning/03-rlbench-simulation.md §8.1a), now that rlbench_train_bc.py
# / rlbench_eval.py seed np.random/random before any RLBench placement call.
# Previously "--seed" only fixed torch weight init / batch sampling order —
# demo composition and eval target placement ran on RLBench's own unseeded
# global RNG. That's the diagnosed root cause of why the original ablation's
# std exceeded its mean (and why v4's 16.7% didn't reproduce).
#
# Design differences from the original ablation, both deliberate:
#  - eval uses seed+100 (not the same seed as training) so eval targets never
#    overlap the training demo's target sequence.
#  - the SAME eval seed is reused across both configs at a given train seed,
#    so the two configs are compared on IDENTICAL eval targets (paired
#    comparison) instead of independently-noisy target sets.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/rlbench_env.sh
OUT=experiments/rlbench_bc_ablation/v2
mkdir -p "$OUT"

for seed in 0 1 2; do
  for cfg in relobs norelobs; do
    if [ "$cfg" = "relobs" ]; then flag="--rel-obs"; else flag="--no-rel-obs"; fi
    tag="${cfg}_s${seed}"
    echo "=== train ${tag} ==="
    PYTHONPATH=. .venv/bin/python scripts/rlbench_train_bc.py \
      --demos 200 --train-steps 800 --hidden-dim 512 --depth 6 \
      --seed "$seed" $flag \
      --out "$OUT/policy_${tag}.pt" --out-json "$OUT/train_${tag}.json"
    echo "=== eval ${tag} ==="
    PYTHONPATH=. .venv/bin/python scripts/rlbench_eval.py \
      --policy-path "$OUT/policy_${tag}.pt" --hidden-dim 512 --depth 6 \
      --episodes 60 --max-joint-vel 1.0 --seed "$((seed + 100))" $flag \
      --out "$OUT/eval_${tag}.json"
  done
done
echo "=== ablation v2 done ==="
