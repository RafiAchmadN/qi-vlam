#!/usr/bin/env bash
# Source this before running scripts/rlbench_eval.py:
#   source scripts/rlbench_env.sh && python scripts/rlbench_eval.py
# Points at the local CoppeliaSim v4.1.0 install (RLBench/PyRep are pinned to
# this exact version) — see docs/USAGE.md section 10.
export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-$HOME/CoppeliaSim}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$COPPELIASIM_ROOT"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
