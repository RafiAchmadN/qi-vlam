#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.evaluation.benchmark --seeds "${SEEDS:-3}" --out results.json
python -m src.evaluation.ablation
python -m src.evaluation.safety_suite
python scripts/generate_report.py
echo "Artifacts: results.json pareto.json safety.json docs/technical_report.md"
