#!/usr/bin/env bash
# run.sh — launch the energy benchmark
#
# Usage:
#   bash run.sh                        # full run (30 iterations, 15s rest)
#   bash run.sh --test                 # disable rest period (faster for testing)
#   bash run.sh --test --iterations 2  # 2 iterations + no rest (smoke test)
#   bash run.sh --iterations 5         # 5 iterations with normal rest

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting benchmark from: $SCRIPT_DIR"
python benchmark.py "$@"
