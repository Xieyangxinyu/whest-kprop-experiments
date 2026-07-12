#!/usr/bin/env bash
# Measure fast-mode FLOP count + accuracy for the current estimator.py.
#
# Usage: ./flop_measure.sh [N_MLPS]     (default 25; use 100 to confirm keepers)
#
# Prints deterministic mean_flops_used (the metric to minimize), final_layer_mse
# and all_layers_mse (must stay bit-identical), and mean_residual_s (guardrail).
set -euo pipefail

N_MLPS="${1:-25}"
ESTIMATOR="${ESTIMATOR:-estimator.py}"
DATASET="${DATASET:-./whest-data}"
SPLIT="${SPLIT:-mini}"
SEED="${SEED:-42}"

WHEST_FAST=1 uv run whest run \
  --estimator "$ESTIMATOR" --dataset "$DATASET" --split "$SPLIT" \
  --runner local --n-mlps "$N_MLPS" --max-threads 4 --seed "$SEED" --json 2>/dev/null \
| python -c "
import sys, json
r = json.load(sys.stdin)['results']
fl = [m['flops_used'] for m in r['per_mlp']]
rw = [m['residual_wall_time_s'] for m in r['per_mlp']]
n = len(fl)
print(f'n_mlps          {n}')
print(f'mean_flops_used {sum(fl)/n:.6e}')
print(f'final_layer_mse {r[\"final_layer_mse\"]:.9e}')
print(f'all_layers_mse  {r[\"all_layers_mse\"]:.9e}')
print(f'mean_residual_s {sum(rw)/n:.6f}')
"
