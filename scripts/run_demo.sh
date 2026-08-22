#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
output_dir="${1:-outputs/demo}"

PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}" python3 -m precure.cli \
  --backend mock \
  --method fixed_cps \
  --iterations 5 \
  --samples-per-persona 3 \
  --personas-per-campaign 3 \
  --output-dir "$output_dir"

printf 'Demo complete. See %s/summary.csv\n' "$output_dir"
