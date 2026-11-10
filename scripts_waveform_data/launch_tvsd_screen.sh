#!/usr/bin/env bash
set -euo pipefail
cd /project

# Usage: bash scripts/launch_tvsd_screen.sh [1|6]
# 1 = train only unchanged Sinz baseline; 6 = baseline + five one-change candidates.
TRIALS="${1:-1}"
if [[ "$TRIALS" != "1" && "$TRIALS" != "6" ]]; then
  echo "Choose 1 (baseline) or 6 (complete controlled screen)" >&2
  exit 2
fi

python scripts/check_tvsd_screen.py
mkdir -p /project/tvsd_search
python -m sinz_model.cli search screen \
  --config configs/tvsd_default.yaml \
  --space configs/tvsd_screen_plan.yaml \
  --output-root /project/tvsd_search \
  --trials "$TRIALS" \
  --ensemble-members 5 \
  --fail-fast
python -m sinz_model.cli search export \
  --output-root /project/tvsd_search
