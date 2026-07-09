#!/usr/bin/env bash
# Run the full CAMELS pipeline (stages 1–7). Requires GPU + data/ (camels_root: data).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/logs results/checkpoints results/history

LOG="results/logs/full_pipeline.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== hydro_tl_ews full pipeline started $(date -Iseconds) ==="
python -c "
import torch
from hydro_tl_ews.utils.device import training_device
print('torch', torch.__version__, 'device', training_device())
"

if [[ ! -d data/basin_dataset_public_v1p2 ]]; then
  echo "ERROR: CAMELS not found under data/. See data/README.md"
  exit 1
fi

STAGES=(
  pretrain
  zero_shot
  finetune_conservative
  finetune_progressive
  local_baseline
  walk_forward
  min_data_sensitivity
)

for stage in "${STAGES[@]}"; do
  echo "=== stage=${stage} $(date -Iseconds) ==="
  python scripts/run_experiment.py --config "configs/${stage}.yaml"
done

echo "=== pipeline finished OK $(date -Iseconds) ==="
