#!/bin/bash
# Frozen whole-dataset exact spatial fit. Does NOT alter any existing benchmark.
set -euo pipefail

ROOT=$(realpath /storage/brno2/home/jonas_prokop/visual_models)
CODE="$ROOT/code/papala_tsvd_trainer"
SIF="$ROOT/containers/tvsd-bookworm.sif"
PY=/mnt/tvsd/envs/zwm-py310/bin/python
OUT=/mnt/tvsd/results/zwm_v1_paper_mua_spatial_exact_h100

for file in "$ROOT/code/bbscore_public/metrics/ridge.py" \
            "$ROOT/code/bbscore_public/metrics/utils.py" \
            "$ROOT/code/bbscore_public/data/utils.py" \
            "$ROOT/data/papale_mua_release/TVSD/monkeyF/THINGS_normMUA.mat" \
            "$ROOT/data/papale_mua_release/TVSD/monkeyN/THINGS_normMUA.mat"; do
  [[ -f "$file" ]] || { echo "Missing $file" >&2; exit 2; }
done
export SINGULARITYENV_PYTHONPATH=/project:/mnt/tvsd/code/ZWM
export SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
export SINGULARITYENV_OPENBLAS_NUM_THREADS=4
export SINGULARITYENV_OMP_NUM_THREADS=4

singularity exec \
  --bind "$CODE:/project" --bind "$ROOT:/mnt/tvsd" \
  --pwd /mnt/tvsd/code/ZWM "$SIF" "$PY" \
  -m transformer_model.benchmark_spatial_exact \
  --phase init --config /project/configs/tvsd_zwm_paper_mua_v1.yaml \
  --output "$OUT" --bbscore-root /mnt/tvsd/code/bbscore_public \
  --mua-root /mnt/tvsd/data/papale_mua_release/TVSD --layer-index 3

array_id=$(qsub -J 0-3%4 "$CODE/jobs/zwm_mua_spatial_exact_extract.pbs")
echo "Extraction array: $array_id"
if ! gram_id=$(qsub -W "depend=afterok:$array_id" -v ZWM_EXACT_PHASE=gram \
                    "$CODE/jobs/zwm_mua_spatial_exact_h100.pbs"); then
  echo 'PBS rejected Gram dependency; no H100 fit scheduled.' >&2
  exit 1
fi
echo "H100 Gram job:    $gram_id"
if ! fit_id=$(qsub -W "depend=afterok:$gram_id" -v ZWM_EXACT_PHASE=fit \
                   "$CODE/jobs/zwm_mua_spatial_exact_h100.pbs"); then
  echo "PBS rejected fitting dependency; submit after Gram completion:" >&2
  echo "qsub -v ZWM_EXACT_PHASE=fit $CODE/jobs/zwm_mua_spatial_exact_h100.pbs" >&2
  exit 1
fi
echo "H100 Ridge job:   $fit_id"
