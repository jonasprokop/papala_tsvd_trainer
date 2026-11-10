#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

ROOT=${ROOT:-/project/data/processed/tvsd_waveform_aware_v1}
IMAGES=${IMAGES:-/project/data/assets/THINGS}
MANIFEST=${MANIFEST:-/project/data/reports/tvsd_manifest.csv}
: "${WAVEFORM_PARAMS:?Set WAVEFORM_PARAMS to a YAML containing waveform class thresholds}"
test -f "$WAVEFORM_PARAMS" || { echo "Missing waveform parameters: $WAVEFORM_PARAMS"; exit 1; }

test -f "$MANIFEST" || { echo "Missing manifest: $MANIFEST"; exit 1; }
test -d "$IMAGES" || { echo "Missing images: $IMAGES"; exit 1; }

echo "=== 1/9 EXTRACT KS4 ==="
python -u scripts/extract_ks4.py --all --area V1 --manifest "$MANIFEST" --output-root "$ROOT"

echo "=== 2/9 AUDIT IMAGES ==="
python -u scripts/audit_images.py --processed-root "$ROOT" --images-root "$IMAGES"

echo "=== 3/9 WAVEFORM CLASSIFICATION (ANNOTATION ONLY) ==="
python -u scripts/waveform_classes.py classify --root "$ROOT" --params "$WAVEFORM_PARAMS"

echo "=== 4/9 TRACK NEURONS ==="
python -u scripts/track_units.py --root "$ROOT"

echo "=== 5/9 SELECT TRACKS (MERGE; NO 3-RECORDING GATE) ==="
python -u scripts/select_tracks.py --root "$ROOT" --min-recordings 1

echo "=== 6/9 SUMMARIZE WAVEFORM CLASSES (NO LABEL VETO) ==="
python -u scripts/waveform_classes.py summarize --root "$ROOT"

echo "=== 7/9 BUILD DATASET ==="
python -u scripts/dataset.py prepare --processed-root "$ROOT" --images-root "$IMAGES" --output-dir "$ROOT/model_dataset"

echo "=== 8/9 MATLAB GRAYSCALE NORMALIZATION ==="
python -u scripts/calibrate_normalization.py --dataset-root "$ROOT/model_dataset" --image-size 224

echo "=== 9/9 VALIDATE ==="
for animal in F N; do
    for partition in train val test; do
        python -u scripts/dataset.py check --animal "$animal" --partition "$partition" --prepared-root "$ROOT/model_dataset"
    done
done

echo "=== ALL DONE ==="
