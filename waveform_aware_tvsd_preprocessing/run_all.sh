#!/bin/bash
set -euo pipefail
cd 

ROOT=
IMAGES=
MANIFEST=

test -f "$MANIFEST" || { echo "Missing manifest: $MANIFEST"; exit 1; }
test -d "$IMAGES" || { echo "Missing images: $IMAGES"; exit 1; }

echo "=== 1/7 EXTRACT KS4 ==="
python -u scripts/extract_ks4.py --all --area V1 --manifest "$MANIFEST" --output-root "$ROOT"

echo "=== 2/7 AUDIT IMAGES ==="
python -u scripts/audit_images.py --processed-root "$ROOT" --images-root "$IMAGES"

echo "=== 3/7 TRACK NEURONS ==="
python -u scripts/track_units.py --root "$ROOT"

echo "=== 4/7 SELECT TRACKS (MERGE) ==="
python -u scripts/select_tracks.py --root "$ROOT"

echo "=== 5/7 BUILD DATASET ==="
python -u scripts/dataset.py prepare --processed-root "$ROOT" --images-root "$IMAGES" --output-dir "$ROOT/model_dataset"

echo "=== 6/7 MATLAB GRAYSCALE NORMALIZATION ==="
python -u scripts/calibrate_normalization.py --dataset-root "$ROOT/model_dataset" --image-size 224

echo "=== 7/7 VALIDATE ==="
for animal in F N; do
    for partition in train val test; do
        python -u scripts/dataset.py check --animal "$animal" --partition "$partition" --prepared-root "$ROOT/model_dataset"
    done
done

echo "=== EXTRA: FIRST-ONLY SELECTION ==="
python -u scripts/select_tracks.py --root "$ROOT" --reappearance first-only --min-train 0 --output "$ROOT/identity_diagnostics/joint/selection/first_only"

echo "=== ALL DONE ==="
