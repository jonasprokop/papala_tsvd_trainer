# TVSD / THINGS preprocessing handoff — Jorge

**Preprocessing and loading only.** This standalone package is distilled from the
existing KS4 → THINGS pipeline. It contains **no cell-type classification data,
annotation joins, label-aware tracking, Gaussian readout mapping, model,
trainer, hyperparameter search, or classical-physiology experiments**.

It needs the *original* filtered KS4 NIX files, TVSD trial metadata, and original
THINGS images on the receiving machine. **No recordings or image assets are
included in this ZIP.** Source recordings are read, never edited.

## What it does

1. Read filtered KS4 NIX units with recording-specific UUIDs and waveform/FR/spike metadata;
   count spikes in the half-open **[40,160) ms** interval by default after each successful
   stimulus onset. Keep a boolean observation mask for truncated recordings.
2. Resolve the original THINGS stimulus `(train_or_test, image_index)` to an image file
   and validate that the stimulus IDs and paths are unambiguous.
3. **Propose** matches of neurons recorded in multiple sessions,
   using only waveform shape, firing rate, ISI distribution, and amplitude;
   one-to-one matching, within-session exclusion, day-first matching.
   These IDs are *provisional*, not verified neuronal identities.
4. Select **provisional longitudinal neuron tracks** with the existing conservative
   quality cutoffs (modifiable), then merge recordings into one response column
   per selected neuron. Keep the complete original unit inventory and UUIDs as
   provenance, along with an explicit `unit_to_neuron.csv` mapping. No
   architecture-specific readout indexing is used. The user can instead retain
   only the first occurrence or exclude recurring tracks.
5. Freeze an image-identity-level **90/10 training/validation split** using seed 8000,
   common across monkeys. Preserve the original repeated test presentations.
6. Decode the **entire original image** into 224×224 pixels using bicubic resizing,
   without cropping or augmentation. Compute a *single global mean and std per
   output color channel* over **distinct training images only**, then apply the
   same z-score transform to train, validation and test. Grayscale is repeated
   to three channels for compatibility with three-channel image encoders.

This follows the **global pixel mean/std normalization principle** in the
Cadena/Sinz modeling code, but the current train-only fit and 224×224 full-field
geometry are intentional TVSD adaptations—not an assertion of identical visual
angle, retinal scale or experimental image pipeline.

## Requirements

Python 3.9+ and `requirements.txt` (the existing training Docker should suffice).
The examples use the existing mount conventions:

- Original NIX + trial CSVs under `/source/tvsd` (read-only).
- Recording manifest `/project/data/reports/tvsd_manifest.csv` containing
  `session,array,area,units,source_path`. It is produced by the existing
  TVSD inventory; replace with your own manifest if needed.
- Original extracted THINGS image tree `/project/data/assets/THINGS`.
- Output `/project/data/processed/jorge_v1` (separate from the current model).

Unpack into `/project/jorge_tvsd_preprocessing/`. Then:

```bash
cd /project/jorge_tvsd_preprocessing
python -m unittest discover -s tests -v

# 1: Recording-specific units and 40–160 ms spike counts (default)
python scripts/extract_ks4.py --all --area V1 --manifest /project/data/reports/tvsd_manifest.csv --output-root /project/data/processed/jorge_v1

# 2: Presentation ID → original THINGS image, checked for collisions
python scripts/audit_images.py --processed-root /project/data/processed/jorge_v1 --images-root /project/data/assets/THINGS

# 3: Unlabelled, provisional longitudinal identity matching (original weights)
python scripts/track_units.py --root /project/data/processed/jorge_v1

# 4: Select tracked neurons, preserving conservative thresholds by default
python scripts/select_tracks.py --root /project/data/processed/jorge_v1

# 5: Prepare selected, cross-session tracked neurons and masked count matrices
python scripts/dataset.py prepare --processed-root /project/data/processed/jorge_v1 --images-root /project/data/assets/THINGS --output-dir /project/data/processed/jorge_v1/model_dataset

# 6: Fit MATLAB/BT.601 grayscale global mean/std on training images (default)
python scripts/calibrate_normalization.py --dataset-root /project/data/processed/jorge_v1/model_dataset --image-size 224

# 7: Validate each animal and partition
python scripts/dataset.py check --animal F --partition train --prepared-root /project/data/processed/jorge_v1/model_dataset --image-size 224
python scripts/dataset.py check --animal F --partition val --prepared-root /project/data/processed/jorge_v1/model_dataset --image-size 224
python scripts/dataset.py check --animal F --partition test --prepared-root /project/data/processed/jorge_v1/model_dataset --image-size 224
python scripts/dataset.py check --animal N --partition train --prepared-root /project/data/processed/jorge_v1/model_dataset --image-size 224
python scripts/dataset.py check --animal N --partition val --prepared-root /project/data/processed/jorge_v1/model_dataset --image-size 224
python scripts/dataset.py check --animal N --partition test --prepared-root /project/data/processed/jorge_v1/model_dataset --image-size 224

# 8: See a Python DataLoader example
python examples/load_data.py
```

`extract_ks4.py --session macaqueF_TVSD_20240112_B1 ...` can be used to
probe one session first. Completed sessions are skipped on subsequent `--all`
runs. The identity and dataset scripts check duplicate unit IDs, original stimulus
split/order, same-session neuron collisions, and absent-response masks.
Neuron selection is explicit, auditable, and independent of neural-network architecture.

## Rebuild, calibration and reproducibility safeguards

- `--expected-sessions` defaults to **45** for the full TVSD dataset (17 F, 28 N);
  this shared default applies to image audit and dataset preparation. Override
  it explicitly when intentionally working on fewer sessions.
- Response extraction uses **[40,160) ms** by default; `extract_ks4.py --window-ms START STOP`
  is supported. Dataset preparation reads the interval stored in each NPZ,
  rejects inconsistent windows across sessions, and records the agreed window
  in `dataset_info.json`. Every loader uses that metadata.
- After `dataset.py prepare --overwrite`, rerun calibration. The previous
  `normalization.json` is deleted, and the loader requires matching SHA-256
  fingerprints for the complete image-level split and image-path mapping.
- `calibrate_normalization.py --max-images N` is *only a diagnostic*. It requires
  `--output /some/other/filename.json`, samples images using a fixed random seed,
  and cannot overwrite production `normalization.json`. A loader rejects partial
  calibration files even when passed explicitly.
- `audit_images.py --images-root ...` checks original pixel dimensions of
  every resolved image. Non-square images are recorded in `images/nonsquare_images.csv`
  and cause the audit to fail rather than silently stretch an image.
- The original **unit inventory is retained**, including singleton units. The
  default selected cohort, however, requires 3 sessions and therefore excludes
  singletons. With `--min-recordings 1 --min-train 0`, singletons are eligible:
  undefined singleton waveform correlation is *not* treated as inconsistency.

## Tracked neurons, selection cutoffs and changing matching weights

These are **provisional longitudinal IDs**, not validated biological identities.
The default selection follows the earlier conservative thresholds **where possible
without Katka labels**:

| Selection flag | Default | Interpretation |
| --- | ---: | --- |
| `--min-recordings` | 3 | Track must span ≥3 distinct sessions |
| `--min-train` | 2000 | Minimum original training-image presentations **retained by the chosen reappearance policy**, before the image-level train/validation split. With `merge` this is the whole track; with `first-only` it is the earliest retained session. |
| `--min-pairwise-r` | 0.93 | Minimum all-pair waveform correlation, when ≥2 recordings exist |
| `--max-rate-ratio` | 3.0 | Maximum ratio of session firing rates |
| `--max-distinct-r` | 0.98 | Optional same-electrode-site ambiguity veto, only when `electrode_sites.csv` is supplied or present |

Katka's class-consistency/polarity checks and manual exclusions **cannot be
replicated** without Katka data. Unless `identity_diagnostics/electrode_sites.csv`
(or `--site-audit ...`) is provided, the simultaneous-unit site ambiguity veto
is **not evaluated**; `selection_config.json` records this fact. Original KS4
units that fail selection stay in the original inventory and the selection audit.
The `--keep-all-tracks` option overrides quality cutoffs; for singletons use
`--min-recordings 1` (and reduce `--min-train` if necessary).

Reappearance options at **neuron selection**:

```bash
# Default: same proposed neuron across different sessions -> one column
python scripts/select_tracks.py --reappearance merge

# Keep the first session's observation of each proposed neuron only
python scripts/select_tracks.py --reappearance first-only

# Discard *entire* tracks found in more than one recording
# (singletons require a different threshold)
python scripts/select_tracks.py --reappearance exclude-repeated --min-recordings 1 --min-train 0
```

For `exclude-repeated`, you **must** explicitly supply `--min-recordings 1`:
tracks appearing in multiple sessions are ineligible by definition. The command
raises an explanatory error rather than silently producing an empty cohort.
Single-session recordings may also require lowering `--min-train 2000`, as shown
above; otherwise the ordinary minimum-data filter still applies.

For `first-only`, track-level quality and recording-count filters are evaluated
using all available sessions, but `--min-train` is evaluated **only on the first
retained session**. This prevents inclusion based on training observations that
are subsequently discarded. `neuron_index.csv` and `track_selection_audit.csv`
contain both the full-track `n_train_presentations` and the effective
`n_kept_train_presentations`, `n_kept_test_presentations`, and
`n_kept_recordings`. These counts refer to the original stimulus train/test
labels, **before** the separate image-level train/validation split; they are
not post-validation counts or measures of response-mask completeness.
`--keep-all-tracks` deliberately bypasses the quality/data thresholds, but
never bypasses the reappearance rule.

Each selection regenerates `unit_to_neuron.csv` and `neuron_index.csv`. To
update an existing prepared dataset, rerun `dataset.py prepare --overwrite`
then recalibrate normalization. Never mix selection files and response NPZs
from different runs.

The **matching score weights** are separate from selection cutoffs. Defaults
in `track_units.py` are firing rate 2.0, waveform 1.6, ISI 1.0 and amplitude
0.35, with the original absolute gates, `--min-score 1.3` and
`--min-margin 0.5`. For example:

```bash
python scripts/track_units.py --root /project/data/processed/jorge_v1 \
  --proposal-rate-weight 1.0 --rate-weights 1.0 2.0 3.0 \
  --waveform-weight 2.0 --isi-weight 1.0 --amplitude-weight 0.35
# IMPORTANT: rerun select_tracks.py and dataset.py prepare --overwrite after changing matching weights.
```

`--rate-weights` writes a sensitivity comparison; `--proposal-rate-weight`
selects which resulting map is saved. Reweighting changes the inferred
identities, so it changes the data representation, **not just the optimizer**.

## Image mode — scientific choice, not hidden preprocessing

The original Cadena V1 experiment used grayscale stimuli; **we have not verified
that TVSD presented grayscale THINGS images or which RGB conversion was used**.
The **default is MATLAB `rgb2gray` / BT.601**, including when no color option
is supplied. Alternatives are activated **only** when explicitly requested:

| `--color-mode` | Output before channel replication | RGB weights |
| --- | --- | --- |
| `gray-bt601` (default) | 1-channel grayscale | 0.2989, 0.5870, 0.1140 |
| `gray-skimage` | 1-channel grayscale | 0.2125, 0.7154, 0.0721 |
| `rgb` | Original 3-channel color | No grayscale conversion |

For an explicit alternative, supply `--color-mode gray-skimage` or `--color-mode rgb`
to `scripts/calibrate_normalization.py`. If `rgb` is chosen, calibration fits *per-channel* training statistics rather
than the single global grayscale mean/std. The selected mode, image resolution,
means and standard deviations are written to `model_dataset/normalization.json`
and read by **every** data loader. Choose **one** mode for a prepared dataset
and do not change that file mid-experiment. To test a different mode, write a
separate dataset directory or normalization file and pass it explicitly to
`TVSDDataset` as `normalization_path`.

### Loader API

```python
from dataset import TVSDDataset
train = TVSDDataset(animal='F', partition='train', image_size=224)
val   = TVSDDataset(animal='F', partition='val',   image_size=224)
test  = TVSDDataset(animal='F', partition='test',  image_size=224)
```

Here `dataset` is the file `scripts/dataset.py`; add that directory to
`PYTHONPATH` or use `examples/load_data.py`.
Each example returns a dict with:

- `image`: `float32 [3,224,224]`, normalized full image
- `responses`: `float32 [n_selected_neurons_for_monkey]` observed spike counts
- `response_mask`: `bool [n_selected_neurons_for_monkey]`, which **must** be applied in loss/evaluation
- `animal`, `session`, `source_row`, `stimulus_id` to trace provenance

Use separate loaders for F and N; their neuron axes differ. `neuron_index.csv` lists each selected provisional neuron; `unit_to_neuron.csv`
maps each constituent original KS4 UUID and session to that neuron. The neuron
axis is not a particular network readout.

## Output and safeguards

`unit_instances.csv` and `sessions/*.npz` preserve original UUIDs, source
metadata, waveforms, raw spike times, counts and observed masks. The `images/`
folder contains the stimulus audit. `identity_diagnostics/joint/` optionally contains
proposed (not ground-truth) matches. `model_dataset/` contains compact count+mask
NPZs, `sample_index.csv`, `neuron_index.csv`, `unit_to_neuron.csv`, `dataset_info.json` and
`normalization.json`.

The selected neuron axis follows provisional cross-session identity tracks,
not independent recording-specific unit instances. Default quality cutoffs are
based on our earlier working protocol; **this will not exactly reproduce the
curated 733-neuron cohort**, which used Katka labels and further manual trimming.
All original KS4 records remain available unmodified. A synthetic NIX extraction test, an unlabelled tracking test, and
masked-loader/normalization tests are provided. No original raw recordings were
available for an end-to-end execution in this handoff environment.
