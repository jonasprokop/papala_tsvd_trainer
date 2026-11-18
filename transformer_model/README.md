# TVSD ZWM: one registry, one Singularity image

The PBS launcher chooses one of two Python prefixes inside `tvsd-bookworm.sif`.
Bind Brno2 **only** at `/mnt/tvsd`; self-binding `"$ROOT:$ROOT"` fails on
MetaCentrum. The original `sinz-base.sif` remains untouched for rollback:

| family | interpreter | strategy |
|---|---|---|
| `sinz` | `/mnt/tvsd/envs/tvsd-py39/bin/python` | `gradient` (existing) |
| `transformer_neural` | `/mnt/tvsd/envs/zwm-py310/bin/python` | `frozen_ridge` |

## Build and stage the modern image

Run the manually triggered GitHub Actions workflow
`.github/workflows/metacentrum-image.yml`. It publishes the image from
`Dockerfile.metacentrum` to GHCR without building on MetaCentrum. The base is
Debian Bookworm (glibc 2.36), with the Sinz Python 3.9 dependencies installed.
No ZWM code, checkpoints or data are baked into the image.

After a successful build, pull **once** during a small allocated CPU job; use
the version-specific `bookworm-<git SHA>` tag shown in Actions:

```bash
ROOT=/storage/brno2/home/jonas_prokop/visual_models
singularity pull "$ROOT/containers/tvsd-bookworm.sif" \
  docker://ghcr.io/jonasprokop/papala_tsvd_trainer:bookworm-<git-sha>
singularity exec "$ROOT/containers/tvsd-bookworm.sif" getconf GNU_LIBC_VERSION
```

GHCR images are initially private; allow a public package download or arrange
registry authentication before pulling. Do not replace `sinz-base.sif`.

First check the **existing** Python 3.9 prefix against the new SIF. It was
created with `/usr/bin/python3.9`, which the new image supplies as an alias.
The image's build check does not prove that every old venv wheel imports:

```bash
CODE="$ROOT/code/papala_tsvd_trainer"
singularity exec --bind "$ROOT:/mnt/tvsd" --bind "$CODE:/project" \
  "$ROOT/containers/tvsd-bookworm.sif" \
  /mnt/tvsd/envs/tvsd-py39/bin/python -c \
  'import torch, transformers, sklearn, optuna, yaml; print(torch.__version__)'
```

Create an independent Python 3.10 prefix with Mamba/Conda on an allocated CPU
node. A Python 3.9 `venv` cannot supply Python 3.10:

```bash
ROOT=/storage/brno2/home/jonas_prokop/visual_models
CODE="$ROOT/code/papala_tsvd_trainer"
module add micromamba
micromamba create --prefix "$ROOT/envs/zwm-py310" -c conda-forge python=3.10 pip -y
# Clone/pin https://github.com/awwkl/ZWM under "$ROOT/code/ZWM" first.
singularity exec --bind "$ROOT:/mnt/tvsd" --bind "$CODE:/project" \
  "$ROOT/containers/tvsd-bookworm.sif" \
  /mnt/tvsd/envs/zwm-py310/bin/python -m pip install \
     -e /mnt/tvsd/code/ZWM \
     -r /project/requirements-transformer-extra.txt
```

**GPU compatibility is unverified.** Check CUDA, PyTorch 2.x, and the upstream
checkpoint inside the image on an allocated node before training. Pin/checksum
the upstream ZWM source and checkpoint; leave the Sinz environment untouched.

## Execution

Register candidates with `sinz_model.cli candidates create` and
`configs/tvsd_zwm_ridge.yaml` in Python 3.9. Pass
`--output-root /mnt/tvsd/results`; the central project registry must be
`/mnt/tvsd/results/results.sqlite`, including for Optuna.

Check the paths before submitting a job:

```bash
ROOT=/storage/brno2/home/jonas_prokop/visual_models
CODE="$ROOT/code/papala_tsvd_trainer"
test -d "$CODE" && test -f "$ROOT/containers/tvsd-bookworm.sif" \
  && test -x "$ROOT/envs/zwm-py310/bin/python" \
  && test -L "$ROOT/envs/tvsd-py39/bin/python"
```

PBS uses `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. Pre-fill the ZWM
checkpoint and Hugging Face cache in a separate allocated-node warm-up if
required; workers intentionally cannot download missing assets.

Run one real-image ZWM smoke test under Python 3.10 before training, through the
same `tvsd-bookworm.sif` with binds `"$CODE:/project"`, `"$ROOT:/mnt/tvsd"`:

```bash
# Run on an allocated GPU compute node (not the login node).
singularity exec --nv \
  --bind "$CODE:/project" --bind "$ROOT:/mnt/tvsd" \
  "$ROOT/containers/tvsd-bookworm.sif" \
  /mnt/tvsd/envs/zwm-py310/bin/python -m transformer_model.smoke \
  --config /project/configs/tvsd_zwm_ridge.yaml \
  --image /mnt/tvsd/data/assets/THINGS/ACTUAL_IMAGE.jpg
```

Once the smoke test succeeds, submit one ZWM job:

```bash
qsub -v CANDIDATE_ID=<uuid> jobs/zwm_train.pbs
```

This runs `transformer_model.launch` directly in Python 3.10 (no Python 3.9
parent). Its **only writable SQLite is scratch-local**. Each job creates a
consistent handoff at
`$ROOT/results/zwm_handoffs/$CANDIDATE_ID/$PBS_JOBID.sqlite`. After it ends,
collect explicitly on the login node, running Python 3.9 **inside the same
image** to avoid host-library incompatibilities:

```bash
ROOT=/storage/brno2/home/jonas_prokop/visual_models
CODE="$ROOT/code/papala_tsvd_trainer"
cd "$CODE"
export SINGULARITYENV_PYTHONPATH=/project
singularity exec --bind "$CODE:/project" --bind "$ROOT:/mnt/tvsd" \
  "$ROOT/containers/tvsd-bookworm.sif" \
  /mnt/tvsd/envs/tvsd-py39/bin/python -m sinz_model.registry_handoff collect \
  --project-db /mnt/tvsd/results/results.sqlite \
  --worker-db "/mnt/tvsd/results/zwm_handoffs/$CANDIDATE_ID/$PBS_JOBID.sqlite" \
  --candidate-id "$CANDIDATE_ID"
```

The collector imports the one run into a locally staged copy of the project
database, then atomically publishes it. A shared project coordinator lock
prevents simultaneous Optuna collection. It is intentionally never merged by
the compute node.

If a collector is killed, recover **only** when the lock owner PID is confirmed
dead on the *same host* where it was created:

```bash
singularity exec --bind "$CODE:/project" --bind "$ROOT:/mnt/tvsd" \
  "$ROOT/containers/tvsd-bookworm.sif" \
  /mnt/tvsd/envs/tvsd-py39/bin/python -m sinz_model.registry_handoff break-lock \
  --project-db /mnt/tvsd/results/results.sqlite
```

The command refuses live PIDs, malformed owner records, and locks belonging to
other hosts; it never recursively deletes the lock directory. Investigate
cross-host leftovers manually. For a worker retry, check a completed PBS ID
with `qstat -x -f JOB_ID` (extended history) before resubmitting.

Optuna also has an independent study-directory lock. If *that* lock is stale,
run `python -m sinz_model.optuna_batch break-lock --root STUDY_DIR` inside the
same image and on the host recorded by its `owner` file. It uses identical
checks; recovering either lock never submits PBS jobs.

Optuna's PBS array workers likewise use node-local SQLite and produce a handoff.
Their wrapper traps SIGTERM, attempts to stop the child, and snapshots the DB
before exit. A hard SIGKILL cannot be trapped; all already completed feature
chunks remain persistent, but the last scratch-only SQLite state may be lost.
Do not collect unfinished trials until the array has exited.

To avoid accidental cluster allocations, `optuna_batch submit` **prints the
PBS array command without submitting it** unless `--execute` is supplied.
`--execute` submits only the array (concurrency defaults to four); an
additional PBS collection job requires the separate `--schedule-collector`
opt-in. Prefer manual `optuna_batch collect` after `qstat` shows the array
has finished. This avoids reserving a separate PBS node merely to merge SQLite.
PBS worker runtimes are capped at four BLAS threads and have offline Hugging
Face mode enabled, so they cannot silently download weights on compute nodes.

For `frozen_ridge`, require full training/validation data and disable Optuna
pruning: ridge yields one final validation score, not SGD epochs.
Feature extraction is resumable in chunks. ZWM's upstream `ZWMPredictor` owns
input preprocessing and single-image inference; do not substitute the Sinz
224px normalized tensor path.

If an Optuna ridge PBS worker fails or is preempted, requeue **the same** trial array
index. The PBS wrapper seeds local SQLite from `worker_result.sqlite` when it
exists (otherwise from `worker_seed.sqlite`). The worker retains the original
run ID, preserves prior failure details in `ridge_retry_history.json`, clears
partial fit metrics and checkpoints, and reuses cached backbone feature chunks.
This is a **restart of the ridge fit**, not an SGD epoch-checkpoint resume.
For a retry to recover features, leave `project.feature_cache_root` intact.

Ridge CV batches neurons with identical observation masks, performs one
spectral decomposition per fold, and selects alpha per neuron. PBS logs report
the mask-group count and largest group, which indicate how much batching helps.
The per-neuron sklearn implementation is retained
as `fit_masked_ridge_cv_reference` and can be selected with
`ridge.solver: sklearn_reference` (default: `spectral`).
The two can be compared in seconds on synthetic CPU data:

```bash
python -m transformer_model.ridge_smoke
```

No PBS job or GPU is necessary for this small test. Actual
walltime must still be measured on TVSD feature dimensions.

The SGD `tvsd_optuna_space.yaml` is **not** a ZWM search space. Layer/pooling
search currently requires re-extracting features; a multi-layer cache would
make that comparison cheaper. Do not launch a ZWM Optuna study with the SGD space.

This pipeline is a **frozen-feature neural encoding baseline**, not a
reproduction of ZWM's published zero-shot benchmarks. The ZWM reference
checkpoint uses 256px inputs, handled by upstream inference; the TVSD training
split and `GroupKFold` stimulus grouping prevent image-identity leakage in
ridge selection. See [ZWM's model/API documentation](https://github.com/awwkl/ZWM)
and [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html).

Each trained checkpoint and exported bundle contains `valid_neuron_mask`,
aligned exactly with `neuron_ids` in F/N order. False means the neuron's ridge
fit lacked enough training observations/groups; its prediction may be NaN.
Evaluation checks the complete ordered IDs against `neuron_index.csv` before
using the model. Bundle loading also checks that the readout payload IDs and
mask match the manifest, and accepts `expected_neuron_ids` to validate an
external experiment's neuron axis before inference.

**Classical Experiments inference in Python 3.9 remains unverified.** Export
contains the backbone checkpoint, readout and neuron ordering, but the upstream
ZWM inference runtime may still require Python 3.10.
