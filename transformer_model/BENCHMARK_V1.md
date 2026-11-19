# ZWM 170M on TVSD V1: paper-method benchmark

**Purpose:** Benchmark the existing published ZWM checkpoint before trying any
new readout, fine-tuning, Optuna optimization or Classical Experiments adapter.
The 1,173 tracked neurons remain a single F (804) + N (369) axis; nothing here
rewrites the completed single-layer run or the central project registry.
Reference source revisions reviewed: ZWM `eea0c5d` and BBScore
`b94b384c87d9032fc9894b19fb302bf82fcec3a9`.

## Protocol and sources

| Choice | Implementation | Source |
|---|---|---|
| Weights | `awwkl/zwm-bvd-170m/model.pt`, SHA-256 recorded | [ZWM](https://github.com/awwkl/ZWM) |
| Stimuli | RGB THINGS photographs processed by ZWM's upstream one-image method | [BBScore TVSD loader](https://github.com/neuroailab/bbscore_public/blob/main/data/TVSD.py) |
| Encoder | Blocks `0,2,…,22`, capture `.ln_1`; 12 activations from **one** forward pass | [ZWM neural-alignment analysis](https://github.com/awwkl/ZWM/blob/main/scripts/eval/neural_alignment/analyze_neural_alignment.py) |
| Probe | `sklearn.RidgeCV(alpha_per_target=True)`, no feature standardization | [BBScore RidgeMetric](https://github.com/neuroailab/bbscore_public/blob/main/metrics/ridge.py) |
| Alphas | All 21 values `10⁻¹⁰, 10⁻⁹, …, 10¹⁰` | BBScore `RidgeMetric` default |
| Cross-validation | 10 outer folds, shuffle seed 42; stimulus-group isolated on our repeated-image rows | [BBScore KFold defaults](https://github.com/neuroailab/bbscore_public/blob/main/metrics/utils.py) |
| Noise ceiling | 50 split-half correlations, each Spearman–Brown corrected, then averaged | [BBScore split-half](https://github.com/neuroailab/bbscore_public/blob/main/data/utils.py) |
| Score | Spearman–Brown-transformed Pearson divided by the noise ceiling, per neuron | [BBScore Pearson scorer](https://github.com/neuroailab/bbscore_public/blob/main/metrics/utils.py) |
| Reference | Peak V1 score **0.95** for ZWM 170M | Aw et al., *Zero-shot World Models Are Developmentally Efficient Learners*, Table S25 |

Each image is extracted once, capturing all 12 layers through independent
hooks. The original `ZWMFeatures` and frozen single-layer trainer remain
untouched. The held-out test score for each layer is computed only after its
ridge probe is fitted on the frozen train+validation partition. The reported
best layer is the maximum across all 12 test scores, as in a best-layer table.
Per-layer out-of-fold correlations are also saved.

**Comparison boundary:** This is a reproduction of the publicly documented
*scoring method* adapted to **our data**, not a reproduction of the exact
published 0.95 result. Their TVSD V1 benchmark uses the original 1024 V1
electrode channels from F and N and the provided `THINGS_normMUA.mat` response
matrices. We instead use **1,173 selected/tracked neurons**, each with masked
session observations and a fixed 40–160 ms response window, from our frozen
preprocessing. The number of samples, repeated-image availability and response
aggregation can therefore differ. BBScore's published ZWM image feature
aggregation code is not in its public package; **mean token pooling remains an
explicit unverified assumption**. We require at least 20 repeated test images
per neuron to estimate a finite ceiling, and exclude nonpositive reliabilities
instead of BBScore's numerical `1e-6` floor. Every difference is material if
you intend to interpret differences from 0.95 as model performance.

## MetaCentrum execution

The benchmark has **no SQLite writes**, no coordinator and no automatic PBS
submissions. Every job writes separate files under
`$ROOT/results/zwm_v1_paper`; input feature caches are immutable after
completion. `--phase fit` saves each observation-mask group individually so a
walltime interruption can be resumed without repeating completed groups.

From the MetaCentrum login node after `git pull`:

First, on a short **allocated GPU job** (not the login node), check the exact
`.ln_1` hooks on one already-present THINGS image:

```bash
ROOT=/storage/brno2/home/jonas_prokop/visual_models
CODE="$ROOT/code/papala_tsvd_trainer"
export SINGULARITYENV_PYTHONPATH=/project:/mnt/tvsd/code/ZWM
singularity exec --nv --bind "$CODE:/project" --bind "$ROOT:/mnt/tvsd" \
  --pwd /mnt/tvsd/code/ZWM "$ROOT/containers/tvsd-bookworm.sif" \
  /mnt/tvsd/envs/zwm-py310/bin/python \
  -m transformer_model.benchmark_v1 --phase smoke \
  --config /project/configs/tvsd_zwm_paper_v1.yaml \
  --output /mnt/tvsd/results/zwm_v1_paper \
  --image /mnt/tvsd/data/assets/THINGS/object_images/doormat/doormat_16s.jpg
```

Expect twelve finite `(1, 768)` feature arrays. This is the full upstream
ZWM forward—not just the mocked unit test. Then, back on the login node:

```bash
cd ~/visual_models/code/papala_tsvd_trainer
bash -n jobs/zwm_v1_extract.pbs jobs/zwm_v1_fit.pbs
qsub jobs/zwm_v1_extract.pbs
```

Wait for the GPU job to finish and confirm all unique train **and test** images
were cached (a little more than the previous 22,248 train/validation images).
Do **not** schedule CPU probes before this.
Then a first **single CPU layer** is the walltime and correctness check:

```bash
qsub -v LAYER_INDEX=0 jobs/zwm_v1_fit.pbs
```

After checking that layer 0 finishes and yields finite V1 scores, submit the
remaining layers as an array with **at most two simultaneous 4-CPU jobs**:

```bash
qsub -J 1-11%2 jobs/zwm_v1_fit.pbs
```

If `%2` is not accepted by the local PBS version, submit individual indices
manually. MetaCentrum schedules each array element as a separate node job.

Finally, on an allocated **CPU** node with the same Singularity image, run:

```bash
ROOT=/storage/brno2/home/jonas_prokop/visual_models
CODE="$ROOT/code/papala_tsvd_trainer"
SIF="$ROOT/containers/tvsd-bookworm.sif"
export SINGULARITYENV_PYTHONPATH=/project:/mnt/tvsd/code/ZWM
singularity exec --bind "$CODE:/project" --bind "$ROOT:/mnt/tvsd" \
  "$SIF" /mnt/tvsd/envs/zwm-py310/bin/python \
  -m transformer_model.benchmark_v1 --phase report \
  --config /project/configs/tvsd_zwm_paper_v1.yaml \
  --output /mnt/tvsd/results/zwm_v1_paper
```

Review `$ROOT/results/zwm_v1_paper/summary.json` for all 12 test results and
the best layer. Each `$ROOT/results/zwm_v1_paper/layers/NN.npz` includes the
learned linear readout, original F/N neuron IDs, chosen per-neuron alpha,
out-of-fold correlation, held-out noise ceiling, raw and corrected scores.
That preserves the exact probe for later Classical Experiments validation.

Run the fast local scorer tests before scheduling:

```bash
python3 -m unittest discover -s tests -p 'test_zwm_paper_benchmark.py' -v
```

The ZWM model forward hooks and single-SIF integration must also be exercised
on an allocated GPU compute node before all 12 layers are submitted.
