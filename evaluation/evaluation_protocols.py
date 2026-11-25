"""Comparable, explicitly named neural-response scoring protocols.

All metrics are per neuron on held-out images. Reliability-based metrics use
only images with >=2 observations for that neuron; `cadena_corr_to_average`
and the FEVE total variance/MSE use every observed image. No model refitting or
test-set parameter selection occurs here. Missing neuron observations are masked.
Random draws are seeded per neuron from a stable hash of its ID (str or int;
e.g. 'F/12'); pass the global neuron IDs when scoring a column subset so each
neuron keeps the same values. The original IDs are returned for joins.

Every output key has a level in METRIC_LEVELS:
  formula_match  the published formula, reproduced, on data meeting its
                 assumptions. NOT the published number: repeat counts, image
                 coverage and neuron selection still differ.
  adapted      the published formula adapted to sparse tracked observations
  model_based  inferred through a noise model, not measured
  diagnostic   project-specific, no published counterpart

Three reliability definitions are kept separate, because they are ceilings for
different targets and are not interchangeable:

  single_trial_pairwise_reliability_estimate
      Estimate of TVSD `reliab` (Papale et al. 2025): Pearson r across images
      between two single repetitions. TVSD's `rep` index is randomised, so its
      repetition vectors are random within-image pairings; here each of
      `splits` draws picks two distinct trials per image at random and the
      draws are averaged. Same expectation as the published quantity on
      exchangeable repeats; equivalence on real data must be checked with
      `papale_reliab_dense` against the released `reliab`.
  split_half_sb_reliability
      Spearman-Brown-corrected split-half r of trial averages: the ceiling for
      predicting the trial-AVERAGED response (CC_norm = r / sqrt(R)).
  papale_oracle_adapted
      Papale et al. 2025 STAR Methods: r across images between one held-out
      repetition and the mean of the remaining 29, repeated for each of the 30
      repetitions and averaged (`papale_oracle_dense`). Adapted to variable
      repeat counts: per draw, one random held-out trial per image vs the mean
      of that image's remaining trials. With unequal repeats the "rest" means
      differ in reliability across images, so on our data this is not
      numerically Papale's 30-repeat oracle. Its expectation is
      sqrt(EV * R_{n-1}): slightly BELOW what a perfect model reaches on single
      trials (sqrt(EV)), and NOT a ceiling for trial-averaged prediction
      (that is sqrt(split_half_sb_reliability)).

`prediction=None` computes only the data-side quantities (reliabilities,
explainable fraction, session diagnostics); prediction-dependent keys are NaN.

Papale's published formulas, applied per neuron ("electrode" -> "neuron"):
  papale_reliab_balanced_adapted / papale_oracle_balanced_adapted
      The dense definitions (`papale_reliab_dense`, `papale_oracle_dense`) on a
      balanced block: k = the largest repeat count that at least
      BALANCED_IMAGE_FRACTION of the neuron's repeated images reach; those
      images, k randomly chosen trials each (TVSD's own rep order is random),
      averaged over BALANCED_DRAWS subsamples. `balanced_order='presentation'`
      keeps presentation order instead (exact, deterministic on a complete
      block), but then repeat vectors line up by session and constant session
      offsets cancel, which TVSD's shuffled repeats do not do; use it only to
      check formula equality.
      reliab does not depend on k in expectation, so it is directly comparable
      to an electrode's `reliab` and to the 0.3 cutoff. The oracle does depend
      on k (the "rest" mean has k-1 trials), so it is only Papale's number when
      k = 30. `papale_oracle_30rep_model_estimate` is NOT measured: it is
      sqrt(rel * R_29), R_29 = 29 rel / (1 + 28 rel), under an i.i.d.
      signal-plus-noise model; keep it apart from measured oracles.
      Not transferred: Papale's MUA normalisation and response window.

With `sessions` [presentations], the pairwise estimate is also computed within
sessions (r across images inside each session, then a weighted mean over
sessions; needs >=20 images with >=2 trials in a session) and for pairs drawn
from two different sessions, plus
repeat counts per session and the size of per-session response offsets, so
trial noise, session variation and too few observations can be told apart.

Cadena et al. 2024, as implemented in NeuroVisFit
(neurovisfit/evaluation/metrics.py, VERIFIED in source):
  correlation  neuralpredictors.measures.corr: Pearson with eps=1e-8 added to
               the ddof=0 std (constant input -> 0, not NaN), across images,
               on trial-averaged targets and outputs.
  FEV / FEVE   per neuron over repeated images: noise = mean over images of
               within-image var (ddof=1); total = var over all presentations
               (ddof=1); FEV = (total-noise)/total;
               FEVE = 1 - (MSE over presentations - noise)/(total - noise).
  aggregation  (in compare_evaluations) keep FEV > 0.15 strictly, NaN
               correlations -> 0, mean over neurons.
The code assumes every test image is repeated for every neuron. Per neuron:
  cadena_formula_applicable  True when every test image in the table has >=2
                           observations for this neuron; then all cadena_*
                           values match NeuroVisFit's calculation (formula
                           match, not the published population).
  otherwise (tracked)      the correlation uses every observed image (project
                           Sinz convention) and FEV/FEVE use only images with
                           >=2 observations, so noise, total and MSE refer to
                           one image set.
"""
from __future__ import annotations

import hashlib
from itertools import combinations

import numpy as np

MIN_PAIR_IMAGES = 20
BALANCED_IMAGE_FRACTION = 0.8
BALANCED_DRAWS = 10
# Meaningless without a model; NaN when score_protocols(prediction=None, ...).
PREDICTION_KEYS = ('cadena_corr_to_average', 'cadena_feve', 'raw_repeated_pearson',
                   'pearson_noise_corrected', 'bbscore_transformed_pearson',
                   'bbscore_noise_corrected')

# Original Brain-Score internal consistency defaults to 10 train/test splits.
# BBScore uses 50; default score_protocols(splits=50) supplies both.
BRAINSCORE_SPLITS = 10


METRIC_LEVELS = {
    'cadena_corr_to_average': 'formula_match if cadena_formula_applicable else adapted',
    'cadena_feve': 'formula_match if cadena_formula_applicable else adapted',
    'cadena_explainable_fraction': 'formula_match if cadena_formula_applicable else adapted',
    'raw_repeated_pearson': 'formula_match',
    'split_half_sb_reliability': 'adapted',
    'split_half_sb_by_split': 'adapted; retains per-split values for Brain-Score aggregation',
    'pearson_noise_corrected': 'adapted',
    'bbscore_transformed_pearson': 'formula_match',
    'bbscore_noise_corrected': 'adapted',
    'bbscore_split_half_sb_reliability': 'adapted; BBScore undefined-half convention',
    'bbscore_clipped_ceiling': 'adapted; BBScore floor of 1e-6',
    'single_trial_pairwise_reliability_estimate': 'adapted',
    'papale_oracle_adapted': 'adapted',
    'papale_reliab_balanced_adapted': 'adapted',
    'papale_oracle_balanced_adapted': 'adapted',
    'papale_oracle_30rep_model_estimate': 'model_based',
    'papale_reliab_balanced_adapted_mc_se': 'diagnostic',
    'pairwise_reliability_within_session': 'diagnostic',
    'pairwise_reliability_between_sessions': 'diagnostic',
    'median_repeats_per_session_cell': 'diagnostic',
    'session_offset_sd_over_signal_sd': 'diagnostic',
}


def neuron_seed(neuron_id):
    """Stable across runs, machines and Python versions (unlike hash())."""
    digest = hashlib.sha256(str(neuron_id).encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'little')


def corr_neuralpredictors(x, y, eps=1e-8):
    """neuralpredictors.measures.corr for 1-D inputs (Cadena/NeuroVisFit)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) == 0:
        return np.nan
    x = (x - x.mean()) / (x.std(ddof=0) + eps)
    y = (y - y.mean()) / (y.std(ddof=0) + eps)
    return float(np.mean(x * y))


def _corr(x, y):
    if len(x) < 2:
        return np.nan
    a = x - np.mean(x)
    b = y - np.mean(y)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 0 else np.nan


def bbscore_pearson_scorer(y_true, y_pred, eps=1e-8):
    """BBScore metrics/utils.py pearson_correlation_scorer, including constants."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if len(y_true) < 2:
        return np.nan
    if np.var(y_true) < eps or np.var(y_pred) < eps:
        return 1.0 if np.allclose(y_true, y_pred, atol=eps) else 0.0
    r = _corr(y_true, y_pred)
    if r == 1.0:
        return 1.0
    with np.errstate(divide='ignore', invalid='ignore'):
        return float(np.divide(2 * np.float64(r), 1 + np.float64(r)))


def bbscore_clip_ceiling(value):
    """BBScore metrics/base.py BaseMetric.__init__: floor at positive 1e-6."""
    return float(max(value, 1e-6))


def _nanmean_or_nan(values):
    """Preserve an undefined estimate without an all-NaN RuntimeWarning."""
    values = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(values)) if np.isfinite(values).any() else np.nan


def papale_reliab_dense(reps):
    """Published definition on dense data: reps [n_reps, n_images] ->
    mean over all repetition pairs of r(rep_a, rep_b) across images."""
    reps = np.asarray(reps, dtype=np.float64)
    return _nanmean_or_nan([_corr(reps[a], reps[b])
                            for a, b in combinations(range(len(reps)), 2)])


def papale_oracle_dense(reps):
    """Published oracle on dense data: reps [n_reps, n_images] -> for each
    repetition k, r(reps[k], mean of the other reps) across images; mean over k."""
    reps = np.asarray(reps, dtype=np.float64)
    total = reps.sum(axis=0)
    n = len(reps)
    return _nanmean_or_nan([_corr(reps[k], (total - reps[k]) / (n - 1)) for k in range(n)])


def _within_session_reliability(cells, rng, draws):
    """cells: per image, {session: trials}. Correlate across images INSIDE each
    session (a session offset is then a constant and cannot pose as signal),
    average sessions weighted by their number of usable images, then draws."""
    labels = sorted({s for c in cells for s in c})
    rs = []
    for _ in range(draws):
        vals, weights = [], []
        for label in labels:
            a, b = [], []
            for c in cells:
                v = c.get(label)
                if v is not None and len(v) >= 2:
                    i, j = rng.choice(len(v), size=2, replace=False)
                    a.append(v[i]); b.append(v[j])
            if len(a) >= MIN_PAIR_IMAGES:
                r = _corr(np.asarray(a), np.asarray(b))
                if np.isfinite(r):
                    vals.append(r); weights.append(len(a))
        if vals:
            rs.append(float(np.average(vals, weights=weights)))
    return float(np.mean(rs)) if rs else np.nan


def papale_balanced(trials_per_image, rng, min_images=MIN_PAIR_IMAGES, order='random'):
    """Published dense formulas on the neuron's largest balanced block.
    Returns (reliab, oracle, k, n_images, Monte-Carlo SE of reliab)."""
    counts = np.array([len(v) for v in trials_per_image])
    if len(counts) == 0:
        return np.nan, np.nan, 0, 0, np.nan
    k = int(np.quantile(counts, 1 - BALANCED_IMAGE_FRACTION, method='lower'))
    use = [v for v in trials_per_image if len(v) >= k]
    if k < 2 or len(use) < min_images:
        return np.nan, np.nan, k, len(use), np.nan
    if order not in ('random', 'presentation'):
        raise ValueError("order must be 'random' or 'presentation'")
    # 'random': k random trials per image, as TVSD's reshuffled rep index
    # (repeat vectors mix sessions; additive session offsets lower reliab).
    # 'presentation': trials in presentation order, deterministic when every
    # image has exactly k; repeat vectors then line up by session and constant
    # session offsets cancel. Use it only to check formula equality on dense data.
    exact = order == 'presentation' and all(len(v) == k for v in use)
    rel, orc = [], []
    for _ in range(1 if exact else BALANCED_DRAWS):
        block = np.stack([v[:k] if order == 'presentation' else rng.permutation(v)[:k]
                          for v in use], axis=1)
        rel.append(papale_reliab_dense(block))
        orc.append(papale_oracle_dense(block))
    # Monte-Carlo standard error over subsamples (0 for one deterministic pass).
    se = float(np.std(rel, ddof=1) / np.sqrt(len(rel))) if len(rel) > 1 else 0.0
    return float(np.mean(rel)), float(np.mean(orc)), k, len(use), se


def oracle_30rep_model_estimate(rel):
    rel = np.asarray(rel, dtype=np.float64)
    r29 = 29 * rel / (1 + 28 * rel)
    with np.errstate(invalid='ignore'):
        return np.sqrt(rel * r29)


def _pair_reliability(cells, rng, draws, kind):
    """cells: per image, list of per-session trial arrays.
    kind 'any' | 'between' restricts where the two trials come from."""
    rs = []
    for _ in range(draws):
        a, b = [], []
        for sessions in cells:
            if kind == 'any':
                pool = np.concatenate(sessions)
                if len(pool) < 2:
                    continue
                i, j = rng.choice(len(pool), size=2, replace=False)
                a.append(pool[i]); b.append(pool[j])
            else:
                if len(sessions) < 2:
                    continue
                s1, s2 = rng.choice(len(sessions), size=2, replace=False)
                a.append(rng.choice(sessions[s1])); b.append(rng.choice(sessions[s2]))
        if len(a) >= MIN_PAIR_IMAGES:
            rs.append(_corr(np.asarray(a), np.asarray(b)))
    rs = np.asarray(rs, dtype=np.float64)
    return float(np.nanmean(rs)) if np.isfinite(rs).any() else np.nan


def score_protocols(prediction, response, mask, image_ids, *, splits=50,
                    seed=42, min_repeated_images=20, neuron_ids=None,
                    sessions=None, balanced_order='random'):
    y = np.asarray(response, dtype=np.float64)
    reliability_only = prediction is None
    p = np.zeros_like(y) if reliability_only else np.asarray(prediction, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    ids = np.asarray(image_ids)
    if p.ndim != 2 or p.shape != y.shape or m.shape != p.shape or ids.shape != (len(p),):
        raise ValueError('Expected matching [presentations, neurons] arrays and image IDs')
    if splits < 1 or min_repeated_images < 2:
        raise ValueError('Invalid split count or repeated-image minimum')
    width = p.shape[1]
    neuron_ids = np.arange(width) if neuron_ids is None else np.asarray(neuron_ids)
    if neuron_ids.shape != (width,):
        raise ValueError('neuron_ids must have one ID per neuron column')
    if len(set(map(str, neuron_ids))) != width:
        raise ValueError('neuron_ids must be unique')
    sess = np.zeros(len(p), dtype=int) if sessions is None else np.asarray(sessions)
    if sess.shape != (len(p),):
        raise ValueError('sessions must have one entry per presentation')
    if not np.isfinite(y[m]).all():
        raise ValueError('Nonfinite observed responses')
    keys = ('cadena_corr_to_average', 'cadena_feve', 'cadena_explainable_fraction',
            'raw_repeated_pearson', 'split_half_sb_reliability',
            'pearson_noise_corrected', 'bbscore_transformed_pearson',
            'bbscore_noise_corrected', 'bbscore_split_half_sb_reliability',
            'bbscore_clipped_ceiling', 'single_trial_pairwise_reliability_estimate',
            'papale_oracle_adapted', 'papale_reliab_balanced_adapted',
            'papale_oracle_balanced_adapted', 'papale_oracle_30rep_model_estimate',
            'papale_reliab_balanced_adapted_mc_se')
    session_keys = ('pairwise_reliability_within_session',
                    'pairwise_reliability_between_sessions',
                    'median_repeats_per_session_cell', 'session_offset_sd_over_signal_sd')
    if sessions is not None:
        keys = keys + session_keys
    result = {key: np.full(width, np.nan) for key in keys}
    n_brainscore_splits = min(BRAINSCORE_SPLITS, splits)
    result['split_half_sb_by_split'] = np.full((width, n_brainscore_splits), np.nan)
    for key in ('n_repeated_images', 'n_cadena_images', 'n_observations', 'n_sessions',
                'papale_balanced_k', 'papale_balanced_n_images'):
        result[key] = np.zeros(width, dtype=int)
    result['cadena_formula_applicable'] = np.zeros(width, dtype=bool)
    result['neuron_id'] = np.asarray([str(v) for v in neuron_ids])
    groups = [np.flatnonzero(ids == item) for item in np.unique(ids)]
    for n in range(width):
        rng = np.random.default_rng([seed, neuron_seed(neuron_ids[n])])
        valid = m[:, n] & np.isfinite(p[:, n])
        result['n_observations'][n] = valid.sum()
        result['n_sessions'][n] = len(np.unique(sess[valid]))
        image_predictions, image_responses = [], []
        for rows in groups:
            selected = rows[valid[rows]]
            if len(selected):
                image_predictions.append(float(p[selected, n].mean()))
                image_responses.append(float(y[selected, n].mean()))
        result['n_cadena_images'][n] = len(image_predictions)
        result['cadena_corr_to_average'][n] = corr_neuralpredictors(
            np.asarray(image_predictions), np.asarray(image_responses))
        # FEV/FEVE on images with >=2 observations (NeuroVisFit assumes all are).
        rep_rows = [rows[valid[rows]] for rows in groups if valid[rows].sum() >= 2]
        result['cadena_formula_applicable'][n] = len(rep_rows) == len(groups)
        noise = total = explainable = np.nan
        if rep_rows:
            values = np.concatenate([y[r, n] for r in rep_rows])
            preds = np.concatenate([p[r, n] for r in rep_rows])
            noise = np.float64(np.mean([np.var(y[r, n], ddof=1) for r in rep_rows]))
            total = np.float64(np.var(values, ddof=1))
            explainable = total - noise   # numpy scalars: 0/0 -> NaN as in the source
            with np.errstate(divide='ignore', invalid='ignore'):
                result['cadena_explainable_fraction'][n] = explainable / total
                result['cadena_feve'][n] = 1 - (np.mean((values - preds) ** 2) - noise) / explainable
        repeated = []
        for rows in groups:
            selected = rows[valid[rows]]
            if len(selected) >= 2:
                values, labels = y[selected, n], sess[selected]
                cells = {s: values[labels == s] for s in np.unique(labels)}
                repeated.append((values, float(p[selected, n].mean()), cells))
        result['n_repeated_images'][n] = len(repeated)
        if len(repeated) < min_repeated_images:
            continue
        cells = [list(c.values()) for _, _, c in repeated]
        cell_dicts = [c for _, _, c in repeated]
        result['single_trial_pairwise_reliability_estimate'][n] = _pair_reliability(
            cells, rng, splits, 'any')
        if sessions is not None:
            result['pairwise_reliability_within_session'][n] = _within_session_reliability(
                cell_dicts, rng, splits)
            result['pairwise_reliability_between_sessions'][n] = _pair_reliability(
                cells, rng, splits, 'between')
            result['median_repeats_per_session_cell'][n] = float(
                np.median([len(v) for c in cells for v in c]))
            if result['n_sessions'][n] < 2:
                result['pairwise_reliability_between_sessions'][n] = np.nan
            # Session offset: mean of (trial - its image mean) per session, so
            # the image composition of a session does not masquerade as drift.
            resid, labels = [], []
            for vals, _, c in repeated:
                for label, v in c.items():
                    resid.append(v - vals.mean())
                    labels.append(np.full(len(v), label))
            resid, labels = np.concatenate(resid), np.concatenate(labels)
            offsets = [resid[labels == s].mean() for s in np.unique(labels)]
            if len(offsets) >= 2 and explainable > 0:
                result['session_offset_sd_over_signal_sd'][n] = float(
                    np.std(offsets, ddof=1) / np.sqrt(explainable))
        rel_b, orc_b, k_b, n_b, se_b = papale_balanced([v for v, _, _ in repeated], rng,
                                                        order=balanced_order)
        result['papale_reliab_balanced_adapted_mc_se'][n] = se_b
        result['papale_reliab_balanced_adapted'][n] = rel_b
        result['papale_oracle_balanced_adapted'][n] = orc_b
        result['papale_balanced_k'][n] = k_b
        result['papale_balanced_n_images'][n] = n_b
        if np.isfinite(rel_b) and rel_b > 0:
            result['papale_oracle_30rep_model_estimate'][n] = float(oracle_30rep_model_estimate(rel_b))
        oracle = []
        for _ in range(splits):
            held, rest = [], []
            for values, _, _ in repeated:
                k = rng.integers(len(values))
                held.append(values[k])
                rest.append((values.sum() - values[k]) / (len(values) - 1))
            oracle.append(_corr(np.asarray(held), np.asarray(rest)))
        result['papale_oracle_adapted'][n] = _nanmean_or_nan(oracle)
        observed_mean = np.array([values.mean() for values, _, _ in repeated])
        predicted_mean = np.array([pred for _, pred, _ in repeated])
        r = _corr(predicted_mean, observed_mean)
        result['raw_repeated_pearson'][n] = r
        halves = []
        bb_halves = []
        for _ in range(splits):
            first, second = [], []
            for values, _, _ in repeated:
                shuffled = rng.permutation(values)
                midpoint = len(values) // 2
                first.append(shuffled[:midpoint].mean())
                second.append(shuffled[midpoint:].mean())
            half_r = _corr(np.asarray(first), np.asarray(second))
            halves.append(2 * half_r / (1 + half_r) if np.isfinite(half_r) and half_r > -1 else np.nan)
            # BBScore's data/utils.py sets undefined half-correlations to 0
            # before Spearman-Brown correction (unlike scipy Pearson).
            bb_r = np.float64(half_r) if np.isfinite(half_r) else np.float64(0)
            with np.errstate(divide='ignore', invalid='ignore'):
                bb_halves.append(float(np.divide(2 * bb_r, 1 + bb_r)))
        result['split_half_sb_by_split'][n] = halves[:n_brainscore_splits]
        reliability = float(np.mean(halves)) if np.isfinite(halves).all() else np.nan
        result['split_half_sb_reliability'][n] = reliability
        bb_reliability = float(np.mean(bb_halves))
        result['bbscore_split_half_sb_reliability'][n] = bb_reliability
        if not np.isnan(bb_reliability):
            ceiling = bbscore_clip_ceiling(bb_reliability)
            result['bbscore_clipped_ceiling'][n] = ceiling
            transformed = bbscore_pearson_scorer(observed_mean, predicted_mean)
            result['bbscore_transformed_pearson'][n] = transformed
            if np.isfinite(transformed):
                result['bbscore_noise_corrected'][n] = transformed / ceiling
        if not (np.isfinite(r) and reliability > 1e-6):
            continue
        result['pearson_noise_corrected'][n] = r / np.sqrt(reliability)
    if reliability_only:
        for key in PREDICTION_KEYS:
            result[key][:] = np.nan
    return result

