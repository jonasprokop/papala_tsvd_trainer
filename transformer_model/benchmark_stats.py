"""TVSD split-half and score transformations from BBScore's public backend.

Source: neuroailab/bbscore_public, data/utils.py and metrics/utils.py.
The sparse tracked-neuron cohort requires per-neuron repeat grouping, which is
not present in BBScore's dense original-channel implementation.
"""
from __future__ import annotations

import numpy as np


def _pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2:
        return np.nan
    a = a - a.mean()
    b = b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator > 0 else np.nan


def score_repeated_stimuli(predictions, responses, mask, image_ids, *,
                           splits=50, seed=42, min_images=20):
    """Score only images with >=2 observed trials for each individual neuron.

    Randomly partition each image's repeats into two nonempty halves. Match
    BBScore's convention: Spearman-Brown transform each split correlation and
    average 50 estimates. Its scorer *also* applies 2r/(1+r) to prediction
    Pearson r, then divides by the split-half consistency. Low/no consistency
    is excluded instead of silently assigning a huge normalized score.

    Responses and predictions never enter fitting, layer choice or alpha choice
    here. The held-out image set is identical for the model score and ceiling.
    """
    prediction = np.asarray(predictions, dtype=np.float64)
    response = np.asarray(responses, dtype=np.float64)
    observed = np.asarray(mask, dtype=bool)
    ids = np.asarray(image_ids)
    if (prediction.ndim != 2 or prediction.shape != response.shape or
            prediction.shape != observed.shape or ids.shape != (len(prediction),)):
        raise ValueError('Expected prediction/response/mask [presentations,neurons] and image IDs')
    if splits < 1 or min_images < 2:
        raise ValueError('splits must be positive and min_images >= 2')
    if not np.isfinite(response[observed]).all():
        raise ValueError('Nonfinite observed neural responses')
    image_rows = [np.flatnonzero(ids == item) for item in np.unique(ids)]
    width = prediction.shape[1]
    raw = np.full(width, np.nan)
    scorer_transformed = np.full(width, np.nan)
    reliability = np.full(width, np.nan)
    ceiling = np.full(width, np.nan)
    corrected = np.full(width, np.nan)
    counts = np.zeros(width, dtype=int)
    rng = np.random.default_rng(seed)

    for neuron in range(width):
        groups = []
        model = []
        actual = []
        for rows in image_rows:
            valid = rows[observed[rows, neuron]]
            if len(valid) < 2:
                continue
            p = prediction[valid, neuron]
            if not np.isfinite(p).all():
                continue
            values = response[valid, neuron]
            groups.append(values)
            model.append(float(p.mean()))
            actual.append(float(values.mean()))
        counts[neuron] = len(groups)
        if len(groups) < min_images:
            continue
        raw[neuron] = _pearson(model, actual)
        if np.isfinite(raw[neuron]) and raw[neuron] > -1:
            scorer_transformed[neuron] = 2 * raw[neuron] / (1 + raw[neuron])
        half_reliabilities = []
        for _ in range(splits):
            first = []
            second = []
            for values in groups:
                shuffled = rng.permutation(values)
                middle = len(shuffled) // 2
                first.append(float(shuffled[:middle].mean()))
                second.append(float(shuffled[middle:].mean()))
            half_r = _pearson(first, second)
            if not np.isfinite(half_r) or half_r <= -1:
                half_reliabilities.append(np.nan)
            else:
                half_reliabilities.append(2 * half_r / (1 + half_r))
        half_reliabilities = np.asarray(half_reliabilities)
        if not np.isfinite(half_reliabilities).all():
            continue
        full_reliability = float(np.mean(half_reliabilities))
        reliability[neuron] = full_reliability
        if full_reliability <= 1e-6:
            continue
        ceiling[neuron] = full_reliability
        corrected[neuron] = scorer_transformed[neuron] / full_reliability

    return dict(raw_pearson=raw, bbscore_pearson=scorer_transformed,
                split_half_reliability=reliability, noise_ceiling=ceiling,
                noise_corrected_pearson=corrected,
                n_repeated_images=counts)


def finite_summary(values):
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    return dict(mean=float(valid.mean()) if len(valid) else None,
                median=float(np.median(valid)) if len(valid) else None,
                n_valid=int(len(valid)))


def bbscore_floor_corrected_scores(pearson_transformed, split_half_reliability):
    """Reapply BBScore's 1e-6 ceiling floor to already saved layer scores.

    This matches metrics/base.py's normalization convention, but uses our
    existing split-half estimates; it does not rerun BBScore's random splits.
    """
    scores = np.asarray(pearson_transformed, dtype=np.float64)
    reliability = np.asarray(split_half_reliability, dtype=np.float64)
    if scores.shape != reliability.shape:
        raise ValueError('Transformed Pearson and reliability shapes differ')
    with np.errstate(divide='ignore', invalid='ignore'):
        return scores / np.maximum(reliability, 1e-6)
