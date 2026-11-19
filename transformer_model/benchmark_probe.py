"""BBScore-style RidgeCV adapted to TVSD's masked tracked-neuron response axis.

Alpha selection uses sklearn RidgeCV(alpha_per_target=True), without feature
standardization, exactly as public BBScore RidgeMetric. The paper's 10 outer
folds are computed with stimulus-group isolation to prevent repeat leakage.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from .benchmark_stats import _pearson


BB_SCORE_ALPHAS = tuple(10.0 ** power for power in range(-10, 11))


def fit_masked_bbscore(features, responses, mask, stimulus_ids, *,
                        alphas=BB_SCORE_ALPHAS, n_folds=10, seed=42,
                        persist_group=None):
    """Masked F/N neurons, BBScore ridge, and group-safe outer fold estimates.

    ``persist_group(index, neurons, values)`` may return stored values on retry;
    otherwise it receives ``values=None`` and saves a newly fitted group.
    Each group's fit is independent and can resume after a PBS walltime kill.
    """
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(responses, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    ids = np.asarray(stimulus_ids)
    if x.ndim != 2 or y.ndim != 2 or y.shape != m.shape or len(x) != len(y) or ids.shape != (len(x),):
        raise ValueError('Incompatible ridge features, responses, mask or stimulus IDs')
    if not np.isfinite(x).all() or not np.isfinite(y[m]).all():
        raise ValueError('Nonfinite observed training data')
    n_neurons, width = y.shape[1], x.shape[1]
    weights = np.zeros((n_neurons,width), dtype=np.float32)
    bias = np.full(n_neurons, np.nan, dtype=np.float32)
    selected = np.full(n_neurons, np.nan, dtype=np.float64)
    cv_r = np.full(n_neurons, np.nan, dtype=np.float64)
    counts = m.sum(axis=0)
    by_mask = {}
    for neuron in range(n_neurons):
        by_mask.setdefault(m[:,neuron].tobytes(),[]).append(neuron)
    print(f'BBScore masked groups: {len(by_mask)} across {n_neurons} F+N neurons', flush=True)
    for group_number, neuron_list in enumerate(by_mask.values()):
        cols = np.asarray(neuron_list, dtype=int)
        rows = np.flatnonzero(m[:,cols[0]])
        unique_groups = np.unique(ids[rows])
        if len(rows) < 3 or len(unique_groups) < 2:
            continue
        cached = persist_group(group_number, cols, None) if persist_group else None
        if cached is None:
            targets = y[np.ix_(rows,cols)]
            # Full-data BBScore fit: RidgeCV uses efficient GCV to select the
            # published 21 alpha strengths separately for each target neuron.
            ridge = RidgeCV(alphas=alphas, alpha_per_target=True)
            ridge.fit(x[rows], targets)
            alpha = np.atleast_1d(np.asarray(ridge.alpha_, dtype=float))
            coef = np.atleast_2d(ridge.coef_).astype(np.float32)
            intercept = np.atleast_1d(ridge.intercept_).astype(np.float32)
            if alpha.shape != (len(cols),) or coef.shape != (len(cols),width):
                raise RuntimeError('Unexpected RidgeCV per-neuron coefficients')
            oof = np.full_like(targets,np.nan)
            # Outer 10-fold evaluation, consistent with the paper. Group by
            # unique stimulus so repeated presentations never cross folds.
            splitter = KFold(n_splits=min(n_folds,len(unique_groups)),
                             shuffle=True,random_state=seed)
            for train_group_idx,test_group_idx in splitter.split(unique_groups):
                train_row = np.isin(ids[rows],unique_groups[train_group_idx])
                test_row = np.isin(ids[rows],unique_groups[test_group_idx])
                if not train_row.any() or not test_row.any():
                    continue
                fold = RidgeCV(alphas=alphas,alpha_per_target=True)
                fold.fit(x[rows[train_row]],targets[train_row])
                oof[test_row] = np.asarray(fold.predict(x[rows[test_row]])).reshape(-1,len(cols))
            corr = np.asarray([_pearson(oof[:,j],targets[:,j])
                               for j in range(len(cols))])
            cached = dict(weight=coef,bias=intercept,alpha=alpha,cv_r=corr)
            if persist_group:
                persist_group(group_number,cols,cached)
        weights[cols] = cached['weight']
        bias[cols] = cached['bias']
        selected[cols] = cached['alpha']
        cv_r[cols] = cached['cv_r']
        if (group_number+1) % 25 == 0:
            print(f'RidgeCV completed/restored {group_number+1}/{len(by_mask)} groups',flush=True)
    return dict(weight=weights,bias=bias,alpha=selected,
                cv_pearson=cv_r,valid_train_mask=np.isfinite(bias),
                valid_train_counts=counts)
