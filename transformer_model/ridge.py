"""Masked per-neuron ridge with strictly train-fold-fitted feature scalers.

The frozen TVSD split determines evaluation. GroupKFold is used ONLY within its
training partition to select regularization. No validation/test responses used.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold


@dataclass
class FittedRidge:
    readout: 'RidgeReadout'
    alphas: np.ndarray
    cv_scores: np.ndarray
    valid_train_counts: np.ndarray
    valid_neuron_mask: np.ndarray


class RidgeReadout(nn.Module):
    """Frozen neuron-specific linear map, with exportable numeric buffers."""
    def __init__(self, weight, bias, feature_mean, feature_scale):
        super().__init__()
        w = torch.as_tensor(np.asarray(weight).copy(), dtype=torch.float32)
        b = torch.as_tensor(np.asarray(bias).copy(), dtype=torch.float32)
        mu = torch.as_tensor(np.asarray(feature_mean).copy(), dtype=torch.float32)
        sd = torch.as_tensor(np.asarray(feature_scale).copy(), dtype=torch.float32)
        if w.ndim != 2 or b.shape != (w.shape[0],):
            raise ValueError('weight must be [neurons,features], bias [neurons]')
        if mu.shape != (w.shape[1],) or sd.shape != mu.shape or (sd <= 0).any():
            raise ValueError('Invalid feature scaler')
        for name, value in [('weight',w),('bias',b),('feature_mean',mu),('feature_scale',sd)]:
            self.register_buffer(name, value)

    def forward(self, x):
        return nn.functional.linear((x.float() - self.feature_mean) / self.feature_scale,
                                    self.weight, self.bias)


def _scale_fit(x):
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd < 1e-12] = 1.
    return mu, sd


def fit_masked_ridge_cv_reference(features, responses, observation_mask, *, train_indices,
                                  groups, alphas=(0.01, 0.1, 1., 10., 100.), cv_folds=5):
    """Reference sklearn implementation for numerical regression tests.

    Fitted coefficients are transformed back to raw-feature space. This lets a
    single PyTorch linear readout represent independently standardized neurons.
    """
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(responses, dtype=np.float64)
    m = np.asarray(observation_mask, dtype=bool)
    group = np.asarray(groups)
    tr = np.asarray(train_indices, dtype=int)
    grid = np.asarray(alphas, dtype=np.float64)
    if (x.ndim != 2 or y.ndim != 2 or y.shape != m.shape or
            len(x) != len(y) or group.shape != (len(x),)):
        raise ValueError('X, Y, mask or groups have incompatible shapes')
    if not len(tr) or len(np.unique(tr)) != len(tr) or (tr < 0).any() or (tr >= len(x)).any():
        raise ValueError('train_indices must be unique valid row indices')
    if not np.isfinite(x[tr]).all() or not np.isfinite(y[tr][m[tr]]).all():
        raise ValueError('Nonfinite observed training features/responses')
    if not len(grid) or not np.isfinite(grid).all() or (grid <= 0).any():
        raise ValueError('All ridge alpha values must be positive and finite')
    n, d = y.shape[1], x.shape[1]
    w = np.zeros((n,d), dtype=np.float64)
    b = np.full(n, np.nan, dtype=np.float64)
    selected = np.full(n, np.nan, dtype=np.float64)
    scores = np.full(n, np.nan, dtype=np.float64)
    counts = np.zeros(n, dtype=int)
    valid = np.zeros(n, dtype=bool)
    for j in range(n):
        rows = tr[m[tr,j]]
        counts[j] = len(rows)
        k = min(int(cv_folds), len(np.unique(group[rows])))
        if len(rows) < 3 or k < 2:
            continue
        folds = list(GroupKFold(n_splits=k).split(x[rows], y[rows,j], groups=group[rows]))
        losses = []
        for alpha in grid:
            errors = []
            for a, v in folds:
                xa, xv = x[rows[a]], x[rows[v]]
                mu, sd = _scale_fit(xa)  # Critical: held-out fold must not set scale
                reg = Ridge(alpha=float(alpha), fit_intercept=True)
                reg.fit((xa-mu)/sd, y[rows[a],j])
                pred = reg.predict((xv-mu)/sd)
                errors.append(float(np.mean((pred-y[rows[v],j])**2)))
            losses.append(np.mean(errors))
        best = int(np.argmin(losses))
        mu, sd = _scale_fit(x[rows])
        reg = Ridge(alpha=float(grid[best]), fit_intercept=True)
        reg.fit((x[rows]-mu)/sd, y[rows,j])
        w[j] = np.asarray(reg.coef_) / sd
        b[j] = float(reg.intercept_) - float(w[j] @ mu)
        selected[j] = grid[best]
        scores[j] = -float(losses[best])
        valid[j] = True
    # Each neuron's scaling is folded exactly into its raw-space coefficients.
    readout = RidgeReadout(w,b,np.zeros(d),np.ones(d))
    return FittedRidge(readout, selected, scores, counts, valid)


def _spectral_ridge(x, y):
    """One covariance eigendecomposition; solve arbitrary per-output alphas.

    Centering implements an unpenalized intercept, matching sklearn Ridge.
    The dual formulation avoids an expensive d-by-d decomposition when d > n.
    """
    center_x=x.mean(axis=0)
    center_y=y.mean(axis=0)
    centered_x=x-center_x
    centered_y=y-center_y
    if x.shape[1] <= x.shape[0]:
        gram=centered_x.T @ centered_x
        eig, basis=np.linalg.eigh(gram)
        eig=np.maximum(eig,0.)
        projected=basis.T @ (centered_x.T @ centered_y)

        def coefficients(alpha):
            return basis @ (projected / (eig[:,None] + np.asarray(alpha)[None,:]))
    else:
        gram=centered_x @ centered_x.T
        eig, basis=np.linalg.eigh(gram)
        eig=np.maximum(eig,0.)
        projected=basis.T @ centered_y

        def coefficients(alpha):
            fitted=basis @ (projected / (eig[:,None] + np.asarray(alpha)[None,:]))
            return centered_x.T @ fitted
    return coefficients, center_x, center_y


def fit_masked_ridge_cv(features, responses, observation_mask, *, train_indices,
                        groups, alphas=(0.01, 0.1, 1., 10., 100.), cv_folds=5):
    """Batched masked ridge with exact stimulus groups and isolated CV scalers.

    Neurons sharing their *full* training observation mask share the scaler,
    folds and one eigensolve per fold. Each neuron still selects its own alpha.
    """
    x=np.asarray(features,dtype=np.float64)
    y=np.asarray(responses,dtype=np.float64)
    mask=np.asarray(observation_mask,dtype=bool)
    groups=np.asarray(groups)
    tr=np.asarray(train_indices,dtype=int)
    grid=np.asarray(alphas,dtype=np.float64)
    if (x.ndim!=2 or y.ndim!=2 or y.shape!=mask.shape or
            len(x)!=len(y) or groups.shape!=(len(x),)):
        raise ValueError('X, Y, mask or groups have incompatible shapes')
    if not len(tr) or len(np.unique(tr))!=len(tr) or (tr<0).any() or (tr>=len(x)).any():
        raise ValueError('train_indices must be unique valid row indices')
    if not np.isfinite(x[tr]).all() or not np.isfinite(y[tr][mask[tr]]).all():
        raise ValueError('Nonfinite observed training features/responses')
    if not len(grid) or not np.isfinite(grid).all() or (grid<=0).any():
        raise ValueError('All ridge alpha values must be positive and finite')
    n,d=y.shape[1],x.shape[1]
    weights=np.zeros((n,d),dtype=np.float64)
    bias=np.full(n,np.nan,dtype=np.float64)
    selected=np.full(n,np.nan,dtype=np.float64)
    scores=np.full(n,np.nan,dtype=np.float64)
    counts=mask[tr].sum(axis=0).astype(int)
    valid=np.zeros(n,dtype=bool)
    by_mask={}
    for j in range(n):
        by_mask.setdefault(mask[tr,j].tobytes(),[]).append(j)
    print(f'Ridge mask groups: {len(by_mask)} for {n} neurons; '
          f'largest group: {max(map(len, by_mask.values()), default=0)}', flush=True)
    for indices in by_mask.values():
        cols=np.asarray(indices,dtype=int)
        rows=tr[mask[tr,cols[0]]]
        k=min(int(cv_folds),len(np.unique(groups[rows])))
        if len(rows)<3 or k<2:
            continue
        target=y[np.ix_(rows,cols)]
        folds=GroupKFold(n_splits=k).split(x[rows],target,groups=groups[rows])
        losses=np.zeros((len(grid),len(cols)),dtype=np.float64)
        for a,v in folds:
            xa,xv=x[rows[a]],x[rows[v]]
            mu,sd=_scale_fit(xa)
            normalized=(xa-mu)/sd
            solve,xmean,ymean=_spectral_ridge(normalized,target[a])
            val=(xv-mu)/sd-xmean
            for i,alpha in enumerate(grid):
                coef=solve(np.full(len(cols),alpha))
                prediction=val @ coef+ymean
                losses[i]+=np.mean((prediction-target[v])**2,axis=0)/k
        best=np.argmin(losses,axis=0)
        alphas_selected=grid[best]
        mu,sd=_scale_fit(x[rows])
        solve,xmean,ymean=_spectral_ridge((x[rows]-mu)/sd,target)
        coef=solve(alphas_selected)
        raw_coef=coef/sd[:,None]
        weights[cols]=raw_coef.T
        bias[cols]=ymean-(mu/sd+xmean) @ coef
        selected[cols]=alphas_selected
        scores[cols]=-losses[best,np.arange(len(cols))]
        valid[cols]=True
    readout=RidgeReadout(weights,bias,np.zeros(d),np.ones(d))
    return FittedRidge(readout,selected,scores,counts,valid)
