"""Check our reliability definitions against TVSD's released `reliab` and `oracle`.

Usage: python -m evaluation.verify_papale_reliab --mat monkeyN/THINGS_normMUA.mat

Definitions (Papale et al. 2025, Neuron, STAR Methods "Reliability estimations"):
  mean reliability = average pairwise r between each combination of repetitions
                     to the 100 test images; README: `reliab` holds the per-
                     combination values, the paper's number is their mean.
  oracle           = r between a single presentation and the mean of the other
                     29, repeated for each of the 30 repetitions and averaged.

Checks, per electrode, on `test_MUA_reps`:
  1. mean of our all-pairs r      vs  mean of released `reliab`
  2. per-pair r in nchoosek order vs  released `reliab` columns (only if 435 cols)
  3. our leave-one-out oracle     vs  released `oracle`
A pass validates the DEFINITIONS and the data layout. It says nothing about
sorted units, whose preprocessing differs.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations

import h5py
import numpy as np

from .evaluation_protocols import _corr, papale_oracle_dense, papale_reliab_dense


def _electrode_first(arr, n_elec):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        return arr
    return arr if arr.shape[0] == n_elec else arr.T


def _report(ours, published):
    ok = np.isfinite(ours) & np.isfinite(published)
    diff = ours[ok] - published[ok]
    return dict(n=int(ok.sum()), max_abs_diff=float(np.max(np.abs(diff))),
                mean_diff=float(np.mean(diff)),
                corr=float(np.corrcoef(ours[ok], published[ok])[0, 1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mat', required=True)
    ap.add_argument('--electrodes', type=int, default=64, help='first N electrodes (speed)')
    args = ap.parse_args()
    with h5py.File(args.mat, 'r') as f:
        reps = np.asarray(f['test_MUA_reps'], dtype=np.float64)   # h5py reverses MATLAB order
        reliab = np.asarray(f['reliab'])
        oracle = np.asarray(f['oracle'])
    if reps.ndim != 3 or reps.shape[1] != 100:
        raise ValueError(f'Unexpected test_MUA_reps shape {reps.shape}; expected [reps, 100, electrodes]')
    n_rep, _, n_elec = reps.shape
    reliab = _electrode_first(reliab, n_elec)
    oracle = _electrode_first(oracle, n_elec).reshape(n_elec, -1)[:, 0]
    use = range(min(args.electrodes, n_elec))
    out = dict(shapes=dict(test_MUA_reps=list(reps.shape), reliab=list(reliab.shape)))
    ours_rel = np.array([papale_reliab_dense(reps[:, :, e]) for e in use])
    out['mean_reliability'] = _report(ours_rel, np.nanmean(reliab.reshape(n_elec, -1), axis=1)[:len(ours_rel)])
    pairs = list(combinations(range(n_rep), 2))
    if reliab.ndim == 2 and reliab.shape[1] == len(pairs):
        e = 0
        per_pair = np.array([_corr(reps[a, :, e], reps[b, :, e]) for a, b in pairs])
        out['per_pair_electrode0'] = _report(per_pair, reliab[e])
    ours_or = np.array([papale_oracle_dense(reps[:, :, e]) for e in use])
    out['oracle'] = _report(ours_or, oracle[:len(ours_or)])
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
