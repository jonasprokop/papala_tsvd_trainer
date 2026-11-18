"""Tiny CPU-only numerical equivalence check for ridge solvers.

Run on a development machine: python -m transformer_model.ridge_smoke
No checkpoint, network, PBS allocation, or TVSD data is loaded.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np
import torch

from .ridge import fit_masked_ridge_cv, fit_masked_ridge_cv_reference


def main():
    rng=np.random.default_rng(2026)
    x=rng.normal(size=(48,8))
    y=x @ rng.normal(size=(8,6))+rng.normal(scale=0.05,size=(48,6))
    observed=np.ones_like(y,dtype=bool)
    observed[::5,:2]=False
    observed[::7,2:4]=False
    observed[:,5]=False
    observed[:2,5]=True
    args=dict(train_indices=np.arange(36),groups=np.repeat(np.arange(24),2),
              alphas=(.01,.1,1.,10.),cv_folds=3)
    start=perf_counter()
    slow=fit_masked_ridge_cv_reference(x,y,observed,**args)
    slow_s=perf_counter()-start
    start=perf_counter()
    fast=fit_masked_ridge_cv(x,y,observed,**args)
    fast_s=perf_counter()-start
    np.testing.assert_array_equal(fast.valid_neuron_mask,slow.valid_neuron_mask)
    np.testing.assert_allclose(fast.alphas,slow.alphas,equal_nan=True)
    np.testing.assert_allclose(fast.cv_scores,slow.cv_scores,equal_nan=True,
                               rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(fast.readout.weight,slow.readout.weight,
                               rtol=1e-4,atol=1e-5)
    valid=fast.valid_neuron_mask
    torch.testing.assert_close(fast.readout.bias[valid],slow.readout.bias[valid],
                               rtol=1e-4,atol=1e-5)
    print('PASS: fast and sklearn_reference ridge agree on grouped repeated images')
    print(f'CPU synthetic timings: spectral={fast_s:.3f}s sklearn_reference={slow_s:.3f}s')
    print('These tiny timings are not a production-scale performance benchmark.')


if __name__ == '__main__':
    main()
