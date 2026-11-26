"""Mathematical and source-integrity checks for the no-reduction GPU probe."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV

from transformer_model.benchmark_spatial_exact import (
    _loocv_select_numpy, _source_function, _atomic_npy, _score_original,
    _chunk_spec, _read_chunk)


class ExactSpatialRidgeTests(unittest.TestCase):
    def test_full_precision_chunk_order_and_no_float16_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            protocol=dict(train_ids=['train:1','train:2','train:3','train:4'],
                          test_ids=['test:1','test:2'],chunk_size=4)
            self.assertEqual(list(_chunk_spec(protocol)),[
                (0,protocol['train_ids']),(4,protocol['test_ids'])])
            path=Path(temp)/'features.npz'
            x=np.random.default_rng(0).normal(size=(4,7)).astype(np.float32)
            np.savez_compressed(path,stimulus_ids=np.asarray(protocol['train_ids']),
                                features=x)
            np.testing.assert_array_equal(_read_chunk(path,protocol['train_ids'],7),x)
            np.savez_compressed(path,stimulus_ids=np.asarray(protocol['train_ids']),
                                features=x.astype(np.float16))
            with self.assertRaisesRegex(ValueError,'Corrupt'):
                _read_chunk(path,protocol['train_ids'],7)

    def test_per_channel_gcv_loo_is_identical_to_sklearn_ridgecv(self):
        for seed, dimensions in ((11, 48), (12, 8)):
            rng = np.random.default_rng(seed)
            x = rng.standard_normal((32, dimensions))
            xt = rng.standard_normal((7, dimensions))
            y = x @ rng.standard_normal((dimensions, 4)) + rng.normal(
                scale=1.7, size=(32, 4))
            alphas = np.logspace(-3, 4, 8)
            center = x.mean(axis=0)
            kernel = (x-center) @ (x-center).T
            values, vectors = np.linalg.eigh(kernel)
            chosen = _loocv_select_numpy(values, vectors, y, alphas)
            expected = RidgeCV(alphas=alphas, alpha_per_target=True,
                               store_cv_results=True)
            expected.fit(x, y)
            np.testing.assert_allclose(chosen, expected.alpha_)
            ym = y.mean(axis=0)
            uy = vectors.T @ (y-ym)
            dual = vectors @ (uy/(np.maximum(values,0)[:,None]+chosen[None,:]))
            predicted = ((xt-center) @ (x-center).T) @ dual + ym
            np.testing.assert_allclose(predicted, expected.predict(xt),
                                       atol=1e-8, rtol=1e-8)

    def test_original_function_is_loaded_without_executing_module_imports(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)/'original.py'
            source.write_text('import a_nonexistent_module\n'
                              'def pearson_correlation_scorer(a, b):\n'
                              '    return float(np.sum(a*b))\n')
            fn = _source_function(source, 'pearson_correlation_scorer')
            self.assertEqual(fn(np.array([1.,2.]), np.array([3.,4.])),11.)

    def test_atomic_gram_file_preserves_double_precision(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'exact/gram_train.npy'
            expected = np.arange(12,dtype=np.float64).reshape(3,4)
            _atomic_npy(path,expected)
            np.testing.assert_array_equal(np.load(path),expected)
            self.assertEqual(np.load(path).dtype,np.float64)

    def test_original_ceiling_floors_negative_reliability_instead_of_excluding(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)
            (source/'half.py').write_text(
                'def split_half_consistency(data, n_splits=50, rng=None):\n'
                '    return np.full(data.shape[-1], -0.3)\n')
            (source/'scorer.py').write_text(
                'def pearson_correlation_scorer(y_true, y_pred):\n'
                '    return 0.2\n')
            response = np.broadcast_to(np.arange(100,dtype=np.float64)[:,None],
                                       (100,1024)).copy()
            rep = np.broadcast_to(response,(30,100,1024)).copy()
            scores = _score_original({},pred=response,
                                     targets=response,reps=rep,
                                     source_paths=dict(split_half=source/'half.py',
                                                       scorer=source/'scorer.py'))
            self.assertEqual(scores['noise_corrected_pearson'].shape,(1024,))
            self.assertTrue(np.all(scores['noise_ceiling']==1.e-6))
            np.testing.assert_allclose(scores['noise_corrected_pearson'],200000.)


if __name__=='__main__':
    unittest.main()
