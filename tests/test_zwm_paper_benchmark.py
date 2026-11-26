"""Pure-CPU checks of the published BBScore ridge and split-half conventions."""
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from transformer_model import benchmark_v1 as benchmark
from transformer_model.benchmark_probe import BB_SCORE_ALPHAS, fit_masked_bbscore
from transformer_model.benchmark_stats import score_repeated_stimuli, finite_summary


class ZWMPaperProtocolTests(unittest.TestCase):
    @staticmethod
    def _shared_bank_fixture(tmp):
        root = Path(tmp)
        source = root / 'ks4_results'
        source_data = root / 'ks4_data'
        target_data = root / 'mua_data'
        image_root = root / 'images'
        source_data.mkdir()
        target_data.mkdir()
        image_root.mkdir()
        index = {'test:1': 'one.jpg', 'train:2': 'two.jpg'}
        pd.DataFrame({'stimulus_id': list(index),
                      'image_relative_path': list(index.values())}).to_csv(
                          source_data/'sample_index.csv', index=False)
        (source_data/'dataset_info.json').write_text(json.dumps({'split_sha256': 'same'}))
        source_cfg = {'data': {'root': str(source_data), 'image_root': str(image_root)},
                      'model': {'backbone': {'kind': 'zwm', 'model_ref': 'test.pt',
                                             'pooling': 'mean', 'input_mode': 'rgb',
                                             'cache_chunk_size': 256,
                                             'forward_mode': 'one_image'}}}
        target_cfg = json.loads(json.dumps(source_cfg))
        target_cfg['data']['root'] = str(target_data)
        feature_spec = {'dataset_fingerprint': 'mua-fingerprint',
                        'weights_sha256': 'weights',
                        'predictor_sha256': 'predictor',
                        'images': {'test:1': ['one.jpg', 12, 20],
                                   'train:2': ['two.jpg', 12, 20]}}
        fingerprint = hashlib.sha256(json.dumps(
            dict(feature_spec, dataset_fingerprint='ks4-fingerprint'),
            sort_keys=True).encode()).hexdigest()
        protocol = {'source_config': source_cfg,
                    'source_weights_sha256': 'weights',
                    'source_predictor_sha256': 'predictor',
                    'layers': list(benchmark.LAYERS),
                    'dataset_fingerprint': 'ks4-fingerprint',
                    'cache_fingerprint': fingerprint}
        source.mkdir()
        (source/'protocol.json').write_text(json.dumps(protocol))
        bank = source/'features'/fingerprint
        bank.mkdir(parents=True)
        chunk = bank/'0000000.npz'
        np.savez_compressed(chunk, stimulus_ids=list(sorted(index)),
                            features=np.ones((2, len(benchmark.LAYERS), 3)))
        return source, root/'mua_results', target_cfg, index, chunk, feature_spec

    def test_shared_feature_bank_is_read_only_and_validates_both_cohorts(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target, cfg, index, chunk, spec = self._shared_bank_fixture(tmp)
            bundle = SimpleNamespace(info={'split_sha256': 'same'})
            bank, provenance = benchmark._shared_feature_bank(
                source, target, cfg, bundle, index, feature_spec=spec)
            self.assertEqual(provenance['source_cache_fingerprint'], bank.name)
            before = chunk.stat().st_mtime_ns
            chunks = list(benchmark._chunks(target, index, 'different-mua-key', cfg,
                                            extract=False, shared_bank=bank))
            self.assertEqual(chunks[0][0], sorted(index))
            self.assertEqual(chunks[0][1].shape, (2, len(benchmark.LAYERS), 3))
            self.assertEqual(chunk.stat().st_mtime_ns, before)
            self.assertFalse(target.exists())
            with self.assertRaisesRegex(ValueError, 'read-only'):
                list(benchmark._chunks(target, index, 'different-mua-key', cfg,
                                       extract=True, shared_bank=bank))

    def test_shared_features_reject_stale_data_model_or_missing_chunk(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target, cfg, index, chunk, spec = self._shared_bank_fixture(tmp)
            bundle = SimpleNamespace(info={'split_sha256': 'same'})

            def validate(*, given_spec=spec, mapping=index):
                return benchmark._shared_feature_bank(
                    source, target, cfg, bundle, mapping, feature_spec=given_spec)

            with self.assertRaisesRegex(ValueError, 'checkpoint'):
                validate(given_spec=dict(spec, weights_sha256='different'))
            with self.assertRaisesRegex(ValueError, 'mappings'):
                validate(mapping={'test:1': 'wrong.jpg', 'train:2': 'two.jpg'})
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                validate(given_spec=dict(spec, images={'test:1': ['one.jpg', 12, 21]}))
            bundle.info['split_sha256'] = 'changed'
            with self.assertRaisesRegex(ValueError, 'splits differ'):
                validate()
            bundle.info['split_sha256'] = 'same'
            chunk.unlink()
            with self.assertRaisesRegex(FileNotFoundError, 'Incomplete'):
                validate()

    def test_per_layer_predictions_are_saved_and_replay_without_refit(self):
        features = dict(zip(('train:1', 'train:2', 'test:1', 'test:2'),
                            (np.array([1., 0.]), np.array([0., 1.]),
                             np.array([2., 3.]), np.array([4., 5.]))))
        train = SimpleNamespace(
            rows=pd.DataFrame({'stimulus_id': ['train:1', 'train:2']}),
            responses=np.zeros((2, 2)), mask=np.ones((2, 2), dtype=bool))
        test = SimpleNamespace(
            rows=pd.DataFrame({'stimulus_id': ['test:1', 'test:2', 'test:1', 'test:2'],
                               'animal': ['F', 'F', 'N', 'N'],
                               'session': ['macaqueF', 'macaqueF', 'macaqueN', 'macaqueN']}),
            responses=np.array([[.2, 0], [.3, 0], [0, -.4], [0, -.5]]),
            mask=np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=bool),
            image_ids=np.array([1, 2, 1, 2]))
        bundle = SimpleNamespace(full=train, test=test)
        fit = dict(weight=np.array([[1., 0.], [0., 2.]]), bias=np.array([0., -1.]),
                   alpha=np.ones(2), cv_pearson=np.array([.1, .2]))
        scoring = dict(noise_corrected_pearson=np.array([.4, .5]))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with patch.object(benchmark, '_load_layer', return_value=features), \
                 patch.object(benchmark, '_neuron_order', return_value=['F/0', 'N/0']), \
                 patch.object(benchmark, 'fit_masked_bbscore', return_value=fit) as fitter, \
                 patch.object(benchmark, 'score_repeated_stimuli', return_value=scoring):
                benchmark._fit_layer({}, bundle, out, {}, 'fingerprint', 0)
                self.assertEqual(fitter.call_count, 1)
                path = out/'predictions'/'00.npz'
                with np.load(path, allow_pickle=False) as saved:
                    self.assertEqual(set(('prediction', 'response', 'mask', 'image_id',
                                          'neuron_ids', 'groups', 'session')) - set(saved.files), set())
                    np.testing.assert_array_equal(saved['prediction'],
                                                  [[2., 5.], [4., 9.], [2., 5.], [4., 9.]])
                    np.testing.assert_array_equal(saved['response'], test.responses)
                    np.testing.assert_array_equal(saved['mask'], test.mask)
                    np.testing.assert_array_equal(saved['neuron_ids'], ['F/0', 'N/0'])
                    np.testing.assert_array_equal(saved['groups'], ['F', 'N'])
                    np.testing.assert_array_equal(saved['session'], test.rows.session)
                path.unlink()  # Simulate an earlier completed layer without saved predictions.
                benchmark._fit_layer({}, bundle, out, {}, 'fingerprint', 0)
                self.assertEqual(fitter.call_count, 1)  # Replay, never refit.
                self.assertTrue(path.exists())

    @unittest.skipUnless('alpha_per_target' in inspect.signature(RidgeCV).parameters,
                         'BBScore RidgeCV requires the ZWM Python 3.10 environment')
    def test_complete_cohort_matches_unmodified_bbscore_ridgecv(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(30,4))
        y = x @ rng.normal(size=(4,3)) + rng.normal(scale=.2,size=(30,3))
        expected = RidgeCV(alphas=BB_SCORE_ALPHAS,alpha_per_target=True).fit(x,y)
        actual = fit_masked_bbscore(x,y,np.ones_like(y,dtype=bool),
                                    np.arange(len(x)),n_folds=3)
        np.testing.assert_allclose(actual['weight'],expected.coef_,rtol=1e-6,atol=1e-6)
        np.testing.assert_allclose(actual['bias'],expected.intercept_,rtol=1e-6,atol=1e-6)
        np.testing.assert_array_equal(actual['alpha'],expected.alpha_)

    @unittest.skipUnless('alpha_per_target' in inspect.signature(RidgeCV).parameters,
                         'BBScore RidgeCV requires the ZWM Python 3.10 environment')
    def test_published_ridge_grid_and_masked_fold_fit(self):
        self.assertEqual(len(BB_SCORE_ALPHAS), 21)
        self.assertEqual(BB_SCORE_ALPHAS[0], 1e-10)
        self.assertEqual(BB_SCORE_ALPHAS[-1], 1e10)
        rng = np.random.default_rng(42)
        image = np.repeat(np.arange(12), 2)
        x = rng.normal(size=(len(image), 3))
        x[1::2] = x[::2]  # repeated presentations share exact model features
        response = np.column_stack([2*x[:,0] - x[:,1], x[:,0] + 3*x[:,2]])
        response += rng.normal(scale=.03,size=response.shape)
        mask = np.zeros_like(response, dtype=bool)
        mask[:16,0] = True  # monkey F
        mask[8:,1] = True   # monkey N, different observation coverage
        saved = {}

        def persist(group, neurons, result):
            if result is None:
                return saved.get(group)
            saved[group] = result

        fit = fit_masked_bbscore(x,response,mask,image,n_folds=3,
                                 persist_group=persist)
        self.assertEqual(fit['weight'].shape,(2,3))
        self.assertTrue(np.isfinite(fit['cv_pearson']).all())
        self.assertTrue(np.all(fit['cv_pearson'] > .5))
        self.assertTrue(all(value in BB_SCORE_ALPHAS for value in fit['alpha']))
        self.assertEqual(len(saved),2)
        cached = fit_masked_bbscore(x,response,mask,image,n_folds=3,
                                    persist_group=persist)
        np.testing.assert_array_equal(fit['weight'],cached['weight'])

    def test_reliability_and_prediction_use_identical_repeated_stimuli(self):
        rng = np.random.default_rng(123)
        stimulus = np.repeat(np.arange(40),4)
        tuning = rng.normal(size=40)
        neural = tuning[stimulus] + rng.normal(scale=.05,size=len(stimulus))
        prediction = np.column_stack([tuning[stimulus],tuning[stimulus]])
        responses = np.column_stack([neural,neural])
        mask = np.ones_like(responses,dtype=bool)
        mask[:,1] = stimulus < 5  # 5 repeated images, below coverage threshold
        result = score_repeated_stimuli(prediction,responses,mask,stimulus,
                                        splits=50,seed=42,min_images=20)
        self.assertEqual(result['n_repeated_images'].tolist(),[40,5])
        self.assertGreater(result['raw_pearson'][0],.99)
        self.assertGreater(result['split_half_reliability'][0],.99)
        self.assertTrue(np.isnan(result['noise_corrected_pearson'][1]))
        self.assertEqual(finite_summary(result['noise_corrected_pearson'])['n_valid'],1)

    def test_single_repeats_do_not_create_artificial_ceiling(self):
        ids = np.arange(50)
        y = np.arange(50,dtype=float)[:,None]
        result = score_repeated_stimuli(y,y,np.ones_like(y,dtype=bool),ids)
        self.assertEqual(result['n_repeated_images'].tolist(),[0])
        self.assertTrue(np.isnan(result['noise_corrected_pearson'][0]))


if __name__ == '__main__':
    unittest.main()
