"""Pure-CPU checks of the published BBScore ridge and split-half conventions."""
import inspect
import unittest

import numpy as np
from sklearn.linear_model import RidgeCV

from transformer_model.benchmark_probe import BB_SCORE_ALPHAS, fit_masked_bbscore
from transformer_model.benchmark_stats import score_repeated_stimuli, finite_summary


class ZWMPaperProtocolTests(unittest.TestCase):
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
