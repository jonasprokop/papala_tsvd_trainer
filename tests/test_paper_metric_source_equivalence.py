"""Small regression cases transcribed from the original scoring implementations."""

import unittest

import numpy as np

from evaluation.compare_evaluations import brainscore_split_ceiling, summarize_protocols
from evaluation.evaluation_protocols import (bbscore_clip_ceiling,
                                            bbscore_pearson_scorer, score_protocols)


class OriginalBrainScoreAggregationTests(unittest.TestCase):
    def test_mean_of_split_medians_not_median_of_neuron_means(self):
        # Brain-Score neural_common.py uses np.mean(np.median(raw_ceilings_slice, axis=1))
        # with its source array oriented [splits, neurons].
        original_splits_by_neurons = np.array([[.1, .1, .9], [.9, .1, .1]])
        exact = np.mean(np.median(original_splits_by_neurons, axis=1))
        approximate = np.median(np.mean(original_splits_by_neurons, axis=0))
        self.assertAlmostEqual(brainscore_split_ceiling(original_splits_by_neurons.T), exact)
        self.assertAlmostEqual(exact, .1)
        self.assertAlmostEqual(approximate, .5)


class OriginalBBScoreConventionsTests(unittest.TestCase):
    def test_clips_zero_and_negative_ceilings_to_positive_floor(self):
        # BBScore metrics/base.py BaseMetric.__init__ clamps <= 1e-6.
        for ceiling in (-.2, 0.0, 1e-7, 1e-6):
            self.assertEqual(bbscore_clip_ceiling(ceiling), 1e-6)
        self.assertEqual(bbscore_clip_ceiling(.75), .75)

    def test_model_pearson_transform(self):
        y = np.linspace(-1, 1, 50)
        p = .5 * y + np.sin(y * 3)
        r = np.corrcoef(y, p)[0, 1]
        self.assertAlmostEqual(bbscore_pearson_scorer(y, p), 2 * r / (1 + r))

    def test_constant_response_convention(self):
        # BBScore metrics/utils.py returns 1 for matching constants, 0 otherwise.
        y = np.full(25, 3.0)
        self.assertEqual(bbscore_pearson_scorer(y, y.copy()), 1.0)
        self.assertEqual(bbscore_pearson_scorer(y, np.full(25, 4.0)), 0.0)
        self.assertTrue(np.isnan(bbscore_pearson_scorer(y[:1], y[:1])))

    def test_constant_neuron_remains_in_bbscore_with_floored_ceiling(self):
        # Source BBScore: constant matching test signals get transformed r=1;
        # split-half constant signals get r=0, then ceiling floors to 1e-6.
        img = np.repeat(np.arange(30), 4)
        y = np.full((len(img), 1), 3.0)
        scores = score_protocols(y.copy(), y, np.ones_like(y, bool), img, splits=2)
        self.assertTrue(np.isnan(scores['raw_repeated_pearson'][0]))
        self.assertEqual(scores['bbscore_transformed_pearson'][0], 1.0)
        self.assertEqual(scores['bbscore_clipped_ceiling'][0], 1e-6)
        self.assertEqual(scores['bbscore_noise_corrected'][0], 1e6)
        bb = summarize_protocols(scores)['bbscore_zwm']
        self.assertEqual(bb['n_neurons'], 1)
        self.assertEqual(bb['ceiled_median'], 1e6)


if __name__ == '__main__':
    unittest.main()
