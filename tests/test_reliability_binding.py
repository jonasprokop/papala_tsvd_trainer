"""The saved track reliabilities must never silently bind to wrong neurons."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.compare_evaluations import summarize_protocols
from evaluation.reliability_binding import load_tracked_reliability


class ReliabilityBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'prepared'
        self.export = self.root / 'export'
        self.data.mkdir()
        self.export.mkdir()
        self.ids = ['F/0', 'F/1', 'N/0']
        info = dict(split_sha256='split1', frozen_input_sha256={'F': 'data1'},
                    response_window_ms=[40, 100])
        (self.data / 'dataset_info.json').write_text(json.dumps(info))
        (self.export / 'reliability_summary.json').write_text(json.dumps({
            'source': dict(info, neuron_ids_in_column_order=self.ids)}))
        with (self.export / 'reliability_per_neuron.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['neuron_id', 'papale_reliab_balanced_adapted'])
            writer.writeheader()
            writer.writerows([dict(neuron_id=i, papale_reliab_balanced_adapted=v)
                              for i, v in [('N/0', .8), ('F/1', .3), ('F/0', .1)]])

    def test_matches_by_identity_not_csv_row_order(self):
        actual = load_tracked_reliability(self.export, self.ids, self.data)
        np.testing.assert_allclose(actual, [.1, .3, .8])

    def test_rejects_different_order(self):
        with self.assertRaisesRegex(ValueError, 'order/identities'):
            load_tracked_reliability(self.export, list(reversed(self.ids)), self.data)

    def test_rejects_different_dataset(self):
        path = self.data / 'dataset_info.json'
        obj = json.loads(path.read_text())
        obj['split_sha256'] = 'new-split'
        path.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError, 'split_sha256'):
            load_tracked_reliability(self.export, self.ids, self.data)

    def test_missing_provenance_key_is_rejected_not_passed(self):
        for side in ('data', 'export'):
            self.setUp()
            if side == 'data':
                path = self.data / 'dataset_info.json'
                obj = json.loads(path.read_text()); obj.pop('frozen_input_sha256')
            else:
                path = self.export / 'reliability_summary.json'
                obj = json.loads(path.read_text()); obj['source'].pop('frozen_input_sha256')
            path.write_text(json.dumps(obj))
            with self.assertRaisesRegex(ValueError, 'missing'):
                load_tracked_reliability(self.export, self.ids, self.data)

    def test_nan_reliability_row_is_parsed(self):
        path = self.export / 'reliability_per_neuron.csv'
        path.write_text('neuron_id,papale_reliab_balanced_adapted\nF/0,\nF/1,0.3\nN/0,0.8\n')
        actual = load_tracked_reliability(self.export, self.ids, self.data)
        self.assertTrue(np.isnan(actual[0]))
        np.testing.assert_allclose(actual[1:], [.3, .8])

    def test_exported_values_remain_labeled_adapted(self):
        scores = dict(raw_repeated_pearson=np.array([.6, .7, .8]),
                      split_half_sb_reliability=np.array([.6, .6, .6]),
                      split_half_sb_by_split=np.full((3, 2), .6),
                      bbscore_noise_corrected=np.array([.4, .5, .6]),
                      bbscore_split_half_sb_reliability=np.array([.6, .6, .6]),
                      papale_reliab_balanced_adapted=np.array([.5, .5, .5]),
                      n_repeated_images=np.array([20, 20, 20]),
                      n_cadena_images=np.array([30, 30, 30]),
                      n_observations=np.array([60, 60, 60]),
                      cadena_formula_applicable=np.array([True, True, True]),
                      cadena_explainable_fraction=np.array([.3, .3, .3]),
                      cadena_corr_to_average=np.array([.5, .5, .5]),
                      cadena_feve=np.array([.4, .4, .4]),
                      papale_oracle_balanced_adapted=np.array([.2, .2, .2]),
                      papale_oracle_30rep_model_estimate=np.array([.2, .2, .2]),
                      single_trial_pairwise_reliability_estimate=np.array([.2, .2, .2]),
                      papale_balanced_k=np.array([5, 5, 5]),
                      papale_balanced_n_images=np.array([20, 20, 20]),
                      papale_oracle_adapted=np.array([.2, .2, .2]),
                      pearson_noise_corrected=np.array([.7, .8, .9]),
                      papale_reliab_balanced_adapted_mc_se=np.array([.01, .01, .01]))
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            summarize_protocols(scores, groups=np.array(['F', 'F', 'N']),
                                reconstructed_reliab=np.array([.1, .3, .8]))
        scores['papale_reliab_balanced_adapted'] = np.array([.1, .3, .8])
        out = summarize_protocols(scores, groups=np.array(['F', 'F', 'N']),
                                  reconstructed_reliab=np.array([.1, .3, .8]))
        check = out['reliability']['export_vs_recomputed']
        self.assertTrue(check['passed'])
        self.assertEqual(check['max_abs_diff'], 0.0)
        self.assertTrue(out['reliability']['filter_reliability_source'].startswith('exported'))
        self.assertIn('brainscore_current_adapted', out)
        self.assertNotIn('brainscore_current', out)
        self.assertEqual(out['brainscore_current_adapted']['per_group']['N']['n_neurons'], 1)
        self.assertEqual(out['reliability']['n_above']['0.3'], 1)



class ExportConsistencyTests(unittest.TestCase):
    """Criterion in compare_evaluations.export_consistency, on the real estimator."""

    @classmethod
    def setUpClass(cls):
        from evaluation.evaluation_protocols import score_protocols
        rng = np.random.default_rng(0)
        n_neu, n_img = 60, 60
        ev = rng.uniform(.05, .6, n_neu)
        reps = rng.integers(3, 11, size=n_img)
        img = np.repeat(np.arange(n_img), reps)
        sig = rng.normal(size=(n_img, n_neu))
        y = sig[img] + rng.normal(size=(len(img), n_neu)) * np.sqrt((1 - ev) / ev)
        mask = rng.random(y.shape) > .1
        ids = np.array([f'F/{i}' for i in range(n_neu)])

        def rel(order, yy=y, mm=mask):
            s = score_protocols(None, yy[order], mm[order], img[order], splits=2, neuron_ids=ids)
            return s['papale_reliab_balanced_adapted'], s['papale_reliab_balanced_adapted_mc_se']
        cls.base, _ = rel(np.arange(len(img)))
        cls.shuffled = rel(np.lexsort((rng.random(len(img)), img)))
        cls.noisier = rel(np.arange(len(img)), yy=y + rng.normal(size=y.shape) * .7)

    def test_same_data_other_row_order_passes(self):
        from evaluation.compare_evaluations import export_consistency
        check = export_consistency(self.base, *self.shuffled)
        self.assertTrue(check['passed'], check)
        self.assertLess(check['median_abs_z'], 1.0)

    def test_swapped_neurons_fail(self):
        from evaluation.compare_evaluations import export_consistency
        r, se = self.shuffled
        perm = np.random.default_rng(3).permutation(len(r))
        self.assertFalse(export_consistency(self.base, r[perm], se)['passed'])

    def test_differently_preprocessed_responses_fail(self):
        from evaluation.compare_evaluations import export_consistency
        self.assertFalse(export_consistency(self.base, *self.noisier)['passed'])

    def test_missing_se_cannot_pass(self):
        from evaluation.compare_evaluations import export_consistency
        self.assertFalse(export_consistency(self.base, self.base, None)['passed'])

    def test_all_nan_standard_errors_cannot_pass(self):
        from evaluation.compare_evaluations import export_consistency
        check = export_consistency(np.array([.1, .9]), np.array([.9, .1]),
                                   np.array([np.nan, np.nan]))
        self.assertFalse(check['passed'])
        self.assertEqual(check['n_invalid_se'], 2)

    def test_one_sided_missing_reliability_cannot_pass(self):
        from evaluation.compare_evaluations import export_consistency
        check = export_consistency(np.array([.1, np.nan]), np.array([.1, .4]),
                                   np.array([.01, .01]))
        self.assertFalse(check['passed'])
        self.assertEqual(check['n_missing_one_side'], 1)

    def test_no_comparable_neurons_cannot_pass(self):
        from evaluation.compare_evaluations import export_consistency
        check = export_consistency(np.array([np.nan]), np.array([np.nan]),
                                   np.array([np.nan]))
        self.assertFalse(check['passed'])


if __name__ == '__main__':
    unittest.main()
