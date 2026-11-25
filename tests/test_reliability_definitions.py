"""Reliability definitions: dense equivalence, oracle expectation, session separation."""
import unittest

import numpy as np

from evaluation.evaluation_protocols import (oracle_30rep_model_estimate, papale_oracle_dense,
                                     papale_reliab_dense, score_protocols)
from evaluation.compare_evaluations import summarize_protocols


def population(ev, reps, n_img=100, n_neu=60, seed=0, n_sess=1, offset_sd=0.0,
               unbalanced=False):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=(n_img, n_neu))
    image = np.repeat(np.arange(n_img), reps)
    if unbalanced:   # each image seen in a single session
        session = image % n_sess
    else:            # every image seen in every session
        session = np.tile(np.repeat(np.arange(n_sess), reps // n_sess), n_img)
    noise = rng.normal(size=(len(image), n_neu)) * np.sqrt((1 - ev) / ev)
    offsets = rng.normal(scale=offset_sd, size=(n_sess, n_neu))
    y = signal[image] + noise + offsets[session]
    pred = (.8 * signal + .6 * rng.normal(size=signal.shape))[image]
    return pred, y, np.ones_like(y, bool), image, session


class ReliabilityDefinitionTests(unittest.TestCase):
    def test_sampled_estimate_matches_dense_all_pairs(self):
        pred, y, m, img, _ = population(.3, 30)
        s = score_protocols(pred, y, m, img, splits=200)
        dense = [papale_reliab_dense(y[:, n].reshape(100, 30).T) for n in range(y.shape[1])]
        diff = s['single_trial_pairwise_reliability_estimate'] - np.asarray(dense)
        self.assertLess(abs(np.mean(diff)), .005)     # no bias
        self.assertLess(np.std(diff), .02)            # Monte-Carlo spread only

    def test_adapted_oracle_matches_dense_leave_one_out(self):
        pred, y, m, img, _ = population(.3, 30)
        s = score_protocols(pred, y, m, img, splits=200)
        dense = [papale_oracle_dense(y[:, n].reshape(100, 30).T) for n in range(y.shape[1])]
        diff = s['papale_oracle_adapted'] - np.asarray(dense)
        self.assertLess(abs(np.mean(diff)), .005)
        self.assertLess(np.std(diff), .02)

    def test_three_definitions_differ_as_theory_says(self):
        ev, n = .3, 30
        pred, y, m, img, _ = population(ev, n)
        s = score_protocols(pred, y, m, img, splits=20)
        med = lambda k: np.median(s[k])
        self.assertAlmostEqual(med('single_trial_pairwise_reliability_estimate'), ev, delta=.03)
        self.assertAlmostEqual(med('split_half_sb_reliability'), n * ev / (1 + (n - 1) * ev), delta=.02)
        # r(one trial, mean of other n-1) = sqrt(ev * R_{n-1})
        r_rest = (n - 1) * ev / (1 + (n - 2) * ev)
        self.assertAlmostEqual(med('papale_oracle_adapted'), np.sqrt(ev * r_rest), delta=.03)

    def test_session_diagnostics_separate_noise_from_drift(self):
        for offset_sd, expect_gap in ((0.0, False), (1.0, True)):
            pred, y, m, img, ses = population(.3, 30, n_sess=5, offset_sd=offset_sd)
            sess = summarize_protocols(score_protocols(pred, y, m, img, sessions=ses))['reliability']['sessions']
            gap = sess['pairwise_reliability_within_session'] - sess['pairwise_reliability_between_sessions']
            self.assertEqual(gap > .05, expect_gap)
            self.assertEqual(sess['session_offset_sd_over_signal_sd'] > .5, expect_gap)
            self.assertEqual(sess['median_repeats_per_session_cell'], 6)
            # Within-session estimate is drift-free: recovers the true single-trial value.
            self.assertAlmostEqual(sess['pairwise_reliability_within_session'], .3, delta=.03)

    def test_unshared_images_are_exposed_by_pooled_above_within(self):
        pred, y, m, img, ses = population(.3, 30, n_sess=5, offset_sd=1.0, unbalanced=True)
        r = summarize_protocols(score_protocols(pred, y, m, img, sessions=ses))['reliability']
        self.assertIsNone(r['sessions']['pairwise_reliability_between_sessions'])
        self.assertGreater(r['single_trial_pairwise_estimate_median']
                           - r['sessions']['pairwise_reliability_within_session'], .08)

    def test_papale_formulas_per_neuron_exact_on_complete_block(self):
        _, y, m, img, _ = population(.3, 30, n_neu=20)
        s = score_protocols(None, y, m, img, splits=5, balanced_order='presentation')
        blocks = [y[:, n].reshape(100, 30).T for n in range(20)]
        np.testing.assert_allclose(s['papale_reliab_balanced_adapted'],
                                   [papale_reliab_dense(b) for b in blocks], atol=1e-12)
        np.testing.assert_allclose(s['papale_oracle_balanced_adapted'],
                                   [papale_oracle_dense(b) for b in blocks], atol=1e-12)
        self.assertTrue((s['papale_balanced_k'] == 30).all())

    def test_balanced_order_controls_drift_sensitivity(self):
        _, y, m, img, ses = population(.3, 30, n_sess=5, offset_sd=1.0)
        order = np.lexsort((ses, img))           # presentation order = session-aligned
        y, img, ses = y[order], img[order], ses[order]
        aligned = score_protocols(None, y, m, img, splits=5, balanced_order='presentation')
        shuffled = score_protocols(None, y, m, img, splits=5)
        self.assertAlmostEqual(np.median(aligned['papale_reliab_balanced_adapted']), .3, delta=.03)
        self.assertLess(np.median(shuffled['papale_reliab_balanced_adapted']), .26)

    def test_papale_formulas_on_ragged_repeats(self):
        _, y, m, img, _ = population(.3, 30, n_neu=40)
        rng = np.random.default_rng(1)
        for i in range(100):   # 3..30 repeats per image
            m[np.flatnonzero(img == i)[rng.integers(3, 31):], :] = False
        s = score_protocols(None, y, m, img, splits=5)
        self.assertAlmostEqual(np.median(s['papale_reliab_balanced_adapted']), .3, delta=.03)
        self.assertAlmostEqual(np.median(s['papale_oracle_30rep_model_estimate']),
                               float(oracle_30rep_model_estimate(.3)), delta=.03)

    def test_prediction_free_mode(self):
        _, y, m, img, ses = population(.3, 30, n_sess=5)
        s = score_protocols(None, y, m, img, sessions=ses)
        self.assertTrue(np.isnan(s['cadena_feve']).all())
        self.assertAlmostEqual(np.median(s['single_trial_pairwise_reliability_estimate']), .3, delta=.03)


if __name__ == '__main__':
    unittest.main()


def neurovisfit_reference(targets, outputs, threshold=0.15):
    """Verbatim logic of neurovisfit/evaluation/metrics.py (+ neuralpredictors.corr)
    for one session: targets/outputs = list over images of [reps, neurons]."""
    def corr(y1, y2, axis=0, eps=1e-8):
        y1 = (y1 - y1.mean(axis=axis, keepdims=True)) / (y1.std(axis=axis, keepdims=True, ddof=0) + eps)
        y2 = (y2 - y2.mean(axis=axis, keepdims=True)) / (y2.std(axis=axis, keepdims=True, ddof=0) + eps)
        return (y1 * y2).mean(axis=axis)
    image_variance = [np.var(t, axis=0, ddof=1) for t in targets]
    pred_var = np.vstack([(t - o) ** 2 for t, o in zip(targets, outputs)])
    total = np.var(np.vstack(targets), axis=0, ddof=1)
    noise = np.mean(np.vstack(image_variance), axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        fev = (total - noise) / total
        feve = 1 - (np.mean(pred_var, axis=0) - noise) / (total - noise)
        raw = corr(np.array([t.mean(0) for t in targets]), np.array([o.mean(0) for o in outputs]))
    cc = raw[fev > threshold]
    cc[np.isnan(cc)] = 0
    return float(np.mean(cc)), float(np.mean(feve[fev > threshold])), raw, fev, feve


class CadenaSourceTests(unittest.TestCase):
    def dense(self):
        rng = np.random.default_rng(3)
        n_img, reps, n_neu = 75, 12, 30
        signal = rng.normal(size=(n_img, n_neu)) * np.linspace(.1, 2, n_neu)
        img = np.repeat(np.arange(n_img), reps)
        y = rng.poisson(np.exp(signal[img] * .5 + 1)).astype(float)
        p = np.exp(.5 * (.7 * signal + .3 * rng.normal(size=signal.shape)) + 1)[img]
        p[:, 0] = 2.0                          # constant prediction -> corr 0 (eps), not NaN
        y[:, 1] = 3.0                          # constant response -> FEV NaN -> excluded
        return p, y, np.ones_like(y, bool), img, reps

    def test_formula_matches_neurovisfit_source(self):
        p, y, m, img, reps = self.dense()
        s = score_protocols(p, y, m, img, splits=2)
        self.assertTrue(s['cadena_formula_applicable'].all())
        t = [y[img == i] for i in range(75)]
        o = [p[img == i] for i in range(75)]
        cc_ref, fe_ref, raw, fev, feve = neurovisfit_reference(t, o)
        np.testing.assert_allclose(s['cadena_corr_to_average'], raw, atol=1e-12)
        np.testing.assert_allclose(s['cadena_explainable_fraction'], fev, atol=1e-12, equal_nan=True)
        keep = fev > .15
        np.testing.assert_allclose(s['cadena_feve'][keep], feve[keep], atol=1e-12)
        out = summarize_protocols(s)['cadena2024_formula_match']
        self.assertAlmostEqual(out['cc_avg_mean'], cc_ref, places=12)
        self.assertAlmostEqual(out['feve_mean'], fe_ref, places=12)
        self.assertEqual(s['cadena_corr_to_average'][0], 0.0)

    def test_tracked_reports_inapplicable_neurons(self):
        p, y, m, img, _ = self.dense()
        m[img == 0, 5] = False                 # neuron 5 never saw image 0
        m[np.flatnonzero(img == 1)[1:], 6] = False   # neuron 6 saw image 1 once
        s = score_protocols(p, y, m, img, splits=2)
        self.assertFalse(s['cadena_formula_applicable'][5])
        self.assertFalse(s['cadena_formula_applicable'][6])
        summary = summarize_protocols(s)
        self.assertEqual(summary['cadena2024_formula_match']['n_considered'], 28)
        self.assertEqual(summary['cadena2024_tracked']['n_considered'], 30)


    def test_ragged_repeats_match_neurovisfit_source(self):
        """Unequal repeats (2..12 per image): NeuroVisFit accepts ragged lists;
        FEVE pools presentations, noise averages images unweighted. Must match."""
        p, y, m, img, _ = self.dense()
        rng = np.random.default_rng(7)
        keep_reps = rng.integers(2, 13, size=75)
        for i in range(75):
            m[np.flatnonzero(img == i)[keep_reps[i]:], :] = False
        s = score_protocols(p, y, m, img, splits=2)
        self.assertTrue(s['cadena_formula_applicable'].all())
        t = [y[(img == i) & m[:, 0]] for i in range(75)]
        o = [p[(img == i) & m[:, 0]] for i in range(75)]
        cc_ref, fe_ref, raw, fev, feve = neurovisfit_reference(t, o)
        np.testing.assert_allclose(s['cadena_corr_to_average'], raw, atol=1e-12)
        keep = fev > .15
        np.testing.assert_allclose(s['cadena_feve'][keep], feve[keep], atol=1e-12)
        out = summarize_protocols(s)['cadena2024_formula_match']
        self.assertAlmostEqual(out['cc_avg_mean'], cc_ref, places=12)
        self.assertAlmostEqual(out['feve_mean'], fe_ref, places=12)
        # Presentation weighting matters: an image-weighted MSE gives a different FEVE.
        img_weighted = np.mean([np.mean((a - b) ** 2, 0) for a, b in zip(t, o)], 0)
        self.assertGreater(np.max(np.abs(img_weighted - np.mean(np.vstack([(a - b) ** 2 for a, b in zip(t, o)]), 0))), 1e-3)

    def test_single_observation_image_marks_neuron_adapted(self):
        """NeuroVisFit would get NaN noise from a 1-trial image; we drop that image
        for the neuron and flag it, so it leaves the formula-match population."""
        p, y, m, img, _ = self.dense()
        m[np.flatnonzero(img == 3)[1:], 4] = False
        s = score_protocols(p, y, m, img, splits=2)
        self.assertFalse(s['cadena_formula_applicable'][4])
        self.assertTrue(np.isfinite(s['cadena_feve'][4]))


class ReportingTests(unittest.TestCase):
    def test_population_accounting_and_reliability_source(self):
        _, y, m, img, _ = population(.4, 10, n_img=40, n_neu=6)
        pred = y.copy()
        groups = np.array(list('FFFNNN'))
        s = score_protocols(pred, y, m, img, splits=3)
        recon = summarize_protocols(s, groups)
        self.assertIn('brainscore_current_adapted', recon)
        self.assertNotIn('brainscore_current', recon)
        self.assertTrue(recon['brainscore_current_adapted']['reliability_source'].startswith('reconstructed'))
        measured = summarize_protocols(s, groups, measured_reliab=np.full(6, .5))
        self.assertEqual(measured['brainscore_current']['reliability_source'], 'measured_papale_reliab')
        pop = recon['population']
        self.assertEqual(pop['n_neurons'], 6)
        self.assertEqual(set(pop['per_group']), {'F', 'N'})
        for key in ('cadena2024_formula_match', 'cadena2024_tracked', 'bbscore_zwm', 'attenuation'):
            self.assertEqual(set(recon[key]['per_group']), {'F', 'N'})


class NeuronIdTests(unittest.TestCase):
    def test_string_ids_seed_stably_and_subset_invariant(self):
        _, y, m, img, _ = population(.3, 10, n_img=40, n_neu=6)
        ids = np.array(['F/0', 'F/1', 'F/2', 'N/0', 'N/1', 'N/12'])
        full = score_protocols(None, y, m, img, splits=3, neuron_ids=ids, min_repeated_images=20)
        part = score_protocols(None, y[:, 3:], m[:, 3:], img, splits=3, neuron_ids=ids[3:],
                               min_repeated_images=20)
        np.testing.assert_array_equal(full['split_half_sb_reliability'][3:],
                                      part['split_half_sb_reliability'])
        self.assertEqual(list(full['neuron_id']), list(ids))

    def test_duplicate_ids_rejected(self):
        _, y, m, img, _ = population(.3, 4, n_img=30, n_neu=2)
        with self.assertRaises(ValueError):
            score_protocols(None, y, m, img, neuron_ids=np.array(['F/0', 'F/0']))
