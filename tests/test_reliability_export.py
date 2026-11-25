"""End-to-end: exporter on a synthetic prepared dataset in the real on-disk format."""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HAS_TORCH = importlib.util.find_spec('torch') is not None


def build_prepared(root, seed=0):
    """F: 6 tracks / 3 sessions, N: 4 tracks / 2 sessions, 100 test images x 4 reps
    per session. Track 0 of each animal drifts across sessions; the last track is
    absent from the last session."""
    root = Path(root)
    (root / 'sessions').mkdir(parents=True)
    rng = np.random.default_rng(seed)
    widths = {'F': 6, 'N': 4}
    sessions = {'F': ['macaqueF_TVSD_A', 'macaqueF_TVSD_B', 'macaqueF_TVSD_C'],
                'N': ['macaqueN_TVSD_A', 'macaqueN_TVSD_B']}
    rows, units = [], []
    for animal, names in sessions.items():
        w = widths[animal]
        signal = rng.normal(3, 1, size=(100, w)).clip(0.2)
        drift = np.zeros((len(names), w))
        drift[:, 0] = [0, 2, -1][:len(names)]
        for si, s in enumerate(names):
            present = np.ones(w, bool)
            if si == len(names) - 1:
                present[w - 1] = False
            img = np.repeat(np.arange(100), 4)
            rng.shuffle(img)
            img = np.concatenate([img, np.arange(10) + 1000])        # a few train trials
            split = np.where(img >= 1000, 'train', 'test')
            lam = np.where(img[:, None] < 1000, signal[np.minimum(img, 99)] + drift[si], 2.0)
            local = np.flatnonzero(present)
            counts = rng.poisson(lam.clip(0.05) * 3).astype(np.uint16)[:, local]
            np.savez(root / 'sessions' / f'{s}.npz', counts=counts,
                     observed=np.ones_like(counts, bool), neuron_indices=local,
                     source_rows=np.arange(len(img)),
                     stimulus_ids=np.array([f'{sp}:{i}' for sp, i in zip(split, img)]),
                     response_window_ms=np.array([40., 160.]))
            for t, (sp, i) in enumerate(zip(split, img)):
                rows.append(dict(animal=animal, session=s, trial_index=t, source_row=t,
                                 stimulus_id=f'{sp}:{i}', partition='test' if sp == 'test' else 'train',
                                 image_relative_path='x.jpg'))
            for n in local:
                units.append(dict(unit_instance_id=f'{s}-{n}', session=s, animal=animal,
                                  candidate_joint_track_id=f'{animal}t{n}', neuron_index=n,
                                  neuron_id=f'{animal}/{n}'))
    pd.DataFrame(rows).to_csv(root / 'sample_index.csv', index=False)
    u = pd.DataFrame(units)
    u.to_csv(root / 'unit_to_neuron.csv', index=False)
    (u.groupby(['animal', 'neuron_index', 'neuron_id', 'candidate_joint_track_id'])
      .session.nunique().rename('n_kept_recordings').reset_index()
      .to_csv(root / 'neuron_index.csv', index=False))
    labels = u[['unit_instance_id']].assign(
        final_class=np.where(u.neuron_index % 2 == 0, 'narrow', 'broad'), area='V1')
    labels.to_csv(root / 'labels.csv', index=False)
    (root / 'dataset_info.json').write_text(json.dumps(dict(
        version='synthetic', n_neurons=widths, split_sha256='x',
        frozen_input_sha256={}, response_window_ms=[40, 160])))
    return root


def run_export(prepared, out, extra=()):
    return subprocess.run([sys.executable, '-m', 'evaluation.reliability_export',
                           '--prepared-root', str(prepared), '--output-dir', str(out),
                           '--splits', '10', *extra],
                          cwd=Path(__file__).resolve().parents[1],
                          capture_output=True, text=True)


@unittest.skipUnless(HAS_TORCH, 'sinz_model.tvsd_data imports torch')
class ReliabilityExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.prepared = build_prepared(self.tmp / 'prepared')

    def test_export_flags_drift_and_missing_sessions(self):
        out = self.tmp / 'out'
        r = run_export(self.prepared, out, ['--unit-labels', str(self.prepared / 'labels.csv')])
        self.assertEqual(r.returncode, 0, r.stderr)
        t = pd.read_csv(out / 'reliability_per_neuron.csv').set_index('neuron_id')
        for drifted in ('F/0', 'N/0'):
            self.assertGreater(t.loc[drifted, 'pairwise_reliability_within_session']
                               - t.loc[drifted, 'pairwise_reliability_between_sessions'], .2)
            self.assertGreater(t.loc[drifted, 'session_offset_sd_over_signal_sd'], .5)
        self.assertLess(t.loc['F/1', 'session_offset_sd_over_signal_sd'], .3)
        self.assertEqual(t.loc['F/5', 'n_sessions'], 2)
        self.assertEqual(t.loc['N/3', 'n_sessions'], 1)
        self.assertTrue(np.isnan(t.loc['N/3', 'pairwise_reliability_between_sessions']))
        self.assertIn(t.loc['F/0', 'final_class'], ('narrow', 'broad'))
        s = json.loads((out / 'reliability_summary.json').read_text())
        self.assertEqual(s['source']['neuron_ids_in_column_order'][:2], ['F/0', 'F/1'])
        self.assertIn('by_final_class', s)
        self.assertTrue((out / 'reliability_per_session.csv').is_file())

    def test_never_overwrites(self):
        out = self.tmp / 'out'
        self.assertEqual(run_export(self.prepared, out).returncode, 0)
        self.assertNotEqual(run_export(self.prepared, out).returncode, 0)

    def test_corrupted_neuron_id_aborts(self):
        idx = pd.read_csv(self.prepared / 'neuron_index.csv')
        idx.loc[0, 'neuron_id'] = 'F/99'
        idx.to_csv(self.prepared / 'neuron_index.csv', index=False)
        r = run_export(self.prepared, self.tmp / 'bad')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('neuron_id', r.stderr)


if __name__ == '__main__':
    unittest.main()
