"""Coordinator tests require Optuna but do not require torch, GPUs or PBS."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path

from sinz_model import optuna_batch as ob


BASE = Path(__file__).resolve().parents[1]


class OptunaCoordinatorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'optuna' / 'study'
        ob.initialize(argparse.Namespace(
            root=str(self.root), config=str(BASE / 'configs/tvsd_default.yaml'),
            space=str(BASE / 'configs/tvsd_optuna_space.yaml'),
            name='unit-test', seed=1203, member_seed=8000, epochs=25,
        ))

    def _prepare(self, n=3):
        ob.prepare(argparse.Namespace(root=str(self.root), trials=n, data_root=None, image_root=None))
        return ob._read(ob._active_batch(self.root))

    def _result(self, record, score=None):
        path = Path(record['trial_dir']) / 'result.json'
        value = {'trial_number': record['number'], 'config_sha256': record['config_sha256'],
                 'status': 'failed' if score is None else 'completed'}
        if score is not None:
            value['objective'] = score
        ob._write(path, value)

    def test_freezes_configs_and_keeps_one_writer(self):
        manifest = self._prepare(3)
        self.assertEqual(len(manifest['records']), 3)
        with self.assertRaises(RuntimeError):
            self._prepare(2)
        for record in manifest['records']:
            cfg = ob._read(Path(record['trial_dir']) / 'config.json')
            self.assertEqual(cfg['training']['total_epochs'], 25)
            self.assertEqual(cfg['training']['ensemble_members'], 1)
            self.assertEqual(cfg['model']['readout']['outdims'], 1173)
            self.assertTrue(cfg['project']['sqlite_path'].startswith('/mnt/tvsd/'))
            self.assertTrue(cfg['data']['image_root'].endswith('data/assets/THINGS'))
            self.assertEqual(ob._digest(cfg), record['config_sha256'])

    def test_collect_idempotent_and_next_batch(self):
        manifest = self._prepare(3)
        self._result(manifest['records'][0], 0.50)
        self._result(manifest['records'][1], None)
        args = argparse.Namespace(root=str(self.root), fail_missing=False)
        ob.collect(args)
        self.assertFalse(ob._read(ob._active_batch(self.root))['collected'])
        self._result(manifest['records'][2], 0.57)
        ob.collect(args)
        self.assertIsNone(ob._active_batch(self.root))
        with ob._study(self.root) as study:
            self.assertEqual(study.best_value, 0.57)
            self.assertEqual(len(study.trials), 3)
        self._prepare(2)
        with ob._study(self.root) as study:
            self.assertEqual(len(study.trials), 5)

    def test_missing_not_failed_without_explicit_override(self):
        m = self._prepare(2)
        self._result(m['records'][0], 0.45)
        ob.collect(argparse.Namespace(root=str(self.root), fail_missing=False))
        with ob._study(self.root) as study:
            self.assertEqual(study.trials[1].state.name, 'RUNNING')
        ob.collect(argparse.Namespace(root=str(self.root), fail_missing=True))
        with ob._study(self.root) as study:
            self.assertEqual(study.trials[1].state.name, 'FAIL')

    def test_invalid_search_paths(self):
        base = ob._read(self.root / 'base.json')
        for key in ('project.output_root', 'data.root', 'training.member_seeds', 'bad.parameter'):
            with self.assertRaises(ValueError):
                ob._resolve(base, key)

    def test_submission_dry_run_does_not_submit(self):
        self._prepare(3)
        ob.submit(argparse.Namespace(root=str(self.root), concurrency=2, dry_run=True))
        m = ob._read(ob._active_batch(self.root))
        self.assertIsNone(m['pbs_job_id'])


if __name__ == '__main__':
    unittest.main()
