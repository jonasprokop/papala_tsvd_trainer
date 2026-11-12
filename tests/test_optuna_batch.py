"""Actual project/Optuna SQLite integration tests; no GPU or PBS required."""
import argparse
import json
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

import optuna
from sinz_model import optuna_batch as ob
from sinz_model.orchestrator import RunRegistry

BASE = Path(__file__).resolve().parents[1]


class BatchTest(unittest.TestCase):
    def setUp(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.workspace = Path(td.name)
        self.root = self.workspace / 'optuna' / 'study'
        self.db = self.workspace / 'results' / 'project.sqlite'
        ob.initialize(argparse.Namespace(
            root=str(self.root), project_db=str(self.db),
            config=str(BASE / 'configs/tvsd_default.yaml'),
            space=str(BASE / 'configs/tvsd_optuna_space.yaml'),
            name='integration-test', seed=3, member_seed=8000, epochs=2,
            phase1_epochs=1, train_batches=2, val_batches=None,
            prune=True, pruner_startup_trials=1, pruner_warmup_steps=0,
        ))

    def prepare(self, n=4):
        ob.prepare(argparse.Namespace(root=str(self.root), trials=n, data_root=None, image_root=None))
        return ob._read(ob._active(self.root))

    def result(self, record, scores=(0.15, 0.3), status='completed'):
        trial_dir = Path(record['trial_dir'])
        worker_db = trial_dir / 'worker_result.sqlite'
        shutil.copy2(trial_dir / 'worker_seed.sqlite', worker_db)
        cfg = ob._read(trial_dir / 'config.json')
        cfg['project']['sqlite_path'] = str(worker_db)
        cfg['project']['output_root'] = str(trial_dir / 'runs')
        registry = RunRegistry(str(worker_db))
        registry.initialize()
        wf = registry.create_workflow(f'optuna:integration-test:trial:{record["number"]}')
        candidate = registry.create_candidate(wf, 'optuna', cfg, overrides=record['params'])['candidate_id']
        run = registry.start_run(candidate_id=candidate, member_index=0, command='synthetic',
                                 seeds={'model': 8000}, dataset_fingerprint='fingerprint',
                                 split_sha256='split_hash', session_mapping=[], cfg=cfg)['run_id']
        for epoch, score in enumerate(scores):
            registry.log_epoch(run, epoch, 'readout_only' if epoch == 0 else 'joint_finetune',
                               1, 1, score, score, 0.001)
        registry.finish_run(run, status)
        registry.finish_candidate(candidate, status)
        ob._write(trial_dir / 'result.json', {
            'trial_number': record['number'], 'config_sha256': record['config_sha256'],
            'status': status, 'objective': max(scores), 'intermediate_values': dict(enumerate(scores)),
            'workflow_id': wf, 'candidate_id': candidate, 'run_id': run,
        })
        return run

    def collect(self, fail_missing=False):
        ob.collect(argparse.Namespace(root=str(self.root), fail_missing=fail_missing))

    def study(self):
        return optuna.load_study(study_name='integration-test', storage='sqlite:///' + str(self.db))

    def test_four_workers_import_metrics_and_retain_history_for_pruning(self):
        batch = self.prepare(4)
        self.assertEqual(len(batch['records']), 4)
        with self.assertRaises(RuntimeError):
            self.prepare(1)
        # Independent snapshots predate worker execution; central DB is unchanged.
        before = self.db.read_bytes()
        ids = [self.result(record, scores=(0.10 + 0.01*i, 0.30 + 0.01*i))
               for i, record in enumerate(batch['records'])]
        self.assertEqual(self.db.read_bytes(), before)
        self.collect()
        with sqlite3.connect(str(self.db)) as con:
            self.assertEqual(con.execute('SELECT count(*) FROM runs').fetchone()[0], 4)
            self.assertEqual(con.execute('SELECT count(*) FROM epochs').fetchone()[0], 8)
            self.assertEqual(con.execute('SELECT count(*) FROM optuna_links').fetchone()[0], 4)
            self.assertEqual(con.execute('SELECT count(DISTINCT run_id) FROM runs').fetchone()[0], 4)
            self.assertEqual({r[0] for r in con.execute('SELECT run_id FROM runs')}, set(ids))
            self.assertEqual(con.execute('PRAGMA foreign_key_check').fetchall(), [])
        study = self.study()
        self.assertEqual([t.state.name for t in study.trials], ['COMPLETE']*4)
        # Epoch metrics live in the project registry, not duplicated in native Optuna tables.
        self.assertTrue(all(len(t.intermediate_values) == 0 for t in study.trials))
        self.assertAlmostEqual(study.best_value, 0.33)
        # Idempotent collection: no repeated imports/trials.
        self.collect()
        self.assertEqual(len(self.study().trials), 4)
        second = self.prepare(4)
        # New workers carry completed history from first batch, available to a MedianPruner.
        first = second['records'][0]
        local = Path(first['trial_dir']) / 'worker_seed.sqlite'
        from sinz_model.optuna_worker import pruning_trial
        local_study, trial = pruning_trial(local, ob._read(self.root/'experiment.json'))
        self.assertEqual(len(local_study.trials), 5)  # four historical + active local trial
        self.assertEqual(len(local_study.trials[0].intermediate_values), 2)
        trial.report(-1.0, step=0)
        self.assertTrue(trial.should_prune())

    def test_afterany_missing_worker_marks_failed(self):
        batch = self.prepare(4)
        self.result(batch['records'][0])
        self.collect(fail_missing=True)
        self.assertEqual([t.state.name for t in self.study().trials],
                         ['COMPLETE', 'FAIL', 'FAIL', 'FAIL'])
        self.assertEqual(self.study().trials[0].value, .3)
        with sqlite3.connect(str(self.db)) as con:
            self.assertEqual(con.execute('SELECT count(*) FROM runs').fetchone()[0], 1)
            self.assertEqual(con.execute('SELECT count(*) FROM optuna_links').fetchone()[0], 4)

    def test_partial_collection_then_repeat(self):
        batch = self.prepare(2)
        self.result(batch['records'][0])
        self.collect(fail_missing=False)
        self.assertIsNotNone(ob._active(self.root))
        self.assertEqual([t.state.name for t in self.study().trials], ['COMPLETE','RUNNING'])
        self.result(batch['records'][1])
        self.collect()
        self.assertIsNone(ob._active(self.root))
        self.assertEqual([t.state.name for t in self.study().trials], ['COMPLETE','COMPLETE'])

    def test_submit_array_then_dependency_collector(self):
        from unittest.mock import patch
        batch = self.prepare(4)
        class Job:
            def __init__(self, value):
                self.stdout = value
        with patch('sinz_model.optuna_batch.subprocess.run', side_effect=[
            Job('123[].pbs-m1.metacentrum.cz\n'), Job('124.pbs-m1.metacentrum.cz\n'),
        ]) as qsub:
            ob.submit(argparse.Namespace(root=str(self.root), concurrency=4, dry_run=False))
        self.assertEqual(qsub.call_count, 2)
        arr = qsub.call_args_list[0].args[0]
        collect = qsub.call_args_list[1].args[0]
        self.assertEqual(arr[:4], ['qsub', '-J', '1-4', '%4'])
        self.assertIn('depend=afterany:123[].pbs-m1.metacentrum.cz', collect)
        manifest = ob._read(ob._active(self.root))
        self.assertEqual(manifest['collector_job_id'], '124.pbs-m1.metacentrum.cz')
        self.assertFalse(manifest['submission_pending'])

    def test_invalid_search_and_submission_dry_run(self):
        batch = self.prepare(2)
        for key in ('data.root', 'project.output_root', 'training.member_seeds', 'unknown'):
            with self.assertRaises(ValueError):
                ob._resolve(ob._read(self.root / 'base.json'), key)
        ob.submit(argparse.Namespace(root=str(self.root), concurrency=2, dry_run=True))
        self.assertIsNone(ob._read(ob._active(self.root))['array_job_id'])


if __name__ == '__main__':
    unittest.main()
