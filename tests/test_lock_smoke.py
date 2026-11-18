"""Pure stdlib tests: no torch, Optuna, GPU, PBS daemon, or database server."""
import os
import socket
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sinz_model.owner_lock import break_owner_lock, owner_lock
from sinz_model.sqlite_snapshot import atomic_snapshot


BASE = Path(__file__).resolve().parents[1]


class CoordinatorLockTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.lock = Path(self.temp.name) / 'project.coordinator.lock'

    def test_live_process_cannot_be_unlocked(self):
        with owner_lock(self.lock):
            with self.assertRaisesRegex(RuntimeError, 'still alive'):
                break_owner_lock(self.lock)
        self.assertFalse(self.lock.exists())

    def test_dead_local_pid_can_be_unlocked(self):
        self.lock.mkdir()
        (self.lock / 'owner').write_text(f'host={socket.gethostname()} pid=999999999\n')
        with patch('sinz_model.owner_lock.os.kill', side_effect=ProcessLookupError):
            break_owner_lock(self.lock)
        self.assertFalse(self.lock.exists())

    def test_reject_foreign_invalid_and_inaccessible_owners(self):
        for record in ('host=another-host pid=123\n', 'host=local pid=0\n', 'broken'):
            self.lock.mkdir()
            (self.lock / 'owner').write_text(record)
            with patch('sinz_model.owner_lock.os.kill') as kill:
                with self.assertRaises(RuntimeError):
                    break_owner_lock(self.lock)
                kill.assert_not_called()
            self.assertTrue(self.lock.exists())
            (self.lock / 'owner').unlink()
            self.lock.rmdir()
        self.lock.mkdir()
        (self.lock / 'owner').write_text(f'host={socket.gethostname()} pid=123\n')
        with patch('sinz_model.owner_lock.os.kill', side_effect=PermissionError):
            with self.assertRaisesRegex(RuntimeError, 'Cannot verify'):
                break_owner_lock(self.lock)
        self.assertTrue(self.lock.exists())

    def test_sqlite_snapshot_replaces_atomically(self):
        source = Path(self.temp.name) / 'source.sqlite'
        dest = Path(self.temp.name) / 'snapshot.sqlite'
        with sqlite3.connect(source) as db:
            db.execute('CREATE TABLE t (i INTEGER)')
            db.execute('INSERT INTO t VALUES (1)')
        atomic_snapshot(source, dest)
        with sqlite3.connect(source) as db:
            db.execute('UPDATE t SET i=2')
        atomic_snapshot(source, dest)
        with sqlite3.connect(dest) as db:
            self.assertEqual(db.execute('SELECT i FROM t').fetchone()[0], 2)


class PBSLockTests(unittest.TestCase):
    def _run(self, state):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / '.worker-running').write_text('job=111.pbs-m1.metacentrum.cz\n')
            bin_dir = directory / 'bin'
            bin_dir.mkdir()
            stub = bin_dir / 'qstat'
            stub.write_text('#!/bin/sh\n'
                            '[ "$1" = -x ] && [ "$2" = -f ] || exit 98\n'
                            'printf "    job_state = %s\\n" "$MOCK_QSTATE"\n')
            stub.chmod(0o700)
            script = (BASE / 'jobs' / 'optuna_trial.pbs').read_text()
            block = 'LOCK="$TRIAL_DIR/.worker-running"' + script.split(
                'LOCK="$TRIAL_DIR/.worker-running"', 1)[1].split(
                    '\nexport SINGULARITYENV_PYTHONPATH=', 1)[0]
            env = dict(os.environ, TRIAL_DIR=str(directory),
                       PBS_JOBID='222.pbs-m1.metacentrum.cz', MOCK_QSTATE=state,
                       PATH=str(bin_dir) + os.pathsep + os.environ['PATH'])
            result = subprocess.run(['bash', '-c', 'set -euo pipefail\n' + block],
                                    env=env, capture_output=True, text=True)
            return result, (directory / '.worker-running').exists()

    def test_finished_job_states_release_stale_lock(self):
        for state in ('F', 'X'):
            with self.subTest(state=state):
                result, left = self._run(state)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertFalse(left)

    def test_running_job_is_never_stolen(self):
        result, left = self._run('R')
        self.assertEqual(result.returncode, 5, result.stderr)
        self.assertTrue(left)


if __name__ == '__main__':
    unittest.main()
