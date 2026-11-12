"""CUDA-mapping and idempotence tests without a GPU."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from sinz_model.optuna_worker import fix_cuda_visible_devices, execute, _digest


class WorkerTest(unittest.TestCase):
    def test_uuid_mapping(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-abc'}):
            with patch('subprocess.check_output', return_value='0, GPU-abc\n1, GPU-def\n'):
                fix_cuda_visible_devices()
                self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '0')

    def test_numeric_mapping(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}):
            with patch('subprocess.check_output') as query:
                fix_cuda_visible_devices()
                query.assert_not_called()

    def test_completed_result_idempotent_without_gpu(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            trial_dir = root/'trials'/'trial-00000'
            batch_dir = root/'batches'/'batch-001'
            batch_dir.mkdir(parents=True)
            trial_dir.mkdir(parents=True)
            cfg = {'test': 1}
            (trial_dir/'config.json').write_text(json.dumps(cfg))
            (batch_dir/'manifest.json').write_text(json.dumps({'records': [{
                'index': 1, 'number': 0, 'trial_dir': str(trial_dir),
                'container_trial_dir': str(trial_dir), 'config_sha256': _digest(cfg),
            }]}))
            (trial_dir/'result.json').write_text(json.dumps({'status': 'completed'}))
            execute(root, 'batch-001', 1, root/'worker.sqlite')
            self.assertFalse((trial_dir/'.worker-running').exists())


if __name__ == '__main__':
    unittest.main()
