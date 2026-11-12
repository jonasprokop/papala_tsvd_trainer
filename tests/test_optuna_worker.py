"""Worker helpers: no GPU, PyTorch or scheduler needed for these tests."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sinz_model.optuna_worker import fix_cuda_visible_devices, execute


class WorkerTest(unittest.TestCase):
    def test_uuid_mapping_before_importing_torch(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-abc'}, clear=False):
            with patch('subprocess.check_output', return_value='0, GPU-abc\n1, GPU-def\n'):
                fix_cuda_visible_devices()
                self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '0')

    def test_numeric_gpu_mask_unmodified(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}, clear=False):
            with patch('subprocess.check_output') as call:
                fix_cuda_visible_devices()
                call.assert_not_called()

    def test_completed_result_is_idempotent(self):
        from sinz_model.optuna_worker import _digest
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            trial = root / 'trials' / 'trial-00000'
            batch = root / 'batches' / 'batch-001'
            trial.mkdir(parents=True)
            batch.mkdir(parents=True)
            cfg = {'toy': 42}
            (trial / 'config.json').write_text(json.dumps(cfg))
            record = {'index': 1, 'number': 0, 'trial_dir': str(trial),
                      'container_trial_dir': str(trial), 'config_sha256': _digest(cfg)}
            (batch / 'manifest.json').write_text(json.dumps({'records': [record]}))
            (trial / 'result.json').write_text(json.dumps({'status': 'completed', 'config_sha256': _digest(cfg)}))
            execute(root, 'batch-001', 1)
            self.assertFalse((trial / '.worker-running').exists())


if __name__ == '__main__':
    unittest.main()
