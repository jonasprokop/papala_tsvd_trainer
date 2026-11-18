"""No-GPU checks for image and PBS mount contracts."""
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MetacentrumLaunchContract(unittest.TestCase):
    def test_pbs_shell_syntax(self):
        scripts = [ROOT / 'jobs' / name for name in
                   ('optuna_trial.pbs', 'optuna_collect.pbs', 'zwm_train.pbs')]
        subprocess.run(['bash', '-n', *(str(p) for p in scripts)], check=True)

    def test_single_image_and_bind(self):
        for name in ('optuna_trial.pbs', 'optuna_collect.pbs', 'zwm_train.pbs'):
            content = (ROOT / 'jobs' / name).read_text()
            self.assertIn('containers/tvsd-bookworm.sif', content)
            self.assertIn('--bind "$ROOT:/mnt/tvsd"', content)
            self.assertNotIn('--bind "$ROOT:$ROOT"', content)
        optuna = (ROOT / 'jobs/optuna_trial.pbs').read_text()
        self.assertIn('PYTHON=/mnt/tvsd/envs/tvsd-py39/bin/python', optuna)
        self.assertIn('PYTHON=/mnt/tvsd/envs/zwm-py310/bin/python', optuna)
        self.assertIn('[ -L "$HOST_PYTHON" ]', optuna)
        zwm = (ROOT / 'jobs/zwm_train.pbs').read_text()
        self.assertIn('PYTHON=/mnt/tvsd/envs/zwm-py310/bin/python', zwm)

    def test_modern_image_build_is_explicit(self):
        dockerfile = (ROOT / 'Dockerfile.metacentrum').read_text()
        workflow = (ROOT / '.github/workflows/metacentrum-image.yml').read_text()
        self.assertIn('FROM python:3.9.25-bookworm', dockerfile)
        self.assertIn('/usr/bin/python3.9', dockerfile)
        self.assertIn('torch==1.13.1+cu117', dockerfile)
        self.assertIn('workflow_dispatch:', workflow)
        self.assertIn('file: ./Dockerfile.metacentrum', workflow)
        self.assertIn('push: true', workflow)


if __name__ == '__main__':
    unittest.main()
