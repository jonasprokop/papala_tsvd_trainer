"""Run the optional transformer strategy in its own Python runtime.

Only JSON/files are exchanged across Python processes. There is no import of
upstream ZWM, PyTorch 2.x or scikit-learn inside the Sinz interpreter.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
import subprocess
import tempfile


def _python(cfg):
    binary = cfg.get('runtime',{}).get('python_executable')
    if not binary:
        raise ValueError('transformer_neural requires runtime.python_executable (ZWM Python 3.10 runtime)')
    binary = Path(binary)
    if not binary.is_file():
        raise FileNotFoundError(f'Transformer Python interpreter not found: {binary}')
    return str(binary)


def _already_in_transformer_runtime(cfg):
    """Do not launch a second Python when PBS selected the ZWM environment."""
    return os.path.realpath(sys.executable) == os.path.realpath(_python(cfg))


def train_transformer_member(cfg, member_index, candidate_id, *, notes=None,
                             resume_run_id=None, dry_run=False,limit_train_batches=None,
                             limit_val_batches=None,evaluate_test=False,on_epoch_end=None):
    if notes is not None:
        raise NotImplementedError('Transformers use frozen-ridge provenance in the run registry')
    if dry_run or limit_train_batches is not None or limit_val_batches is not None:
        raise NotImplementedError('Frozen ridge does not support partial TVSD training; use a small fixture')
    if candidate_id is None:
        raise ValueError('candidate_id is required')
    if _already_in_transformer_runtime(cfg):
        # In the ZWM PBS runtime use one interpreter/process for the full fit.
        from transformer_model.train import train_ridge_member
        run_id = train_ridge_member(cfg, member_index, candidate_id,
                                    evaluate_test=evaluate_test, resume_run_id=resume_run_id)
    else:
        # Compatibility with an explicit Python 3.9 CLI invocation: invoke the
        # *configured* Python 3.10 binary once, without importing ZWM here.
        output=Path(cfg['project']['output_root']); output.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='transformer-dispatch-',dir=output) as folder:
            base=Path(folder)
            config=base/'config.json'; result=base/'result.json'
            config.write_text(json.dumps(cfg,sort_keys=True))
            cmd=[_python(cfg),'-m','transformer_model.train','--config',str(config),
                 '--candidate-id',str(candidate_id),'--member',str(member_index),
                 '--result-file',str(result)]
            if evaluate_test:
                cmd.append('--evaluate-test')
            if resume_run_id is not None:
                cmd.extend(['--resume-run-id', str(resume_run_id)])
            subprocess.run(cmd,check=True)
            if not result.is_file():
                raise RuntimeError('Transformer subprocess did not write run result')
            run_id=json.loads(result.read_text())['run_id']
    if on_epoch_end is not None:
        from .orchestrator import RunRegistry
        value=RunRegistry(cfg['project']['sqlite_path']).best_validation(run_id)
        # Frozen ridge has no intermediate epoch pruning. The worker checks
        # pruning=False; a callback's return cannot invalidate a completed run.
        on_epoch_end(0,float(value['val_corr_mean']))
    return str(run_id)


def export_transformer_bundle(cfg, sqlite_path, candidate_id, destination):
    with tempfile.TemporaryDirectory(prefix='transformer-export-') as folder:
        result=Path(folder)/'exported.json'
        subprocess.run([_python(cfg),'-m','transformer_model.export',
                        '--sqlite-path',str(sqlite_path),'--candidate-id',str(candidate_id),
                        '--destination',str(destination),'--result-file',str(result)],check=True)
        return Path(json.loads(result.read_text())['destination'])


def evaluate_transformer_run(cfg, sqlite_path, run_id):
    with tempfile.TemporaryDirectory(prefix='transformer-eval-') as folder:
        result=Path(folder)/'result.json'
        subprocess.run([_python(cfg),'-m','transformer_model.evaluate',
                        '--sqlite-path',str(sqlite_path),'--run-id',str(run_id),
                        '--result-file',str(result)],check=True)
        return json.loads(result.read_text())['summary']
