"""One PBS array element: train with a private SQLite snapshot, no central DB writes.

The PBS wrapper copies the worker's database from local scratch to the trial
folder *after* this process exits. This process never opens the central DB.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import socket
import subprocess
import time
import traceback
from pathlib import Path


def _digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _atomic_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp." + str(os.getpid()))
    try:
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def fix_cuda_visible_devices():
    """Torch 1.13 cannot parse PBS's UUID CUDA mask (verified on MetaCentrum L40)."""
    original = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if original.startswith("GPU-") and "," not in original:
        report = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True,
        )
        matches = [index.strip() for index, uuid in csv.reader(io.StringIO(report))
                   if uuid.strip() == original]
        if len(matches) != 1:
            raise RuntimeError(f"Cannot uniquely resolve allocated GPU UUID {original}: {matches}")
        os.environ["CUDA_VISIBLE_DEVICES"] = matches[0]
    print(f"PBS GPU mask: {original!r} -> {os.environ.get('CUDA_VISIBLE_DEVICES')!r}", flush=True)


def pruning_trial(worker_db: Path, meta: dict):
    """Build a worker-local Optuna pruner from authoritative project epoch rows.

    Uses only public Optuna APIs: create_trial/add_trials/ask/report/should_prune.
    No connection to central storage and no double-booked central trial IDs.
    The local study lives only in memory; the real trial is owned by the
    coordinator's ask/tell cycle and gets its final state at collection.
    """
    import optuna
    import sqlite3
    from optuna.trial import TrialState
    persisted = optuna.load_study(study_name=meta['name'], storage='sqlite:///' + str(worker_db))
    pruner = (optuna.pruners.MedianPruner(
        n_startup_trials=int(meta['pruner_startup_trials']),
        n_warmup_steps=int(meta['pruner_warmup_steps']),
    ) if meta['prune'] else optuna.pruners.NopPruner())
    local = optuna.create_study(direction='maximize', pruner=pruner)
    old_trials = {t.number: t for t in persisted.get_trials(deepcopy=False)}
    with sqlite3.connect(str(worker_db)) as con:
        links = con.execute('''
            SELECT trial_number,run_id FROM optuna_links
            WHERE study_name=? AND run_id IS NOT NULL ORDER BY trial_number
        ''', (meta['name'],)).fetchall()
        for trial_number, run_id in links:
            historical = old_trials[int(trial_number)]
            if historical.state not in (TrialState.COMPLETE, TrialState.PRUNED):
                continue
            epochs = con.execute('''
                SELECT epoch,val_corr_mean FROM epochs
                WHERE run_id=? ORDER BY epoch
            ''', (run_id,)).fetchall()
            metrics = {int(epoch): float(value) for epoch, value in epochs if value is not None}
            if not metrics:
                continue
            clone = optuna.trial.create_trial(
                state=historical.state,
                value=historical.value,
                params=historical.params,
                distributions=historical.distributions,
                intermediate_values=metrics,
            )
            local.add_trial(clone)
    return local, local.ask()


def execute(experiment: Path, batch: str, index: int, worker_db: Path):
    root = experiment.resolve()
    if not batch.startswith('batch-') or not batch[6:].isdigit() or index < 1:
        raise ValueError('Invalid batch or index')
    manifest = json.loads((root / 'batches' / batch / 'manifest.json').read_text())
    records = {int(r['index']): r for r in manifest['records']}
    record = records[index]
    trial_dir = Path(record['container_trial_dir']).resolve()
    if trial_dir.parent != root / 'trials':
        raise ValueError('Trial directory outside study')
    cfg = json.loads((trial_dir / 'config.json').read_text())
    if _digest(cfg) != record['config_sha256']:
        raise RuntimeError('Frozen trial config modified')
    result_path = trial_dir / 'result.json'
    if result_path.exists() and json.loads(result_path.read_text()).get('status') in ('completed', 'pruned'):
        print('Already finished: ' + str(result_path), flush=True)
        return
    if not worker_db.is_file() or worker_db.resolve() == (trial_dir / 'worker_seed.sqlite').resolve():
        raise ValueError('Worker must use a private LOCAL SQLite copy of worker_seed.sqlite')
    meta = json.loads((root / 'experiment.json').read_text())
    cfg['project']['sqlite_path'] = str(worker_db)
    # project.sqlite_path is execution-only, therefore does not change candidate identity.
    lock = trial_dir / '.worker-running'
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f'Worker lock exists: {lock}') from exc
    started = time.monotonic()
    workflow_id = candidate_id = run_id = None
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(f"job={os.environ.get('PBS_JOBID', 'manual')} host={socket.gethostname()}\n")
        try:
            fix_cuda_visible_devices()
            import torch
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError(f'One GPU required, found {torch.cuda.device_count()}')
            from sinz_model.orchestrator import RunRegistry, candidate_config_sha256
            from sinz_model.trainer import TrainingPruned, train_member
            registry = RunRegistry(str(worker_db))
            registry.initialize()
            workflow_name = f"optuna:{meta['name']}:trial:{record['number']}"
            matches = [w for w in registry.list_workflows() if w['name'] == workflow_name]
            if len(matches) > 1:
                raise RuntimeError('Ambiguous trial workflow in worker snapshot')
            workflow_id = matches[0]['workflow_id'] if matches else registry.create_workflow(workflow_name)
            candidates = registry.list_candidates(workflow_id=workflow_id)
            if len(candidates) > 1:
                raise RuntimeError('Multiple candidates in trial workflow')
            if candidates:
                candidate_id = candidates[0]['candidate_id']
                if registry.get_candidate(candidate_id)['config_sha256'] != candidate_config_sha256(cfg):
                    raise ValueError('Candidate configuration changed')
            else:
                candidate_id = registry.create_candidate(workflow_id, 'optuna', cfg, overrides=record['params'])['candidate_id']
            runs = registry.candidate_runs(candidate_id)
            if len(runs) > 1:
                raise RuntimeError('Multiple runs for one trial')
            run_id = runs[0]['run_id'] if runs else None
            _, trial = pruning_trial(worker_db, meta)

            def report(epoch, val):
                if not math.isfinite(val):
                    raise ValueError(f'Invalid validation metric at epoch {epoch}')
                trial.report(float(val), step=int(epoch))
                return bool(meta['prune']) and trial.should_prune()

            try:
                if runs and runs[0]['status'] == 'completed':
                    pass
                elif runs:
                    checkpoint = Path(runs[0]['run_dir']) / 'checkpoints' / 'latest.pt'
                    if not checkpoint.exists():
                        raise RuntimeError('Cannot resume worker: latest.pt missing')
                    train_member(cfg, 0, candidate_id=candidate_id, resume_run_id=run_id,
                                 evaluate_test=False, limit_train_batches=meta['train_batches'],
                                 limit_val_batches=meta['val_batches'], on_epoch_end=report)
                else:
                    run_id = train_member(cfg, 0, candidate_id=candidate_id,
                                          evaluate_test=False, limit_train_batches=meta['train_batches'],
                                          limit_val_batches=meta['val_batches'], on_epoch_end=report)
            except TrainingPruned:
                registry.finish_candidate(candidate_id, status='pruned')
                status = 'pruned'
            else:
                status = 'completed'
                registry.finish_candidate(candidate_id, status='completed')
            if run_id is None:
                runs = registry.candidate_runs(candidate_id)
                run_id = runs[0]['run_id'] if len(runs) == 1 else None
            if run_id is None:
                raise RuntimeError('No recorded training run')
            run = registry.get_run(run_id)
            if run['status'] != status:
                raise RuntimeError(f'Expected run status {status}, got {run["status"]}')
            with registry.connect() as con:
                rows = con.execute('SELECT epoch,val_corr_mean FROM epochs WHERE run_id=? ORDER BY epoch', (run_id,)).fetchall()
            intermediates = {int(r['epoch']): float(r['val_corr_mean']) for r in rows}
            if not intermediates:
                raise RuntimeError('No recorded validation epochs')
            best = registry.best_validation(run_id)
            value = float(best['val_corr_mean'])
            if not math.isfinite(value):
                raise ValueError('Nonfinite objective')
            _atomic_json(result_path, {
                'trial_number': record['number'], 'config_sha256': record['config_sha256'],
                'status': status, 'objective': value, 'best_epoch': int(best['epoch']),
                'intermediate_values': intermediates,
                'workflow_id': workflow_id, 'candidate_id': candidate_id, 'run_id': run_id,
                'pbs_job_id': os.environ.get('PBS_JOBID'), 'host': socket.gethostname(),
                'gpu': torch.cuda.get_device_name(0), 'seconds': time.monotonic() - started,
            })
            print(f"TRIAL {record['number']}: {status}, best val r={value}", flush=True)
        except Exception:
            if candidate_id and not run_id:
                runs = registry.candidate_runs(candidate_id)
                run_id = runs[0]['run_id'] if len(runs) == 1 else None
            err = traceback.format_exc()
            _atomic_json(result_path, {
                'trial_number': record['number'], 'config_sha256': record['config_sha256'],
                'status': 'failed', 'error': err, 'workflow_id': workflow_id,
                'candidate_id': candidate_id, 'run_id': run_id,
                'pbs_job_id': os.environ.get('PBS_JOBID'), 'host': socket.gethostname(),
            })
            print(err, flush=True)
            raise
    finally:
        lock.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', required=True)
    p.add_argument('--batch', required=True)
    p.add_argument('--index', type=int, required=True)
    p.add_argument('--worker-db', required=True)
    a = p.parse_args()
    execute(Path(a.experiment), a.batch, a.index, Path(a.worker_db))


if __name__ == '__main__':
    main()
