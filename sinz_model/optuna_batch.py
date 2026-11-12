"""Single-writer Optuna ask/tell batches, with PBS as the scheduler.

The ONE durable SQLite DB contains RunRegistry schema-v4 and Optuna's tables.
Each PBS worker receives a private snapshot of that DB; the collection job
imports *only* its new project rows, then calls study.tell().  The shared DB
is never open for live Optuna writes over the cluster filesystem.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sqlite3
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import optuna
import yaml
from optuna.trial import TrialState

from .orchestrator import RunRegistry

OPTUNA_VERSION = optuna.__version__
RESERVED = {'training.member_seeds', 'training.ensemble_members'}
PROJECT_CHILDREN = {
    'candidate_metrics': 'candidate_id',
    'candidate_artifacts': 'candidate_id',
    'params': 'run_id', 'seeds': 'run_id', 'epochs': 'run_id',
    'metrics': 'run_id', 'checkpoints': 'run_id',
    'artifacts': 'run_id', 'progress': 'run_id',
}
LINK_SCHEMA = '''
CREATE TABLE IF NOT EXISTS optuna_links (
  study_name TEXT NOT NULL,
  trial_number INTEGER NOT NULL,
  batch_name TEXT NOT NULL,
  workflow_id TEXT,
  candidate_id TEXT,
  run_id TEXT,
  terminal_state TEXT NOT NULL,
  PRIMARY KEY (study_name, trial_number),
  UNIQUE(run_id)
);
'''


def _digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _write(path: Path, value: Any):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'{path.name}.tmp.{os.getpid()}')
    try:
        temp.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _read(path):
    return json.loads(Path(path).read_text())


def _resolve(data: dict, dotted: str) -> Any:
    if dotted.startswith(('project.', 'data.')) or dotted in RESERVED:
        raise ValueError(f'Forbidden optimisation parameter: {dotted}')
    node = data
    for part in dotted.split('.'):
        if not isinstance(node, dict) or part not in node:
            raise ValueError(f'Unknown parameter: {dotted}')
        node = node[part]
    return node


def _override(data: dict, dotted: str, value: Any):
    _resolve(data, dotted)
    node = data
    path = dotted.split('.')
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value


def _space(path: Path, base: dict) -> dict:
    raw = yaml.safe_load(path.read_text())
    specs = raw.get('search') if isinstance(raw, dict) else None
    if not isinstance(specs, dict) or not specs:
        raise ValueError('Expected a nonempty search: mapping')
    for name, spec in specs.items():
        original = _resolve(base, name)
        if not isinstance(spec, dict):
            raise ValueError(f'Invalid spec: {name}')
        kind = spec.get('type')
        if kind == 'choice':
            choices = spec.get('values')
            if not isinstance(choices, list) or not choices or any(isinstance(x, (list, dict)) for x in choices):
                raise ValueError(f'Invalid choices: {name}')
            if isinstance(original, (str, bool)) and any(type(v) is not type(original) for v in choices):
                raise ValueError(f'Choice type mismatch: {name}')
        elif kind in ('loguniform', 'uniform'):
            if not isinstance(original, (int, float)) or isinstance(original, bool):
                raise ValueError(f'Expected numeric parameter: {name}')
            low, high = float(spec['low']), float(spec['high'])
            if not (math.isfinite(low) and math.isfinite(high) and low < high):
                raise ValueError(f'Invalid range: {name}')
            if kind == 'loguniform' and low <= 0:
                raise ValueError(f'Log range must be positive: {name}')
        elif kind == 'randint':
            if not isinstance(original, int) or isinstance(original, bool):
                raise ValueError(f'Expected integer parameter: {name}')
            if not isinstance(spec.get('low'), int) or not isinstance(spec.get('high'), int) or spec['high'] <= spec['low']:
                raise ValueError(f'Invalid randint range: {name}')
        else:
            raise ValueError(f'Unsupported distribution {kind}: {name}')
    return specs


def _sample(trial, specs):
    params = {}
    for name, spec in sorted(specs.items()):
        kind = spec['type']
        if kind == 'choice':
            value = trial.suggest_categorical(name, spec['values'])
        elif kind == 'randint':
            value = trial.suggest_int(name, spec['low'], spec['high'])
        else:
            value = trial.suggest_float(name, spec['low'], spec['high'], log=(kind == 'loguniform'))
        params[name] = value
    return params


@contextmanager
def _lock(root: Path):
    """Coordinator jobs serialize through atomic directory creation (not flock/NFS)."""
    root.mkdir(parents=True, exist_ok=True)
    lock = root / '.coordinator.lock'
    try:
        lock.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise RuntimeError(f'Coordinator lock exists: {lock}. Inspect before removing.') from exc
    try:
        (lock / 'owner').write_text(f'pid={os.getpid()}\n')
        yield
    finally:
        (lock / 'owner').unlink(missing_ok=True)
        lock.rmdir()


def _experiment(root: Path):
    path = root / 'experiment.json'
    if not path.exists():
        raise FileNotFoundError(f'Initialise the study first: {path}')
    meta = _read(path)
    if meta.get('schema') != 2 or meta.get('optuna_version') != OPTUNA_VERSION:
        raise RuntimeError('Unsupported study schema or Optuna version (expected 4.8.0)')
    return meta


def _db(meta):
    return Path(meta['project_db']).resolve()


def _sqlite_snapshot(source: Path, target: Path):
    """Consistent SQLite backup even if the source has a journal."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    if not source.exists():
        raise FileNotFoundError(source)
    with sqlite3.connect(f'file:{source}?mode=ro', uri=True) as s:
        with sqlite3.connect(str(target)) as d:
            s.backup(d)


@contextmanager
def _central(root: Path, meta: dict):
    """Run the sole coordinator on local scratch, publish atomically on success.

    Do not edit the authoritative DB from any other process while the
    coordinator owns this operation; normal training's *other* registry is safe.
    """
    db = _db(meta)
    db.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='tvsd-coordinator-', dir=os.environ.get('TMPDIR') or None) as tmp:
        local = Path(tmp) / 'project.sqlite'
        fingerprint = ((db.stat().st_size, db.stat().st_mtime_ns) if db.exists() else None)
        if db.exists():
            _sqlite_snapshot(db, local)
        registry = RunRegistry(str(local))
        registry.initialize()
        with registry.connect() as con:
            con.executescript(LINK_SCHEMA)
        storage = optuna.storages.RDBStorage(url='sqlite:///' + str(local))
        sampler = optuna.samplers.TPESampler(seed=int(meta['seed']), constant_liar=True)
        study = optuna.create_study(
            storage=storage, study_name=meta['name'], direction='maximize',
            sampler=sampler, load_if_exists=True,
        )
        # The study's trial history, not a shared process, carries state.
        n = len(study.get_trials(deepcopy=False))
        if n:
            study.sampler = optuna.samplers.TPESampler(
                seed=int(meta['seed']) + n, constant_liar=True)
        try:
            yield study, registry
        except BaseException:
            storage.engine.dispose()
            raise  # Do not publish a partial collection or sampling operation.
        else:
            storage.engine.dispose()
            current = ((db.stat().st_size, db.stat().st_mtime_ns) if db.exists() else None)
            if current != fingerprint:
                raise RuntimeError('Central project database changed during coordination; refusing overwrite')
            published = db.with_name(f'{db.name}.tmp.{os.getpid()}')
            try:
                with sqlite3.connect(str(local)) as source:
                    with sqlite3.connect(str(published)) as dest:
                        source.backup(dest)
                os.replace(published, db)
            finally:
                published.unlink(missing_ok=True)


def _workspace(root: Path):
    root = root.resolve()
    if root.parent.name != 'optuna':
        raise ValueError('Study root must be <workspace>/optuna/<study-name>')
    return root.parents[1]


def _mapped(path: Path, workspace: Path):
    path = Path(path)
    if path.is_relative_to(Path('/mnt/tvsd')):
        return str(path)
    return str(Path('/mnt/tvsd') / path.resolve().relative_to(workspace.resolve()))


def _active(root: Path):
    for manifest_path in sorted((root / 'batches').glob('*/manifest.json')):
        if not _read(manifest_path).get('collected', False):
            return manifest_path
    return None


def initialize(args):
    root = Path(args.root).resolve()
    workspace = _workspace(root)
    with _lock(root):
        if (root / 'experiment.json').exists():
            raise FileExistsError('Study already initialized')
        base = yaml.safe_load(Path(args.config).read_text())
        specs = _space(Path(args.space), base)
        if args.epochs is not None:
            base['training']['total_epochs'] = int(args.epochs)
        if args.phase1_epochs is not None:
            base['training']['readout_only_epochs'] = int(args.phase1_epochs)
        if not 0 < int(base['training']['readout_only_epochs']) < int(base['training']['total_epochs']):
            raise ValueError('Require >=1 readout-only and >=1 joint epoch')
        base['training']['ensemble_members'] = 1
        base['training']['member_seeds'] = [int(args.member_seed)]
        project_db = Path(args.project_db or workspace / 'results' / 'project.sqlite').resolve()
        if not project_db.is_relative_to(workspace):
            raise ValueError('Project DB must be inside the shared workspace')
        meta = {
            'schema': 2, 'name': args.name, 'seed': int(args.seed),
            'project_db': str(project_db), 'base_digest': _digest(base),
            'space_digest': _digest(specs), 'optuna_version': OPTUNA_VERSION,
            'train_batches': args.train_batches, 'val_batches': args.val_batches,
            'prune': bool(args.prune), 'pruner_startup_trials': int(args.pruner_startup_trials),
            'pruner_warmup_steps': int(args.pruner_warmup_steps),
        }
        if meta['train_batches'] is not None and meta['train_batches'] < 1:
            raise ValueError('train-batches must be >=1')
        if meta['val_batches'] is not None and meta['val_batches'] < 1:
            raise ValueError('val-batches must be >=1')
        if project_db.exists():
            # Protect manual training data against implicit Optuna replacement.
            with sqlite3.connect(str(project_db)) as c:
                if c.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                    raise RuntimeError('Existing project DB failed integrity_check')
        with _central(root, meta):
            pass
        _write(root / 'base.json', base)
        _write(root / 'space.json', specs)
        _write(root / 'experiment.json', meta)
        print(f'Project registry + Optuna study: {project_db}')


def prepare(args):
    root = Path(args.root).resolve()
    workspace = _workspace(root)
    meta = _experiment(root)
    n = int(args.trials)
    if not (1 <= n <= 1000):
        raise ValueError('trials must be 1..1000')
    with _lock(root):
        if _active(root):
            raise RuntimeError('An uncollected batch already exists')
        base, specs = _read(root / 'base.json'), _read(root / 'space.json')
        if _digest(base) != meta['base_digest'] or _digest(specs) != meta['space_digest']:
            raise RuntimeError('Frozen experiment settings changed')
        batch_index = len(list((root / 'batches').glob('*/manifest.json'))) + 1
        batch_name = f'batch-{batch_index:03d}'
        batch = root / 'batches' / batch_name
        if batch.exists():
            raise FileExistsError(batch)
        batch.mkdir(parents=True)
        records = []
        with _central(root, meta) as (study, _):
            if any(t.state == TrialState.RUNNING for t in study.get_trials(deepcopy=False)):
                raise RuntimeError('Unaccounted running trial; collect or repair first')
            for index in range(1, n + 1):
                trial = study.ask()
                params = _sample(trial, specs)
                trial_dir = root / 'trials' / f'trial-{trial.number:05d}'
                if trial_dir.exists():
                    raise FileExistsError(trial_dir)
                cfg = copy.deepcopy(base)
                for key, val in params.items():
                    _override(cfg, key, val)
                cfg['project']['name'] = f'{base["project"]["name"]}-optuna-{trial.number}'
                ct = Path(_mapped(trial_dir, workspace))
                cfg['project'].update(
                    # Only worker-local SQLite is writable during training.
                    sqlite_path='/worker/worker.sqlite',
                    # Persistent checkpoint/record paths remain valid on skirit.
                    output_root=str(trial_dir / 'runs'),
                    result_sheet_path=str(trial_dir / 'runs.csv'),
                )
                data_root = Path(args.data_root or workspace / 'data' / 'tvsd_waveform_aware_v1' / 'model_dataset').resolve()
                cfg['data']['root'] = str(data_root)
                cfg['data']['image_root'] = str(Path(args.image_root or workspace / 'data' / 'assets' / 'THINGS').resolve())
                cfg['data']['preprocessing']['normalization_path'] = str(Path(cfg['data']['root']) / 'normalization.json')
                cfg['data']['dataset_module_path'] = '/project/waveform_aware_tvsd_preprocessing/scripts/dataset.py'
                cfg['training']['progress']['enabled'] = False
                trial.set_user_attr('batch', batch_name)
                _write(trial_dir / 'config.json', cfg)
                records.append({
                    'index': index, 'number': trial.number,
                    'trial_dir': str(trial_dir), 'container_trial_dir': str(ct),
                    'params': params, 'config_sha256': _digest(cfg),
                })
        _write(batch / 'manifest.json', {
            'schema': 2, 'batch': batch_name, 'records': records,
            'array_job_id': None, 'collector_job_id': None,
            'submission_pending': False, 'collected': False,
        })
        # Optuna has committed its RUNNING trials; snapshots are now coherent.
        for record in records:
            snapshot = Path(record['trial_dir']) / 'worker_seed.sqlite'
            if not snapshot.exists():
                _sqlite_snapshot(_db(meta), snapshot)
        print(f'Prepared {n} private read-only snapshots and configs in {batch}')


def _pbs_env(root, batch):
    if any(s in str(root) for s in (' ', ',', '\n')):
        raise ValueError('PBS environment path cannot contain spaces/commas/newlines')
    return f'OPTUNA_EXPERIMENT={root},OPTUNA_BATCH={batch}'


def submit(args):
    root = Path(args.root).resolve()
    _experiment(root)
    with _lock(root):
        active = _active(root)
        if not active:
            raise RuntimeError('No pending batch')
        manifest = _read(active)
        if manifest['array_job_id'] or manifest['submission_pending']:
            raise RuntimeError('Already submitted or submission uncertain; inspect qstat')
        n, cap = len(manifest['records']), int(args.concurrency)
        if not 1 <= cap <= n:
            raise ValueError('concurrency must be 1..n')
        paths = Path(__file__).resolve().parents[1] / 'jobs'
        env = _pbs_env(root, manifest['batch'])
        array = ['qsub', '-J', f'1-{n}', f'%{cap}', '-v', env, str(paths / 'optuna_trial.pbs')]
        if args.dry_run:
            print('Array:', ' '.join(array))
            print('Collector: qsub -W depend=afterany:<array-id> -v ' + env + ' ' + str(paths / 'optuna_collect.pbs'))
            return
        manifest['submission_pending'] = True
        _write(active, manifest)
        try:
            array_id = subprocess.run(array, text=True, check=True, capture_output=True).stdout.strip()
        except subprocess.CalledProcessError:
            manifest['submission_pending'] = False
            _write(active, manifest)
            raise
        if not array_id:
            raise RuntimeError('qsub array returned empty job ID; inspect qstat')
        manifest['array_job_id'] = array_id
        _write(active, manifest)
        collector_cmd = ['qsub', '-W', f'depend=afterany:{array_id}', '-v', env,
                         str(paths / 'optuna_collect.pbs')]
        # A collection submission failure must NOT schedule the array twice.
        try:
            collector_id = subprocess.run(collector_cmd, text=True, check=True, capture_output=True).stdout.strip()
        except subprocess.CalledProcessError:
            print('Array submitted, collector submission failed. Submit collector manually:\n' + ' '.join(collector_cmd))
            raise
        if not collector_id:
            raise RuntimeError('Collector qsub returned empty ID; inspect qstat')
        manifest['collector_job_id'] = collector_id
        manifest['submission_pending'] = False
        _write(active, manifest)
        print(f'Scheduled array {array_id} ({n} trials, cap {cap}) then collector {collector_id}')


def _insert_from_worker(con, table, selector, param):
    # Enumerate column names from *our* local project schema. AUTOINCREMENT
    # columns are deliberately omitted so IDs can never collide on import.
    columns = [r[1] for r in con.execute(f'PRAGMA main.table_info({table})')
               if r[1] not in ('metric_id', 'artifact_id', 'checkpoint_id')]
    if not columns:
        raise RuntimeError(f'Missing target table: {table}')
    cols = ', '.join('"' + x + '"' for x in columns)
    con.execute(f'INSERT INTO main."{table}" ({cols}) SELECT {cols} FROM worker."{table}" WHERE {selector}', (param,))


def _import_one(registry: RunRegistry, worker_path: Path, result: dict, study_name: str,
                trial_number: int, batch_name: str, terminal_state: str):
    """Import exactly the workflow/candidate/run spawned by one worker."""
    wf, candidate, run = (result.get(key) for key in ('workflow_id', 'candidate_id', 'run_id'))
    with registry.connect() as con:
        previous = con.execute(
            'SELECT workflow_id,candidate_id,run_id FROM optuna_links WHERE study_name=? AND trial_number=?',
            (study_name, trial_number),
        ).fetchone()
        if previous:
            if (previous['workflow_id'], previous['candidate_id'], previous['run_id']) != (wf, candidate, run):
                raise RuntimeError(f'Conflicting import for trial {trial_number}')
            return
        if not run:
            with con:
                con.execute('INSERT INTO optuna_links VALUES (?,?,?,?,?,?,?)',
                            (study_name, trial_number, batch_name, wf, candidate, run, terminal_state))
            return
        if not all((wf, candidate, run)):
            raise ValueError('Run result lacks workflow/candidate/run identifiers')
        if not worker_path.is_file():
            raise FileNotFoundError(worker_path)
        if con.execute('SELECT 1 FROM runs WHERE run_id=?', (run,)).fetchone():
            raise RuntimeError('Run ID already exists in central DB but has no Optuna link')
        # ATTACH is supported for immutable snapshots and single-worker DBs;
        # only this coordinator reads the worker DB during collection.
        con.execute('ATTACH DATABASE ? AS worker', (str(worker_path),))
        try:
            with con:
                src = con.execute('SELECT candidate_id,status FROM worker.runs WHERE run_id=?', (run,)).fetchone()
                if not src or src['candidate_id'] != candidate:
                    raise RuntimeError('Worker run ID / candidate mismatch')
                if terminal_state == 'COMPLETE' and src['status'] != 'completed':
                    raise RuntimeError('Worker reports success but run is incomplete')
                if terminal_state == 'PRUNED' and src['status'] != 'pruned':
                    raise RuntimeError('Worker reports pruned but run status disagrees')
                if terminal_state == 'FAIL' and src['status'] == 'completed':
                    raise RuntimeError('Worker reports failure but run completed')
                if con.execute('SELECT 1 FROM worker.candidates WHERE candidate_id=? AND workflow_id=?', (candidate, wf)).fetchone() is None:
                    raise RuntimeError('Worker candidate/workflow mismatch')
                _insert_from_worker(con, 'workflows', 'workflow_id=?', wf)
                _insert_from_worker(con, 'candidates', 'candidate_id=?', candidate)
                for table in ('candidate_metrics', 'candidate_artifacts'):
                    _insert_from_worker(con, table, 'candidate_id=?', candidate)
                _insert_from_worker(con, 'runs', 'run_id=?', run)
                if terminal_state == 'FAIL' and src['status'] == 'running':
                    con.execute("UPDATE runs SET status='failed', error=? WHERE run_id=?",
                                (result.get('error', 'Worker terminated'), run))
                for table, key in PROJECT_CHILDREN.items():
                    if key == 'run_id':
                        _insert_from_worker(con, table, 'run_id=?', run)
                con.execute('INSERT INTO optuna_links VALUES (?,?,?,?,?,?,?)',
                            (study_name, trial_number, batch_name, wf, candidate, run, terminal_state))
        finally:
            con.execute('DETACH DATABASE worker')


def _result_state(result: dict):
    status = result['status']
    return {'completed': TrialState.COMPLETE, 'pruned': TrialState.PRUNED,
            'failed': TrialState.FAIL}[status]


def _validate_epoch_metrics(registry: RunRegistry, run_id: str, result: dict):
    """The project registry, not a worker's JSON, defines epoch metrics."""
    with registry.connect() as con:
        rows = con.execute('''SELECT epoch, val_corr_mean FROM epochs
                              WHERE run_id=? ORDER BY epoch''', (run_id,)).fetchall()
    recorded = {int(r['epoch']): float(r['val_corr_mean']) for r in rows}
    declared = {int(k): float(v) for k, v in result.get('intermediate_values', {}).items()}
    if recorded != declared or not all(math.isfinite(v) for v in recorded.values()):
        raise ValueError(f'Worker result disagrees with run registry: {run_id}')
    return recorded


def collect(args):
    root = Path(args.root).resolve()
    meta = _experiment(root)
    with _lock(root):
        active = _active(root)
        if active is None:
            print('Nothing to collect')
            return
        manifest = _read(active)
        pending = done = failed = pruned = 0
        with _central(root, meta) as (study, registry):
            for record in manifest['records']:
                trial_number = record['number']
                current = study.trials[trial_number]
                if current.state.is_finished():
                    # A successful re-collection must be idempotent.
                    continue
                result_path = Path(record['trial_dir']) / 'result.json'
                if not result_path.exists():
                    if not args.fail_missing:
                        pending += 1
                        continue
                    result = {'status': 'failed', 'trial_number': trial_number,
                              'config_sha256': record['config_sha256'], 'error': 'Missing worker result after array termination'}
                else:
                    result = _read(result_path)
                if result.get('trial_number') != trial_number or result.get('config_sha256') != record['config_sha256']:
                    raise ValueError(f'Result mismatch: {result_path}')
                state = _result_state(result)
                worker_db = Path(record['trial_dir']) / 'worker_result.sqlite'
                _import_one(registry, worker_db, result, meta['name'], trial_number, manifest['batch'], state.name)
                if result.get('run_id') and state in (TrialState.COMPLETE, TrialState.PRUNED):
                    epochs = _validate_epoch_metrics(registry, result['run_id'], result)
                    value = float(result['objective'])
                    if not epochs or not math.isfinite(value) or abs(value - max(epochs.values())) > 1e-9:
                        raise ValueError(f'Invalid objective in {result_path}')
                if state == TrialState.COMPLETE:
                    if not result.get('run_id'):
                        raise ValueError('Successful worker must provide recorded run_id')
                    study.tell(trial_number, values=value)
                    done += 1
                else:
                    study.tell(trial_number, state=state)
                    if state == TrialState.PRUNED:
                        pruned += 1
                    else:
                        failed += 1
        if pending == 0:
            manifest['collected'] = True
            _write(active, manifest)
        print(f'Collected completed={done} pruned={pruned} failed={failed} pending={pending}')


def status(args):
    root = Path(args.root).resolve()
    meta = _experiment(root)
    with _lock(root):
        with _central(root, meta) as (study, registry):
            trials = study.get_trials(deepcopy=False)
            states = {s.name: sum(t.state == s for t in trials) for s in TrialState}
            with registry.connect() as con:
                n_links = con.execute('SELECT count(*) FROM optuna_links WHERE study_name=?', (meta['name'],)).fetchone()[0]
            complete = [t for t in trials if t.state == TrialState.COMPLETE]
            best = max(complete, key=lambda t: t.value) if complete else None
            print(json.dumps({'db': meta['project_db'], 'n_trials': len(trials), 'states': states,
                              'imported_runs': n_links,
                              'best': ({'number': best.number, 'value': best.value, 'params': best.params}
                                       if best else None)}, indent=2))
        active = _active(root)
        if active:
            manifest = _read(active)
            print('Pending:', manifest['batch'], 'array:', manifest['array_job_id'],
                  'collector:', manifest['collector_job_id'])


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    i = sub.add_parser('init')
    i.add_argument('--root', required=True)
    i.add_argument('--project-db', default=None, help='Existing or new authoritative project SQLite file')
    i.add_argument('--config', required=True)
    i.add_argument('--space', required=True)
    i.add_argument('--name', default='tvsd-optuna')
    i.add_argument('--seed', type=int, default=107173)
    i.add_argument('--member-seed', type=int, default=8000)
    i.add_argument('--epochs', type=int)
    i.add_argument('--phase1-epochs', type=int)
    i.add_argument('--train-batches', type=int, default=None)
    i.add_argument('--val-batches', type=int, default=None)
    i.add_argument('--prune', action='store_true', help='Enable MedianPruner; disabled for short smoke tests')
    i.add_argument('--pruner-startup-trials', type=int, default=5)
    i.add_argument('--pruner-warmup-steps', type=int, default=10)
    i.set_defaults(func=initialize)
    q = sub.add_parser('prepare')
    q.add_argument('--root', required=True)
    q.add_argument('--trials', type=int, required=True)
    q.add_argument('--data-root')
    q.add_argument('--image-root')
    q.set_defaults(func=prepare)
    q = sub.add_parser('submit')
    q.add_argument('--root', required=True)
    q.add_argument('--concurrency', type=int, default=4)
    q.add_argument('--dry-run', action='store_true')
    q.set_defaults(func=submit)
    q = sub.add_parser('collect')
    q.add_argument('--root', required=True)
    q.add_argument('--fail-missing', action='store_true')
    q.set_defaults(func=collect)
    q = sub.add_parser('status')
    q.add_argument('--root', required=True)
    q.set_defaults(func=status)
    return p


def main():
    a = parser().parse_args()
    a.func(a)


if __name__ == '__main__':
    main()
