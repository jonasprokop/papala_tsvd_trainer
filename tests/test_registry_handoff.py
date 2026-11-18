"""Single-writer worker-to-project handoff, without Optuna or a PBS node."""
import sqlite3
import socket
from pathlib import Path

import pytest

from sinz_model.orchestrator import RunRegistry
from sinz_model.registry_handoff import break_lock, collect, project_writer_lock
from sinz_model.sqlite_snapshot import atomic_snapshot


def _central(tmp_path):
    central = tmp_path / 'results' / 'results.sqlite'
    reg = RunRegistry(str(central))
    reg.initialize()
    wf = reg.create_workflow('manual-zwm')
    cfg = {'project': {'sqlite_path': str(central),
                       'output_root': str(tmp_path/'outputs')},
           'model': {'family': 'transformer_neural'},
           'training': {'strategy': 'frozen_ridge'}}
    candidate = reg.create_candidate(wf, 'manual', cfg)['candidate_id']
    return central, reg, candidate, cfg


def test_worker_snapshot_collects_one_run_and_is_idempotent(tmp_path):
    central, project, candidate, cfg = _central(tmp_path)
    worker = tmp_path/'worker.sqlite'
    atomic_snapshot(central, worker)
    isolated = RunRegistry(str(worker))
    cfg['project']['sqlite_path'] = str(worker)
    run = isolated.start_run(candidate,0,'training',{'root':8000},'data','split',[],cfg=cfg)
    isolated.log_epoch(run['run_id'],0,'frozen_ridge',float('nan'),float('nan'),.5,.4,0.)
    isolated.finish_run(run['run_id'],'completed')
    assert project.candidate_runs(candidate)==[]  # No compute-node central writes.
    assert collect(central,worker,candidate)==run['run_id']
    assert project.get_run(run['run_id'])['status']=='completed'
    assert project.best_validation(run['run_id'])['val_corr_mean']==.5
    assert collect(central,worker,candidate)==run['run_id']
    assert len(project.candidate_runs(candidate))==1


def test_collected_failed_run_can_be_retried_without_duplicate_run(tmp_path):
    central, project, candidate, cfg = _central(tmp_path)
    worker=tmp_path/'worker.sqlite'
    atomic_snapshot(central,worker)
    isolated=RunRegistry(str(worker))
    cfg['project']['sqlite_path']=str(worker)
    run=isolated.start_run(candidate,0,'training',{'root':8000},'data','split',[],cfg=cfg)
    isolated.finish_run(run['run_id'],'failed',error='simulated walltime')
    collect(central,worker,candidate)
    assert project.get_run(run['run_id'])['status']=='failed'
    isolated.mark_running(run['run_id'])
    isolated.log_epoch(run['run_id'],0,'frozen_ridge',float('nan'),float('nan'),.5,.4,0.)
    isolated.finish_run(run['run_id'],'completed')
    collect(central,worker,candidate)
    assert project.get_run(run['run_id'])['status']=='completed'
    assert len(project.candidate_runs(candidate))==1


def test_handoff_rejects_other_candidate_or_active_coordinator(tmp_path):
    central,_,candidate,_ = _central(tmp_path)
    worker=tmp_path/'worker.sqlite'
    atomic_snapshot(central,worker)
    with pytest.raises(ValueError,match='exactly one'):
        collect(central,worker,candidate)
    with project_writer_lock(central):
        with pytest.raises(RuntimeError,match='already active'):
            collect(central,worker,candidate)


def test_sqlite_snapshot_contains_committed_rows(tmp_path):
    source=tmp_path/'source.sqlite'
    with sqlite3.connect(source) as con:
        con.execute('CREATE TABLE t (i INTEGER)')
        con.execute('INSERT INTO t VALUES (7)')
    target=tmp_path/'snapshot.sqlite'
    atomic_snapshot(source,target)
    with sqlite3.connect(target) as con:
        assert con.execute('SELECT i FROM t').fetchone()[0]==7


def test_break_lock_rejects_running_owner(tmp_path):
    central = tmp_path/'project.sqlite'
    with project_writer_lock(central):
        with pytest.raises(RuntimeError, match='still alive'):
            break_lock(central)
    assert not Path(str(central.resolve()) + '.coordinator.lock').exists()


def test_break_lock_only_removes_confirmed_dead_local_owner(tmp_path, monkeypatch):
    central = tmp_path/'project.sqlite'
    lock = Path(str(central.resolve()) + '.coordinator.lock')
    lock.mkdir()
    (lock/'owner').write_text(f'host={socket.gethostname()} pid=999999999\n')

    def dead(_pid, _signal):
        raise ProcessLookupError('no such process')

    monkeypatch.setattr('sinz_model.owner_lock.os.kill', dead)
    assert break_lock(central) == lock
    assert not lock.exists()
