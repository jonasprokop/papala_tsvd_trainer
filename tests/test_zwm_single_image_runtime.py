"""Contract checks for selecting two Python prefixes inside ONE SIF."""
import json
import os
import sys
from pathlib import Path

import pytest

from sinz_model.orchestrator import RunRegistry
from sinz_model import transformer_bridge
from transformer_model import launch


def test_bridge_uses_current_python_when_zwm_is_active(tmp_path, monkeypatch):
    from transformer_model import train as training
    called=[]
    monkeypatch.setattr(training,'train_ridge_member',lambda cfg,member_index,candidate_id,**kw:
                        called.append((cfg,member_index,candidate_id)) or 'fake-run')
    cfg={'model':{'family':'transformer_neural'},
         'runtime':{'python_executable':sys.executable},
         'project':{'output_root':str(tmp_path)}}
    assert transformer_bridge.train_transformer_member(cfg,0,'candidate',resume_run_id='old-id')=='fake-run'
    assert called and called[0][2]=='candidate'


def test_direct_entrypoint_loads_existing_candidate(tmp_path,monkeypatch):
    registry=RunRegistry(str(tmp_path/'results.sqlite'))
    registry.initialize()
    wf=registry.create_workflow('zwm-test')
    cfg={'model':{'family':'transformer_neural'},'training':{'strategy':'frozen_ridge'},
         'project':{'sqlite_path':str(tmp_path/'results.sqlite'),'output_root':str(tmp_path)}}
    candidate=registry.create_candidate(wf,'test',cfg)['candidate_id']
    monkeypatch.setattr(launch,'train_ridge_member',
                        lambda cfg,member,candidate_id,**kw: 'direct-run')
    assert launch.run(registry.sqlite_path,candidate,0)=='direct-run'
    with pytest.raises(ValueError, match='not a transformer'):
        cfg['model']['family']='sinz'
        other=registry.create_candidate(wf,'test',cfg)['candidate_id']
        launch.run(registry.sqlite_path,other)


def test_one_sif_two_python_prefixes():
    root=Path(__file__).resolve().parents[1]
    optuna=(root/'jobs/optuna_trial.pbs').read_text()
    zwm=(root/'jobs/zwm_train.pbs').read_text()
    cfg=(root/'configs/tvsd_zwm_ridge.yaml').read_text()
    assert 'containers/tvsd-bookworm.sif' in zwm
    assert 'containers/tvsd-bookworm.sif' in optuna
    assert 'envs/tvsd-py39/bin/python' in optuna
    assert 'envs/zwm-py310/bin/python' in optuna
    assert 'envs/zwm-py310/bin/python' in zwm
    assert '/mnt/tvsd/envs/zwm-py310/bin/python' in cfg
    assert 'strategy: frozen_ridge' in cfg
    assert 'zwm.sif' not in optuna and 'ZWM_SIF' not in zwm


def test_metacentrum_paths_and_single_project_registry():
    root = Path(__file__).resolve().parents[1]
    zwm = (root / 'jobs/zwm_train.pbs').read_text()
    optuna = (root / 'jobs/optuna_trial.pbs').read_text()
    collector = (root / 'jobs/optuna_collect.pbs').read_text()
    for script in (zwm, optuna, collector):
        assert 'code/papala_tsvd_trainer' in script
        assert 'code/papala_tvsd_trainer' not in script
        assert '--bind "$ROOT:$ROOT"' not in script
        assert '--bind "$ROOT:/mnt/tvsd"' in script
        assert 'containers/tvsd-bookworm.sif' in script
    assert 'HOST_PYTHON=' in optuna
    assert '[ -L "$HOST_PYTHON" ]' in optuna
    assert 'SINGULARITYENV_PYTHONPATH="/project:/mnt/tvsd/code/ZWM"' in optuna
    assert 'SINGULARITYENV_PYTHONPATH="/project:/mnt/tvsd/code/ZWM"' in zwm
    import yaml
    config = yaml.safe_load((root / 'configs/tvsd_zwm_ridge.yaml').read_text())
    assert config['project']['sqlite_path'] == config['project']['output_root'] + '/results.sqlite'
    assert 'results/results.sqlite' in zwm
    assert '--sqlite-path /worker/zwm_worker.sqlite' in zwm
    assert 'sqlite_snapshot.py' in zwm
    assert 'trap on_term TERM' in optuna
    assert 'trap on_term TERM' in zwm
