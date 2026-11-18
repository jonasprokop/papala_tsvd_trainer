"""Synthetic contract tests: TVSD rows -> ridge -> shared registry -> export.

These tests do NOT claim that a released ZWM checkpoint has been executed.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

from sinz_model.orchestrator import RunRegistry
from transformer_model import train as train_module
from transformer_model.model import export_zwm_bundle, load_zwm_bundle


class FakeExtractor:
    def __init__(self):
        self.calls=0
        self.closed=False
    def extract(self, frame):
        self.calls+=1
        p=np.asarray(frame, dtype=float)[...,0].mean()/255.
        return np.array([[p, p*p, p*p*p]],dtype=np.float32)
    def close(self):
        self.closed=True


def _synthetic_ridge_fixture(tmp_path, monkeypatch, *, chunk_size=256,
                             solver='spectral'):
    dataset=tmp_path/'dataset'; dataset.mkdir()
    images=tmp_path/'images'; images.mkdir()
    upstream=tmp_path/'ZWM'; chk=upstream/'out'/'awwkl'/'test'/'model.pt'
    chk.parent.mkdir(parents=True); chk.write_bytes(b'fake-upstream-checkpoint')
    rows=[]
    y=[]
    for k in range(16):
        pixel=25+k*12
        Image.new('RGB',(12,12),(pixel,pixel,pixel)).save(images/f'{k}.png')
        rows.append({'stimulus_id':f'train:{k}','image_relative_path':f'{k}.png',
                     'partition':'train' if k<12 else 'val'})
        x=pixel/255.
        y.append([0.5+2*x,1.-x])
    df=pd.DataFrame(rows)
    # The F/N columns are still disjoint in the shared two-neuron axis.
    (dataset/'neuron_index.csv').write_text('animal,neuron_index,neuron_id\nF,0,F/0\nN,0,N/0\n')
    table=SimpleNamespace(rows=df,responses=np.asarray(y,dtype=np.float32),
                          mask=np.ones((len(df),2),dtype=bool))
    bundle=SimpleNamespace(widths={'F':1,'N':1},full=table,
            mapping=[{'animal':'F','session_id':'F1'},{'animal':'N','session_id':'N1'}],
            fingerprint='synthetic-fingerprint',split=SimpleNamespace(sha256='synthetic-split'))
    monkeypatch.setattr(train_module,'TVSDTrainingBundle',lambda cfg:bundle)
    root=tmp_path/'run'
    cfg={'model':{'family':'transformer_neural',
                  'backbone':{'kind':'zwm','repo_root':str(upstream),
                              'model_ref':'awwkl/test/model.pt','layer':'transformer.h.22',
                              'input_mode':'grayscale_rgb','pooling':'mean',
                              'cache_chunk_size':chunk_size},
                  'readout':{'outdims':2,'outdims_by_animal':{'F':1,'N':1}}},
         'project':{'output_root':str(root),'sqlite_path':str(root/'results.sqlite'),
                    'feature_cache_root':str(root/'cache')},
         'data':{'kind':'tvsd','root':str(dataset),'image_root':str(images)},
         'training':{'strategy':'frozen_ridge','ensemble_members':1,'member_seeds':[8000],'seed':8000},
         'ridge':{'alphas':[0.001,0.01,0.1], 'cv_folds':2, 'solver':solver}}
    registry=RunRegistry(cfg['project']['sqlite_path']); registry.initialize()
    wf=registry.create_workflow('synthetic-zwm')
    candidate=registry.create_candidate(wf,'search',cfg)['candidate_id']
    return cfg,registry,candidate,chk


def test_frozen_ridge_is_registered_and_exportable(tmp_path,monkeypatch):
    cfg,registry,candidate,chk=_synthetic_ridge_fixture(tmp_path,monkeypatch)
    fake=FakeExtractor()
    run_id=train_module.train_ridge_member(cfg,0,candidate,extractor=fake)
    assert fake.calls==16 and fake.closed
    assert registry.get_run(run_id)['status']=='completed'
    assert registry.best_validation(run_id)['val_corr_mean']>.95
    assert registry.best_checkpoint(run_id).endswith('best.pt')
    with registry.connect() as con:
        assert con.execute('SELECT COUNT(*) FROM epochs WHERE run_id=?',(run_id,)).fetchone()[0]==1
        assert con.execute('SELECT COUNT(*) FROM metrics WHERE run_id=? AND split=?',(run_id,'val')).fetchone()[0]==4
    bundle_dir=export_zwm_bundle(registry.sqlite_path,candidate, tmp_path/'export')
    manifest=json.loads((bundle_dir/'manifest.json').read_text())
    assert manifest['neuron_ids']==['F/0','N/0']
    assert manifest['valid_neuron_mask']==[True,True]
    assert manifest['run_id']==run_id
    assert (bundle_dir/'model.pt').read_bytes()==chk.read_bytes()
    assert 'readout_state.pt' in {p.name for p in bundle_dir.iterdir()}
    state=torch.load(bundle_dir/'readout_state.pt',weights_only=True)
    assert state['weight'].shape==(2,3)
    assert state['neuron_ids']==manifest['neuron_ids']
    assert state['valid_neuron_mask'].tolist()==manifest['valid_neuron_mask']
    from transformer_model import backbones
    class StubZWM(FakeExtractor):
        def __init__(self,*args,**kwargs): super().__init__()
    monkeypatch.setattr(backbones,'ZWMFeatures',StubZWM)
    loaded=load_zwm_bundle(bundle_dir,zwm_repo_root=tmp_path/'ZWM',
                           expected_neuron_ids=['F/0','N/0'])
    assert loaded.predict(Image.new('RGB',(8,8),(100,)*3)).shape==(1,2)
    loaded.close()
    with pytest.raises(ValueError,match='expected dataset'):
        load_zwm_bundle(bundle_dir,zwm_repo_root=tmp_path/'ZWM',
                        expected_neuron_ids=['N/0','F/0'])
    (bundle_dir/'manifest.json').write_text(json.dumps({**manifest,'neuron_ids':['N/0','F/0']}))
    with pytest.raises(ValueError,match='identities/order'):
        load_zwm_bundle(bundle_dir,zwm_repo_root=tmp_path/'ZWM')
    with pytest.raises(FileExistsError):
        export_zwm_bundle(registry.sqlite_path,candidate,tmp_path/'export')


def test_frozen_ridge_retry_preserves_run_and_reuses_features(tmp_path,monkeypatch):
    cfg,registry,candidate,_=_synthetic_ridge_fixture(tmp_path,monkeypatch,chunk_size=1)
    class Preempted(FakeExtractor):
        def extract(self,frame):
            if self.calls:
                raise RuntimeError('simulated PBS preemption')
            return super().extract(frame)
    first=Preempted()
    with pytest.raises(RuntimeError,match='simulated PBS preemption'):
        train_module.train_ridge_member(cfg,0,candidate,extractor=first)
    original=registry.candidate_runs(candidate)[0]
    assert original['status']=='failed'
    assert first.calls==1 and first.closed
    second=FakeExtractor()
    same_run=train_module.train_ridge_member(cfg,0,candidate,
                                             extractor=second,resume_run_id=original['run_id'])
    assert same_run==original['run_id']
    assert len(registry.candidate_runs(candidate))==1
    assert second.calls==15 and second.closed
    assert registry.get_run(same_run)['status']=='completed'
    assert registry.get_run(same_run)['created_at_utc']==original['created_at_utc']
    history=json.loads((Path(original['run_dir'])/'ridge_retry_history.json').read_text())
    assert len(history)==1 and history[0]['previous_status']=='failed'
    assert 'simulated PBS preemption' in history[0]['previous_error']
    with registry.connect() as con:
        assert con.execute('SELECT COUNT(*) FROM epochs WHERE run_id=?',(same_run,)).fetchone()[0]==1
        assert con.execute('SELECT COUNT(*) FROM checkpoints WHERE run_id=?',(same_run,)).fetchone()[0]==1
    with pytest.raises(ValueError,match='incomplete'):
        train_module.train_ridge_member(cfg,0,candidate,extractor=FakeExtractor(),resume_run_id=same_run)


def test_reference_ridge_solver_uses_same_project_registry(tmp_path,monkeypatch):
    cfg,registry,candidate,_=_synthetic_ridge_fixture(
        tmp_path,monkeypatch,solver='sklearn_reference')
    run_id=train_module.train_ridge_member(cfg,0,candidate,extractor=FakeExtractor())
    assert registry.get_run(run_id)['status']=='completed'
    assert registry.best_validation(run_id)['val_corr_mean']>.95


def test_export_rejects_changed_neuron_identity(tmp_path,monkeypatch):
    cfg,registry,candidate,_=_synthetic_ridge_fixture(tmp_path,monkeypatch)
    run_id=train_module.train_ridge_member(cfg,0,candidate,extractor=FakeExtractor())
    (Path(cfg['data']['root'])/'neuron_index.csv').write_text(
        'animal,neuron_index,neuron_id\nF,0,F/changed\nN,0,N/0\n')
    with pytest.raises(ValueError,match='checkpoint neuron order'):
        export_zwm_bundle(registry.sqlite_path,candidate,tmp_path/'export')
    from transformer_model import evaluate as eval_module
    monkeypatch.setattr(eval_module,'TVSDTrainingBundle',train_module.TVSDTrainingBundle)
    with pytest.raises(ValueError,match='Readout neuron order'):
        eval_module.evaluate_ridge_run(registry.sqlite_path,run_id)


def test_feature_cache_invalidates_when_inputs_change(tmp_path):
    root=tmp_path/'images'; root.mkdir()
    Image.new('RGB',(2,2),(70,70,70)).save(root/'a.png')
    upstream=tmp_path/'ZWM'; checkpoint=upstream/'out'/'test.pt'
    checkpoint.parent.mkdir(parents=True); checkpoint.write_text('source')
    cfg={'model':{'backbone':{'model_ref':'test.pt','repo_root':str(upstream),
                   'layer':'x','pooling':'mean','input_mode':'grayscale_rgb'}},
         'data':{'image_root':str(root)}}
    fake=FakeExtractor()
    idx={'train:1':'a.png'}
    lookup,path=train_module.extract_cached_features(idx,cfg,tmp_path/'cache',fake)
    assert fake.calls==1
    lookup2,path2=train_module.extract_cached_features(idx,cfg,tmp_path/'cache',None)
    assert path==path2 and np.allclose(lookup['train:1'],lookup2['train:1'])
    checkpoint.write_text('changed')
    with pytest.raises(ValueError,match='required'):
        train_module.extract_cached_features(idx,cfg,tmp_path/'cache',None)


def test_upstream_adapter_calls_single_image_forward_and_hooks(tmp_path, monkeypatch):
    """Contract test for external ZWMPredictor, without a real downloaded model."""
    import sys
    import types
    from transformer_model.backbones import ZWMFeatures

    checkpoint=tmp_path/'out'/'stub.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'pretend-weights')
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer=torch.nn.Module()
            self.transformer.h=torch.nn.ModuleList([torch.nn.Identity() for _ in range(23)])
        def forward(self,x):
            return self.transformer.h[22](x)
    class Predictor:
        def __init__(self, checkpoint_name):
            assert checkpoint_name=='stub.pt'
            self.model=Backbone()
            self.calls=0
        def single_image_forward(self, image, frame_gap):
            assert frame_gap==-1
            assert isinstance(image,Image.Image)
            self.calls+=1
            self.model(torch.ones(1,4,3))
    parent=types.ModuleType('zwm'); parent.__path__=[]
    sub=types.ModuleType('zwm.zwm_predictor'); sub.ZWMPredictor=Predictor
    source=tmp_path/'zwm'/'zwm_predictor.py'
    source.parent.mkdir()
    source.write_text('# stub predictor source for provenance validation\n')
    sub.__file__=str(source)
    monkeypatch.setitem(sys.modules,'zwm',parent)
    monkeypatch.setitem(sys.modules,'zwm.zwm_predictor',sub)
    adapter=ZWMFeatures('stub.pt',repo_root=str(tmp_path),layer='transformer.h.22')
    assert adapter.extract(Image.new('RGB',(2,2))).shape==(1,3)
    adapter.hook.count=99  # A previous failed extraction must not poison this one.
    assert adapter.extract(Image.new('RGB',(2,2))).shape==(1,3)
    assert adapter.predictor.calls==2
    adapter.close()


def test_shared_train_entrypoint_uses_separate_python_process(tmp_path,monkeypatch):
    """Manual train and Optuna both call train_member; no ZWM imported in Sinz."""
    import sys
    from sinz_model.trainer import train_member
    from sinz_model import transformer_bridge
    called=[]
    def fake_subprocess(cmd,check):
        assert check
        called.append(cmd)
        resolved=json.loads(Path(cmd[cmd.index('--config')+1]).read_text())
        assert resolved['model']['family']=='transformer_neural'
        Path(cmd[cmd.index('--result-file')+1]).write_text(json.dumps({'run_id':'mock-id'}))
    monkeypatch.setattr(transformer_bridge.subprocess,'run',fake_subprocess)
    monkeypatch.setattr(transformer_bridge, '_already_in_transformer_runtime', lambda cfg: False)
    cfg={'model':{'family':'transformer_neural'},
         'training':{'strategy':'frozen_ridge'},
         'runtime':{'python_executable':sys.executable},
         'project':{'output_root':str(tmp_path)}}
    value=train_member(cfg,0,candidate_id='test-candidate',evaluate_test=False)
    assert value=='mock-id'
    assert called[0][:3]==[sys.executable,'-m','transformer_model.train']
    assert train_member(cfg,0,candidate_id='test-candidate',resume_run_id='abc',
                        evaluate_test=False)=='mock-id'
    assert called[-1][called[-1].index('--resume-run-id')+1]=='abc'


def test_feature_chunks_resume_after_interrupted_extraction(tmp_path):
    images=tmp_path/'images'; images.mkdir()
    for k in range(3):
        Image.new('RGB',(2,2),(40+k*20,)*3).save(images/f'{k}.png')
    upstream=tmp_path/'ZWM'; checkpoint=upstream/'out'/'model.pt'
    checkpoint.parent.mkdir(parents=True); checkpoint.write_bytes(b'checkpoint')
    cfg={'model':{'backbone':{'model_ref':'model.pt','repo_root':str(upstream),
               'layer':'layer','pooling':'mean','input_mode':'rgb','cache_chunk_size':1}},
         'data':{'image_root':str(images)}}
    idx={f'train:{k}':f'{k}.png' for k in range(3)}
    class FailingExtractor(FakeExtractor):
        def extract(self,frame):
            if self.calls==1:
                self.calls+=1
                raise RuntimeError('simulated PBS preemption')
            return super().extract(frame)
    first=FailingExtractor()
    with pytest.raises(RuntimeError,match='preemption'):
        train_module.extract_cached_features(idx,cfg,tmp_path/'cache',first)
    assert first.calls==2
    second=FakeExtractor()
    lookup,_=train_module.extract_cached_features(idx,cfg,tmp_path/'cache',second)
    assert len(lookup)==3
    assert second.calls==2  # first chunk was recovered rather than recomputed
