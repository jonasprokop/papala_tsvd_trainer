"""CUDA-mapping and idempotence tests without a GPU."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from sinz_model.optuna_worker import fix_cuda_visible_devices, execute, _digest
from sinz_model.orchestrator import RunRegistry


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


def test_frozen_ridge_retry_uses_existing_run_without_sgd_checkpoint(tmp_path):
    """Worker orchestration keeps ridge run identity; SGD resume remains separate."""
    from sinz_model import optuna_worker
    root=tmp_path/'experiment'
    trial_dir=root/'trials'/'trial-00007'
    batch_dir=root/'batches'/'batch-001'
    trial_dir.mkdir(parents=True)
    batch_dir.mkdir(parents=True)
    worker_db=tmp_path/'worker.sqlite'
    cfg={'model':{'family':'transformer_neural'},
         'training':{'strategy':'frozen_ridge'},
         'project':{'sqlite_path':str(worker_db),'output_root':str(tmp_path/'runs')}}
    (trial_dir/'config.json').write_text(json.dumps(cfg))
    (batch_dir/'manifest.json').write_text(json.dumps({'records':[{
        'index':1,'number':7,'params':{},'trial_dir':str(trial_dir),
        'container_trial_dir':str(trial_dir),'config_sha256':_digest(cfg)}]}))
    (root/'experiment.json').write_text(json.dumps({
        'name':'test-ridge','prune':False,'train_batches':None,'val_batches':None}))
    registry=RunRegistry(str(worker_db)); registry.initialize()
    workflow=registry.create_workflow('optuna:test-ridge:trial:7')
    candidate=registry.create_candidate(workflow,'optuna',cfg,overrides={})['candidate_id']
    run=registry.start_run(candidate,0,'test',{'root':8000},'dataset','split',[],cfg=cfg)
    registry.finish_run(run['run_id'],'failed',error='PBS preemption')
    assert not (Path(run['run_dir'])/'checkpoints/latest.pt').exists()
    called=[]
    def fake_train(resolved,member,*,candidate_id,resume_run_id,**kwargs):
        called.append((candidate_id,resume_run_id))
        assert resolved['training']['strategy']=='frozen_ridge'
        registry.mark_running(resume_run_id)
        registry.log_epoch(resume_run_id,0,'frozen_ridge',float('nan'),
                           float('nan'),.4,.3,0.)
        registry.finish_run(resume_run_id,'completed')
        return resume_run_id
    class Trial:
        def report(self,value,step): pass
        def should_prune(self): return False
    with patch.object(optuna_worker,'fix_cuda_visible_devices'), \
         patch.object(optuna_worker,'pruning_trial',return_value=(None,Trial())), \
         patch('torch.cuda.is_available',return_value=True), \
         patch('torch.cuda.device_count',return_value=1), \
         patch('torch.cuda.get_device_name',return_value='fake GPU'), \
         patch('sinz_model.trainer.train_member',side_effect=fake_train):
        execute(root,'batch-001',1,worker_db)
    assert called==[(candidate,run['run_id'])]
    assert len(registry.candidate_runs(candidate))==1
    result=json.loads((trial_dir/'result.json').read_text())
    assert result['status']=='completed' and result['run_id']==run['run_id']
    assert result['intermediate_values']=={'0':.4}


def test_retry_pbs_uses_latest_worker_database():
    script=(Path(__file__).resolve().parents[1]/'jobs/optuna_trial.pbs').read_text()
    assert 'SOURCE_DB="$TRIAL_DIR/worker_result.sqlite"' in script
    assert 'cp "$SOURCE_DB" "$SCRATCHDIR/worker.sqlite"' in script
