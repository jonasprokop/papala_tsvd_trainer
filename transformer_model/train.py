"""Frozen ZWM -> masked TVSD ridge, using the existing RunRegistry.

This is an independent Python-process entrypoint. Run it in ZWM's Python 3.10
runtime; Sinz's Python 3.9 and optimizer implementation are unaffected.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

from sinz_model.analysis import per_neuron_masked_pearson, summarize_correlations
from sinz_model.orchestrator import RunRegistry
from sinz_model.tvsd_data import TVSDTrainingBundle
from sinz_model.runtime import config_sha256, sha256_file
from .ridge import fit_masked_ridge_cv, fit_masked_ridge_cv_reference


def _hash_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',',':')).encode()).hexdigest()


def _images_index(df):
    """One image path per unique stimulus, independent of session/animal."""
    paths = {}
    for stimulus, rel in zip(df.stimulus_id, df.image_relative_path):
        rel = str(rel)
        if stimulus in paths and paths[stimulus] != rel:
            raise ValueError(f'Conflicting paths for stimulus {stimulus}')
        paths[stimulus] = rel
    return dict(sorted(paths.items()))


def _safe_image(root:Path, relative:str):
    rel = Path(relative)
    if rel.is_absolute() or not rel.parts or '..' in rel.parts:
        raise ValueError(f'Unsafe image path: {relative}')
    path = (root/rel).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Image escapes image root: {relative}')
    return path


def extract_cached_features(index, cfg, cache_dir:Path, extractor):
    """Unique-image feature cache; fully validated before reuse.

    Caches features only (never responses), keyed by model/input/split identity.
    Neither val nor test observations participate in readout fitting.
    """
    bc = cfg['model']['backbone']
    root=Path(cfg['data']['image_root'])
    file_stats={}
    for stimulus, relative in index.items():
        path=_safe_image(root,relative)
        st=path.stat()
        file_stats[stimulus]=(st.st_size,st.st_mtime_ns)
    spec = {
        'source_files':file_stats, 'extractor_version':1,
        'cache_chunk_size':int(bc.get('cache_chunk_size',256)),
        'model_ref':bc['model_ref'], 'checkpoint_sha256':sha256_file(str(Path(bc['repo_root'])/'out'/bc['model_ref'])),
        'layer':bc['layer'], 'pooling':bc.get('pooling','mean'), 'forward_mode':'one_image',
        'input_mode':bc.get('input_mode','grayscale_rgb'), 'image_root':str(Path(cfg['data']['image_root']).resolve()),
        'images':index,
    }
    key = _hash_json(spec)
    cache_dir.mkdir(parents=True,exist_ok=True)
    cache_path = cache_dir / f'{key}.npz'
    stimuli = list(index)
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as dat:
            ids = dat['stimulus_ids'].astype(str).tolist()
            vectors = dat['features']
            if ids != stimuli or vectors.ndim != 2 or not np.isfinite(vectors).all():
                raise RuntimeError('Feature cache corrupt or stale')
            return dict(zip(ids, vectors)), cache_path
    if extractor is None:
        raise ValueError('Feature extractor required on cache miss')
    # Persist completed chunks separately; a PBS timeout does not discard hours
    # of inference. A final consolidated NPZ is written only after all chunks.
    chunk_size=int(bc.get('cache_chunk_size',256))
    if chunk_size < 1:
        raise ValueError('cache_chunk_size must be positive')
    part_dir=cache_dir/key
    part_dir.mkdir(exist_ok=True)
    vectors=[]
    for start in range(0,len(stimuli),chunk_size):
        names=stimuli[start:start+chunk_size]
        part_path=part_dir/f'{start:07d}.npz'
        if part_path.is_file():
            with np.load(part_path,allow_pickle=False) as part:
                ids=part['stimulus_ids'].astype(str).tolist()
                subset=part['features']
                if (ids!=names or subset.ndim!=2 or len(subset)!=len(names) or
                        not np.isfinite(subset).all()):
                    raise RuntimeError(f'Corrupt feature chunk {part_path}')
            if vectors and subset.shape[1]!=vectors[0].shape[0]:
                raise RuntimeError('Feature width changed between cached chunks')
            vectors.extend(subset)
            continue
        batch=[]
        for stimulus in names:
            with Image.open(_safe_image(root,index[stimulus])) as file:
                frame=file.convert('RGB')
                if bc.get('input_mode','grayscale_rgb')=='grayscale_rgb':
                    frame=ImageOps.grayscale(frame).convert('RGB')
                elif bc.get('input_mode')!='rgb':
                    raise ValueError('input_mode must be rgb or grayscale_rgb')
                features=np.asarray(extractor.extract(frame),dtype=np.float32)
            if features.ndim!=2 or features.shape[0]!=1 or not np.isfinite(features).all():
                raise ValueError(f'Bad backbone features for {stimulus}')
            if (batch and features.shape[1]!=batch[0].shape[0]) or (vectors and features.shape[1]!=vectors[0].shape[0]):
                raise ValueError(f'Feature width changed at {stimulus}')
            batch.append(features[0])
        subset=np.stack(batch)
        with tempfile.NamedTemporaryFile(dir=part_dir,suffix='.npz',delete=False) as handle:
            temp=Path(handle.name)
            try:
                np.savez_compressed(handle,stimulus_ids=np.asarray(names,dtype=str),features=subset)
            except BaseException:
                temp.unlink(missing_ok=True)
                raise
        temp.replace(part_path)
        vectors.extend(subset)
        print(f'Extracted or restored {len(vectors)}/{len(stimuli)} images',flush=True)
    matrix=np.stack(vectors)
    with tempfile.NamedTemporaryFile(dir=cache_dir,suffix='.npz',delete=False) as handle:
        tmp=Path(handle.name)
        try:
            np.savez_compressed(handle,stimulus_ids=np.asarray(stimuli,dtype=str),features=matrix)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    tmp.replace(cache_path)
    return dict(zip(stimuli,matrix)), cache_path


def _neuron_order(cfg, bundle):
    df=pd.read_csv(Path(cfg['data']['root'])/'neuron_index.csv')
    ids=[]
    for animal in ('F','N'):
        rows=df.loc[df.animal.eq(animal)].sort_values('neuron_index')
        if rows.neuron_index.astype(int).tolist()!=list(range(bundle.widths[animal])):
            raise ValueError(f'Neuron index is not contiguous for {animal}')
        ids.extend(rows.neuron_id.astype(str).tolist())
    if len(ids)!=cfg['model']['readout']['outdims']:
        raise ValueError('Neurons do not match model readout')
    if len(set(ids))!=len(ids) or any(not v or v=='nan' for v in ids):
        raise ValueError('Neuron IDs must be present and globally unique')
    return ids


def _restart_ridge_run(reg, run_id, candidate_id, cfg, bundle):
    """Retry a failed/preempted ridge fit in-place, retaining its provenance.

    Unlike SGD, ridge has no mid-epoch checkpoint to resume. Cached backbone
    features survive and make re-fitting cheap. The unique registry run row and
    its original git, environment and dataset records must not be replaced.
    """
    previous=reg.get_run(str(run_id))
    if (previous['candidate_id']!=str(candidate_id) or
            int(previous['member_index'])!=0):
        raise ValueError('Ridge retry candidate/member mismatch')
    if previous['status'] not in ('running','failed'):
        raise ValueError('Only incomplete ridge runs can be retried')
    if (previous['config_sha256']!=config_sha256(cfg) or
            previous['dataset_fingerprint']!=bundle.fingerprint or
            previous['split_sha256']!=bundle.split.sha256):
        raise ValueError('Ridge retry refused: config, dataset or split changed')
    run_dir=Path(previous['run_dir'])
    if not run_dir.is_dir():
        raise FileNotFoundError(f'Ridge run directory missing: {run_dir}')
    history_path=run_dir/'ridge_retry_history.json'
    history=json.loads(history_path.read_text()) if history_path.exists() else []
    if not isinstance(history,list):
        raise ValueError('Invalid ridge retry history')
    history.append({'retried_at_utc':datetime.now(timezone.utc).isoformat(),
                    'previous_status':previous['status'], 'previous_error':previous['error'],
                    'command':' '.join(sys.argv)})
    pending=history_path.with_name(history_path.name+f'.tmp.{os.getpid()}')
    try:
        pending.write_text(json.dumps(history,indent=2)+'\n')
        pending.replace(history_path)
    finally:
        pending.unlink(missing_ok=True)
    # Incomplete fit metrics/checkpoint rows cannot leak into this attempt.
    # This leaves the original run, parameters, provenance and cached features.
    with reg.connect() as con:
        for table in ('epochs','metrics','checkpoints','artifacts'):
            con.execute(f'DELETE FROM {table} WHERE run_id=?',(run_id,))
        con.execute('''UPDATE runs SET status='running', finished_at_utc=NULL,
                       error=NULL WHERE run_id=?''',(run_id,))
        con.execute('''UPDATE candidates SET status='running', finished_at_utc=NULL
                       WHERE candidate_id=?''',(candidate_id,))
    reg.log_artifact(run_id,'ridge_retry_history',str(history_path))
    return {'run_id':str(run_id), 'run_dir':str(run_dir)}


def train_ridge_member(cfg, member_index, candidate_id, *, evaluate_test=False, extractor=None,
                       resume_run_id=None):
    if cfg['model'].get('family')!='transformer_neural':
        raise ValueError('Incorrect model family for frozen transformer training')
    if cfg.get('training',{}).get('strategy')!='frozen_ridge':
        raise NotImplementedError('Only training.strategy=frozen_ridge is implemented')
    if int(member_index)!=0:
        raise ValueError('Frozen ridge currently has one deterministic member: member_index=0')
    if evaluate_test:
        raise NotImplementedError('Test evaluation must be an explicit post-selection operation')
    bundle = TVSDTrainingBundle(cfg)
    neuron_ids=_neuron_order(cfg,bundle)
    reg=RunRegistry(cfg['project']['sqlite_path'])
    reg.initialize()
    bc=cfg['model']['backbone']
    if bc.get('forward_mode','one_image')!='one_image':
        raise NotImplementedError('Only upstream ZWM one_image inference is supported')
    if bc.get('kind')!='zwm':
        raise NotImplementedError('Only ZWM is integrated; V-JEPA2 requires its own image adapter')
    if resume_run_id is None:
        run=reg.start_run(candidate_id,0,' '.join(sys.argv),
                          {'root':int(cfg.get('training',{}).get('seed',8000))},
                          bundle.fingerprint,bundle.split.sha256,bundle.mapping,cfg=cfg,
                          notes='frozen ZWM features; per-neuron masked ridge')
    else:
        run=_restart_ridge_run(reg,resume_run_id,candidate_id,cfg,bundle)
    run_id=run['run_id']; run_dir=Path(run['run_dir'])
    try:
        if extractor is None:
            from .backbones import ZWMFeatures
            extractor=ZWMFeatures(bc['model_ref'],repo_root=bc['repo_root'],
                                  layer=bc['layer'],pooling=bc.get('pooling','mean'))
        # No test images or responses: fit and validate on the frozen train/val only.
        unique = _images_index(bundle.full.rows)
        cache_base=Path(cfg['project'].get('feature_cache_root', str(run_dir/'features')))
        lookup, cache_path=extract_cached_features(unique,cfg,cache_base,extractor)
        df=bundle.full.rows
        X=np.stack([lookup[s] for s in df.stimulus_id])
        train_idx=np.flatnonzero(df.partition.eq('train').to_numpy())
        val_idx=np.flatnonzero(df.partition.eq('val').to_numpy())
        if not len(train_idx) or not len(val_idx):
            raise ValueError('Frozen TVSD partitions are empty')
        if set(df.stimulus_id.iloc[train_idx]) & set(df.stimulus_id.iloc[val_idx]):
            raise ValueError('Image identity leakage across train and val')
        rc=cfg['ridge']
        solver=rc.get('solver','spectral')
        if solver not in ('spectral','sklearn_reference'):
            raise ValueError('ridge.solver must be spectral or sklearn_reference')
        fit_fn=(fit_masked_ridge_cv if solver=='spectral' else fit_masked_ridge_cv_reference)
        fit=fit_fn(X,bundle.full.responses,bundle.full.mask,
                                train_indices=train_idx,groups=df.stimulus_id.to_numpy(),
                                alphas=tuple(rc['alphas']),cv_folds=int(rc['cv_folds']))
        with torch.inference_mode():
            prediction=fit.readout(torch.from_numpy(X[val_idx])).cpu().numpy()
        corr=per_neuron_masked_pearson(prediction,bundle.full.responses[val_idx],bundle.full.mask[val_idx])
        summary=summarize_correlations(corr)
        if summary['n_valid']<=0 or not np.isfinite(summary['mean']):
            raise ValueError('No valid validation correlations')
        cv_errors=-fit.cv_scores[np.isfinite(fit.cv_scores)]
        if len(cv_errors):
            reg.log_metric(run_id,'train','ridge_cv_mse','mean',float(cv_errors.mean()))
        reg.log_epoch(run_id,0,'frozen_ridge',float('nan'),
                      float('nan'),summary['mean'],summary['median'],0.)
        for neuron_index,value in enumerate(corr):
            if np.isfinite(value):
                reg.log_metric(run_id,'val','pearson','neuron',float(value),neuron_id=neuron_index)
        reg.log_metric(run_id,'val','pearson','mean',summary['mean'])
        reg.log_metric(run_id,'val','pearson','median',summary['median'])
        # Fitted is distinct from a neuron with a finite validation score.
        # Unfitted neurons retain NaN intercepts and must be masked downstream.
        state={'readout_state_dict':fit.readout.state_dict(),
               'neuron_ids':neuron_ids,'alpha_by_neuron':torch.as_tensor(fit.alphas,dtype=torch.float32),
               'cv_score_by_neuron':torch.as_tensor(fit.cv_scores,dtype=torch.float32),'valid_train_counts':torch.as_tensor(fit.valid_train_counts,dtype=torch.int64),
               'valid_neuron_mask':torch.as_tensor(fit.valid_neuron_mask,dtype=torch.bool),
               'feature_cache_key':cache_path.stem,'validation':summary}
        checkpoint_dir=run_dir/'checkpoints'; checkpoint_dir.mkdir(exist_ok=True)
        checkpoint=checkpoint_dir/'best.pt'
        torch.save(state,checkpoint)
        from sinz_model.runtime import hash_state_dict
        reg.log_checkpoint(run_id,0,'frozen_ridge',str(checkpoint),
                           sha256_file(str(checkpoint)),hash_state_dict(fit.readout.state_dict()),True)
        reg.log_artifact(run_id,'feature_cache',str(cache_path))
        reg.log_artifact(run_id,'readout',str(checkpoint))
        with (run_dir/'validation.json').open('w') as f:
            json.dump({'summary':summary,'neuron_ids':neuron_ids,'alpha':[float(v) if np.isfinite(v) else None for v in fit.alphas],
                       'cv_score':[float(v) if np.isfinite(v) else None for v in fit.cv_scores],
                       'valid_train_counts':fit.valid_train_counts.tolist(),
                       'valid_neuron_mask':fit.valid_neuron_mask.tolist()},
                      f,indent=2,allow_nan=False)
        reg.finish_run(run_id,'completed')
        print('TVSD_TRANSFORMER_RUN_ID='+run_id,flush=True)
        return run_id
    except BaseException:
        reg.finish_run(run_id,'failed',error=traceback.format_exc())
        raise
    finally:
        if extractor is not None and hasattr(extractor,'close'):
            extractor.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True,help='Resolved execution config JSON')
    p.add_argument('--candidate-id',required=True)
    p.add_argument('--member',type=int,default=0)
    p.add_argument('--resume-run-id',default=None)
    p.add_argument('--result-file',required=True)
    p.add_argument('--evaluate-test',action='store_true')
    a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text())
    run_id=train_ridge_member(cfg,a.member,a.candidate_id,evaluate_test=a.evaluate_test,
                              resume_run_id=a.resume_run_id)
    dest=Path(a.result_file)
    dest.write_text(json.dumps({'run_id':run_id})+'\n')

if __name__=='__main__':
    main()
