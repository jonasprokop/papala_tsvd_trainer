"""Explicit post-selection test evaluation for a frozen transformer candidate.

No training path calls this automatically. Uses the project's own test metric
(correlation to repeated-image average) and writes the existing registry.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sinz_model.analysis import correlation_to_average, summarize_correlations
from sinz_model.orchestrator import RunRegistry
from sinz_model.tvsd_data import TVSDTrainingBundle
from .ridge import RidgeReadout
from .train import _images_index, _neuron_order, extract_cached_features


def evaluate_ridge_run(sqlite_path, run_id, *, extractor=None):
    registry=RunRegistry(sqlite_path)
    run=registry.get_run(run_id)
    if run['status']!='completed':
        raise ValueError('Cannot evaluate incomplete ridge run')
    cfg=json.loads(run['config_json'])
    if cfg.get('model',{}).get('family')!='transformer_neural':
        raise ValueError('Not a transformer neural run')
    ckpt=Path(registry.best_checkpoint(run_id))
    state=torch.load(ckpt,map_location='cpu',weights_only=True)
    params=state['readout_state_dict']
    readout=RidgeReadout(params['weight'],params['bias'],params['feature_mean'],params['feature_scale'])
    bundle=TVSDTrainingBundle(cfg)
    if state['neuron_ids'] != _neuron_order(cfg,bundle):
        raise ValueError('Readout neuron order differs from TVSD dataset')
    valid_mask=np.asarray(state['valid_neuron_mask'],dtype=bool)
    if valid_mask.shape != (len(state['neuron_ids']),):
        raise ValueError('Invalid fitted-neuron mask in checkpoint')
    if extractor is None:
        from .backbones import ZWMFeatures
        bc=cfg['model']['backbone']
        extractor=ZWMFeatures(bc['model_ref'],repo_root=bc['repo_root'],
                              layer=bc['layer'],pooling=bc.get('pooling','mean'))
    try:
        lookup,_=extract_cached_features(_images_index(bundle.test.rows),cfg,
                      Path(cfg['project']['feature_cache_root']),extractor)
        features=np.stack([lookup[s] for s in bundle.test.rows.stimulus_id])
        with torch.inference_mode():
            prediction=readout(torch.from_numpy(features)).numpy()
        corr,_=correlation_to_average(prediction,bundle.test.responses,
                                      bundle.test.mask,bundle.test.image_ids)
        corr[~valid_mask] = np.nan
        summary=summarize_correlations(corr)
        if summary['n_valid']==0:
            raise ValueError('No valid test correlations')
        out=Path(run['run_dir'])/'evaluation_recheck'
        out.mkdir(parents=True,exist_ok=True)
        report=out/'transformer_test.json'
        report.write_text(json.dumps({'summary':summary, 'run_id':run_id,
                'valid_neurons':summary['n_valid'],
                'test_metric':'corr_to_average'},indent=2)+'\n')
        for idx,value in enumerate(corr):
            if np.isfinite(value):
                registry.log_metric(run_id,'test','corr_to_average','neuron',float(value),neuron_id=idx)
        registry.log_metric(run_id,'test','corr_to_average','mean',summary['mean'])
        registry.log_metric(run_id,'test','corr_to_average','median',summary['median'])
        registry.log_artifact(run_id,'test_evaluation',str(report))
        print(json.dumps(summary,sort_keys=True),flush=True)
        return summary
    finally:
        if hasattr(extractor,'close'):
            extractor.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--sqlite-path',required=True)
    p.add_argument('--run-id',required=True)
    p.add_argument('--result-file',required=True)
    args=p.parse_args()
    result=evaluate_ridge_run(args.sqlite_path,args.run_id)
    Path(args.result_file).write_text(json.dumps({'summary':result})+'\n')

if __name__=='__main__':
    main()
