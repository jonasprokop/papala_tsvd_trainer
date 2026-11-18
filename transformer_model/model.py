"""Frozen transformer readout: generic toy adapter and portable ZWM handoff."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from .features import FeatureHook,pool_tokens
from .ridge import RidgeReadout


class TransformerNeuralModel(nn.Module):
    """Generic hook model used for tests and alternative pretrained backbones."""
    def __init__(self, backbone, readout, *, forward_image, layer, pooling='mean'):
        super().__init__()
        self.backbone=backbone.requires_grad_(False).eval()
        self.readout=readout
        self.forward_image=forward_image
        self.layer=layer
        self.pooling=pooling
        self._hook=FeatureHook(backbone,layer)
    def train(self,mode=True):
        super().train(mode)
        self.backbone.eval()
        return self
    def features(self, images):
        self._hook.activation=None
        self._hook.count=0
        with torch.no_grad():
            self.forward_image(self.backbone,images)
            return pool_tokens(self._hook.take(),self.pooling)
    def forward(self,images):
        return self.readout(self.features(images))
    @torch.inference_mode()
    def predict(self,images):
        return self(images)
    def close(self):
        self._hook.close()


def export_model(model, output_dir, *, metadata):
    """Prototype generic exporter retained for toy models; no pickle of classes."""
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    config=dict(metadata)
    config.update({'layer':model.layer,'pooling':model.pooling,
                   'readout_features':model.readout.weight.shape[1],
                   'num_neurons':model.readout.weight.shape[0],'format_version':1})
    (out/'manifest.json').write_text(json.dumps(config,indent=2,default=str))
    torch.save(model.backbone.state_dict(),out/'backbone_state.pt')
    torch.save(model.readout.state_dict(),out/'readout_state.pt')
    return out


def load_export(path, *, build_backbone, forward_image, map_location='cpu'):
    p=Path(path); manifest=json.loads((p/'manifest.json').read_text())
    backbone=build_backbone(manifest)
    backbone.load_state_dict(torch.load(p/'backbone_state.pt',map_location=map_location,weights_only=True))
    state=torch.load(p/'readout_state.pt',map_location=map_location,weights_only=True)
    readout=RidgeReadout(state['weight'],state['bias'],state['feature_mean'],state['feature_scale'])
    model=TransformerNeuralModel(backbone,readout,forward_image=forward_image,
                                 layer=manifest['layer'],pooling=manifest['pooling'])
    return model,manifest


def _sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for buf in iter(lambda:stream.read(1024*1024),b''):
            digest.update(buf)
    return digest.hexdigest()


def export_zwm_bundle(sqlite_path, candidate_id, output_dir):
    """Self-contained data/weights bundle for a selected frozen ZWM candidate.

    Inference still requires upstream ZWM source and a compatible Python runtime,
    but needs no training database, image bank, or downloaded checkpoint.
    """
    from sinz_model.orchestrator import RunRegistry
    reg=RunRegistry(sqlite_path)
    candidate=reg.get_candidate(candidate_id)
    cfg=json.loads(candidate['config_json'])
    if cfg.get('model',{}).get('family')!='transformer_neural':
        raise ValueError('Not a transformer neural candidate')
    runs=reg.candidate_runs(candidate_id)
    if len(runs)!=1 or runs[0]['status']!='completed':
        raise ValueError('Export requires one completed ridge run')
    run=runs[0]
    checkpoint=Path(run['run_dir'])/'checkpoints/best.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    from .train import _neuron_order
    widths=cfg['model']['readout']['outdims_by_animal']
    if state['neuron_ids'] != _neuron_order(cfg,SimpleNamespace(widths=widths)):
        raise ValueError('Cannot export: checkpoint neuron order differs from neuron_index.csv')
    valid_mask=state['valid_neuron_mask'].to(dtype=torch.bool)
    if (valid_mask.shape!=(len(state['neuron_ids']),) or
            not torch.equal(valid_mask,torch.isfinite(state['readout_state_dict']['bias']))):
        raise ValueError('Invalid fitted-neuron mask in checkpoint')
    bc=cfg['model']['backbone']
    source=Path(bc['repo_root'])/'out'/bc['model_ref']
    if not source.is_file():
        raise FileNotFoundError(f'Upstream checkpoint not found: {source}')
    dest=Path(output_dir)
    if dest.exists():
        raise FileExistsError(dest)
    dest.mkdir(parents=True)
    try:
        shutil.copyfile(source,dest/'model.pt')
        # Duplicate ordered IDs and mask in the readout payload, so reload
        # can detect a mismatched manifest even without the training dataset.
        torch.save({**state['readout_state_dict'],
                    'neuron_ids':list(state['neuron_ids']),
                    'valid_neuron_mask':valid_mask},dest/'readout_state.pt')
        manifest={
            'format':'tvsd_zwm_ridge_v1',
            'backbone_kind':'zwm', 'upstream_ref':bc['model_ref'],
            'checkpoint_sha256':_sha256(dest/'model.pt'),
            'readout_sha256':_sha256(dest/'readout_state.pt'),
            'forward_mode':'one_image', 'frame_gap':-1,
            'layer':bc['layer'], 'pooling':bc.get('pooling','mean'),
            'input_mode':bc.get('input_mode','grayscale_rgb'),
            'neuron_ids':state['neuron_ids'],
            'valid_neuron_mask':valid_mask.tolist(),
            'num_neurons':len(state['neuron_ids']),
            'candidate_id':candidate_id, 'run_id':run['run_id'],
            'split_sha256':run['split_sha256'], 'dataset_fingerprint':run['dataset_fingerprint'],
            'validation':state['validation'],
            'feature_scaling':'per_neuron_train_fold_zscore; coefficients converted to raw features',
        }
        (dest/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    except Exception:
        shutil.rmtree(dest,ignore_errors=True)
        raise
    return dest


class ZWMNeuralModel(nn.Module):
    """Inference-only readout; expects PIL images (not Sinz-normalized tensors)."""
    def __init__(self, extractor, readout, manifest):
        super().__init__()
        self.extractor=extractor
        self.readout=readout
        self.manifest=manifest
    @torch.inference_mode()
    def predict(self, image):
        from PIL import ImageOps
        images=image if isinstance(image,(list,tuple)) else [image]
        features=[]
        for img in images:
            rgb=img.convert('RGB')
            if self.manifest['input_mode']=='grayscale_rgb':
                rgb=ImageOps.grayscale(rgb).convert('RGB')
            arr=np.asarray(self.extractor.extract(rgb),dtype=np.float32)
            if arr.ndim!=2 or arr.shape[0]!=1:
                raise ValueError('Unexpected ZWM feature shape')
            features.append(arr[0])
        return self.readout(torch.from_numpy(np.stack(features)))
    def close(self):
        self.extractor.close()


def load_zwm_bundle(path, *, zwm_repo_root, expected_neuron_ids=None):
    """Load packaged checkpoint through upstream predictor in the new ZWM env.

    A stable symlink under ZWM's out/tvsd_bundles allows its native ModelFactory
    to find the bundle without copying another 170M checkpoint into the source.
    """
    from .backbones import ZWMFeatures
    path=Path(path).resolve()
    manifest=json.loads((path/'manifest.json').read_text())
    if manifest['format']!='tvsd_zwm_ridge_v1':
        raise ValueError('Unrecognized model bundle')
    if _sha256(path/'model.pt')!=manifest['checkpoint_sha256'] or _sha256(path/'readout_state.pt')!=manifest['readout_sha256']:
        raise ValueError('Bundle checksum mismatch')
    state=torch.load(path/'readout_state.pt',map_location='cpu',weights_only=True)
    neuron_ids=manifest['neuron_ids']
    valid_mask=manifest['valid_neuron_mask']
    if (len(neuron_ids)!=manifest['num_neurons'] or
            any(not isinstance(n,str) or not n for n in neuron_ids) or
            len(set(neuron_ids))!=len(neuron_ids) or
            state['neuron_ids']!=neuron_ids):
        raise ValueError('Bundle neuron identities/order mismatch')
    if (len(valid_mask)!=len(neuron_ids) or
            any(type(v) is not bool for v in valid_mask) or
            state['valid_neuron_mask'].tolist()!=valid_mask or
            torch.isfinite(state['bias']).tolist()!=valid_mask):
        raise ValueError('Bundle fitted-neuron mask mismatch')
    if expected_neuron_ids is not None and list(expected_neuron_ids)!=neuron_ids:
        raise ValueError('Bundle neuron order differs from expected dataset')
    readout=RidgeReadout(state['weight'],state['bias'],state['feature_mean'],state['feature_scale'])
    if readout.weight.shape[0]!=manifest['num_neurons']:
        raise ValueError('Neuron count mismatch')
    ref=Path('tvsd_bundles')/manifest['checkpoint_sha256']/'model.pt'
    target=Path(zwm_repo_root).resolve()/'out'/ref
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        target.symlink_to(path/'model.pt')
    if target.resolve()!= (path/'model.pt').resolve():
        raise FileExistsError(f'Different model at {target}')
    extractor=ZWMFeatures(str(ref),repo_root=str(zwm_repo_root),
                          layer=manifest['layer'],pooling=manifest['pooling'])
    return ZWMNeuralModel(extractor,readout,manifest)
