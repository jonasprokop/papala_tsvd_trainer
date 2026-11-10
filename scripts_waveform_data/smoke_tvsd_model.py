#!/usr/bin/env python3
"""GPU/CPU smoke test: real THINGS images -> pretrained core -> one Gaussian readout -> masked Poisson."""
import argparse
import math
from torch.utils.data._utils.collate import default_collate
import torch
from sinz_model.runtime import load_config
from sinz_model.model import TVSDV1Model
from sinz_model.tvsd_data import TVSDTrainingBundle
from sinz_model.analysis import training_loss
from sinz_model.trainer import _initialize_readout_bias_from_targets, _configure_model_for_phase, _optimizer


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/tvsd_default.yaml')
    p.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args=p.parse_args()
    cfg=load_config(args.config)
    device=torch.device(args.device)
    print('Loading frozen TVSD data ...',flush=True)
    bundle=TVSDTrainingBundle(cfg)
    train=bundle.full.subset_by_image_ids(bundle.split.train_image_ids)
    ds=bundle.dataset('train')
    # One presentation from each animal. No global readout columns overlap.
    n_f=len(ds.datasets[0])
    batch=default_collate([ds[0],ds[n_f]])
    x=batch['image'].to(device)
    target=batch['response'].to(device)
    mask=batch['mask'].to(device)
    assert mask[0,:bundle.widths['F']].any() and not mask[0,bundle.widths['F']:].any()
    assert mask[1,bundle.widths['F']:].any() and not mask[1,:bundle.widths['F']].any()
    print('Input:',tuple(x.shape),'target:',tuple(target.shape),
          'mask entries:',int(mask.sum()),'mean/std:',bundle.mean,bundle.std,flush=True)
    print('Loading pretrained ConvNeXtV2-Atto ...',flush=True)
    model=TVSDV1Model(cfg).to(device)
    print('Core:',model.core.output_shape, 'Gaussian readout:',model.readout.outdims, 'F/N:',bundle.widths,flush=True)
    _initialize_readout_bias_from_targets(model,train.responses,train.mask)
    for phase,lr in [('readout_only',cfg['training']['phase1_lr']),
                     ('joint_finetune',cfg['training']['phase2_lr'])]:
        _configure_model_for_phase(model,phase,cfg)
        optimizer=_optimizer(model,cfg,lr)
        optimizer.zero_grad(set_to_none=True)
        prediction=model(x)
        loss=training_loss(model,prediction,target,mask,
              gamma_readout=float(cfg['training']['loss']['gamma_readout']),
              dataset_size=len(train.image_ids),batch_size=len(x),
              eps=float(cfg['training']['loss']['eps']))
        if not bool(torch.isfinite(loss['loss']).item()) or not bool(torch.all(prediction>0).item()):
            raise RuntimeError('Nonfinite loss or nonpositive output')
        loss['loss'].backward()
        assert model.readout._features.grad is not None
        assert model.readout._features.grad.shape[-1] == sum(bundle.widths.values())
        if phase=='joint_finetune':
            assert any(p.grad is not None for p in model.core.parameters() if p.requires_grad)
        optimizer.step()
        print('PASS',phase,'loss:',float(loss['loss'].detach().cpu()),flush=True)
    print('TVSD MODEL SMOKE PASS: one Gaussian readout, frozen neuron mapping, both training phases')

if __name__=='__main__':main()
