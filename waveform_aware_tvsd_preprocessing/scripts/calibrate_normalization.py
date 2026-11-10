#!/usr/bin/env python3
"""Fit dataset-wide train-only image mean/std, Sinz-style; do not fit val/test.

Each DISTINCT training image contributes once. Calculated after exactly the same
full-image bicubic resize/color transform applied by the loader.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from image_transform import MODES,WEIGHTS,DEFAULT_COLOR_MODE,image_hwc
from dataset import _safe_image, _split_sha256

DEFAULT = Path('/project/data/processed/tvsd_waveform_aware_v1/model_dataset')

def calibrate(dataset_root: Path, image_root: Path | None = None, *,
              image_size: int = 224, mode: str = DEFAULT_COLOR_MODE,
              max_images: int | None = None, output: Path | None = None) -> dict:
    dataset_root=Path(dataset_root)
    if mode not in MODES:raise ValueError(f'Bad mode {mode}')
    if max_images is not None and max_images<1:raise ValueError('max_images must be positive')
    production_path = (dataset_root / 'normalization.json').resolve()
    if max_images is not None:
        if output is None:
            raise ValueError('--max-images is diagnostic only; pass an explicit --output '
                             'so production normalization.json cannot be overwritten')
        if Path(output).resolve() == production_path:
            raise ValueError('Refusing to overwrite production normalization.json '
                             'with subset statistics')
    info=json.loads((dataset_root/'dataset_info.json').read_text())
    image_root=Path(image_root) if image_root is not None else Path(info['image_root'])
    sample=pd.read_csv(dataset_root/'sample_index.csv')
    split_sha256 = _split_sha256(sample)
    if not info.get('split_sha256') or split_sha256 != info['split_sha256']:
        raise ValueError('Dataset split fingerprint mismatch; rebuild or repair prepared dataset')
    train=(sample.loc[sample.partition.eq('train'),['stimulus_id','image_relative_path']]
           .drop_duplicates('stimulus_id').sort_values('stimulus_id'))
    if train.stimulus_id.empty:raise ValueError('No train images')
    if max_images is not None:
        # Stable random sample (not the first lexicographic IDs), diagnostic only.
        take = np.sort(np.random.default_rng(8000).choice(
            len(train), size=min(max_images, len(train)), replace=False))
        train = train.iloc[take]
    n_channels=3 if mode=='rgb' else 1
    sums=np.zeros(n_channels,dtype=np.float64)
    squares=np.zeros(n_channels,dtype=np.float64)
    n=0
    for i,rel in enumerate(train.image_relative_path,1):
        x=image_hwc(_safe_image(image_root,rel),image_size,mode).astype(np.float64)
        sums+=x.sum(axis=(0,1)); squares+=(x*x).sum(axis=(0,1))
        n+=x.shape[0]*x.shape[1]
        if i%2000==0:print('Normalization:',i,'/',len(train),flush=True)
    mean=sums/n;var=np.maximum(0,squares/n-mean*mean);std=np.sqrt(var)
    if not np.isfinite(std).all() or (std<=0).any():raise ValueError('Invalid std')
    payload={'method':'global_pixel_zscore_train_unique_images','color_mode':mode,
             'grayscale_weights':WEIGHTS[mode].astype(float).tolist() if mode!='rgb' else None,
             'image_size':image_size,'mean':mean.tolist(),'std':std.tolist(),
             'n_images':len(train),'n_pixels_per_channel':n,
             'partial':max_images is not None, 'split_sha256':split_sha256,
             'geometry':'full-image-bicubic-resize-no-crop'}
    output=Path(output) if output else dataset_root/'normalization.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(payload,indent=2)+'\n')
    print('Saved',output,'mean',payload['mean'],'std',payload['std'])
    return payload

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,default=DEFAULT)
    p.add_argument('--images-root',type=Path)
    p.add_argument('--image-size',type=int,default=224)
    p.add_argument('--color-mode',choices=MODES,default=DEFAULT_COLOR_MODE)
    p.add_argument('--max-images',type=int,help='Diagnostic random subset; requires --output')
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    calibrate(a.dataset_root,a.images_root,image_size=a.image_size,mode=a.color_mode,
              max_images=a.max_images,output=a.output)
if __name__=='__main__':main()
