"""One image: verify ZWM checkpoint, upstream inference path, and selected layer."""
from __future__ import annotations
import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
from sinz_model.runtime import load_config
from .backbones import ZWMFeatures


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--image',required=True,help='Path to one real TVSD THINGS image')
    args=parser.parse_args()
    cfg=load_config(args.config)
    bc=cfg['model']['backbone']
    if bc.get('forward_mode','one_image')!='one_image':
        raise ValueError('Unsupported inference mode')
    extractor=ZWMFeatures(bc['model_ref'],repo_root=bc['repo_root'],
                          layer=bc['layer'],pooling=bc.get('pooling','mean'))
    try:
        with Image.open(args.image) as src:
            img=src.convert('RGB')
        if bc.get('input_mode','grayscale_rgb')=='grayscale_rgb':
            img=ImageOps.grayscale(img).convert('RGB')
        out=extractor.extract(img)
        repeat=extractor.extract(img)
        if out.ndim != 2 or out.shape[0] != 1 or out.shape[1] == 0:
            raise RuntimeError(f'Unexpected ZWM feature shape: {out.shape}')
        if not np.isfinite(out).all() or not np.isfinite(repeat).all():
            raise RuntimeError('Non-finite ZWM features')
        if not np.allclose(out, repeat, rtol=1e-3, atol=1e-4):
            raise RuntimeError('Repeated ZWM inference is not numerically stable')
        print(json.dumps({'model_ref':bc['model_ref'],'layer':bc['layer'],
                          'feature_shape':list(out.shape),'mode':'one_image',
                          'finite':True,'repeat_close':True}))
    finally:
        extractor.close()

if __name__=='__main__':
    main()
