#!/usr/bin/env python3
"""Measure grayscale mean/std on frozen TRAIN images only.
Usage from server:
  python scripts/calibrate_tvsd_normalization.py
For a fast integration test, --max-images 64 writes a distinct smoke-test file.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset-root',type=Path,default=Path('/project/data/processed/ks4_v1/model_dataset_v1'))
    p.add_argument('--images-root',type=Path,default=Path('/project/data/assets/THINGS'))
    p.add_argument('--image-size',type=int,default=224)
    p.add_argument('--max-images',type=int,default=None)
    p.add_argument('--output',type=Path,default=None)
    args=p.parse_args()
    if args.max_images is not None and args.max_images<1:
        p.error('--max-images must be positive')
    df=pd.read_csv(args.dataset_root/'sample_index.csv')
    frame=df.loc[df.partition.eq('train'),['stimulus_id','image_relative_path']].drop_duplicates('stimulus_id')
    if frame.stimulus_id.nunique()!=len(frame): raise ValueError('Duplicate train stimulus ID')
    frame=frame.sort_values('stimulus_id')
    if args.max_images is not None: frame=frame.iloc[:args.max_images]
    n=0; total=0.; total2=0.
    for k, rel in enumerate(frame.image_relative_path,1):
        q=Path(rel)
        if q.is_absolute() or '..' in q.parts: raise ValueError('Unsafe image path')
        path=args.images_root/q
        with Image.open(path) as im:
            im=im.convert('RGB').resize((args.image_size,args.image_size),Image.Resampling.BICUBIC)
            pixels=np.asarray(im,dtype=np.float64)/255.
        gray=pixels[:,:,0]*0.2989+pixels[:,:,1]*0.5870+pixels[:,:,2]*0.1140
        total+=float(gray.sum()); total2+=float(np.square(gray).sum()); n+=gray.size
        if k%2000==0: print('Measured',k,'/',len(frame),flush=True)
    mean=total/n; std=float(np.sqrt(max(0,total2/n-mean*mean)))
    if not np.isfinite(std) or std <= 0: raise ValueError('Invalid grayscale variance')
    payload={'mean':mean,'std':std,'n_images':len(frame),'n_pixels':n,'input_hw':[args.image_size]*2,
             'preprocessing':'rgb_resize_bicubic_then_luma','source_partition':'train',
             'max_images':args.max_images}
    if args.output is None:
        out=args.dataset_root/('tvsd_normalization.json' if args.max_images is None else 'tvsd_normalization_smoke.json')
    else: out=args.output
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    print('Saved',out,'mean',round(mean,6),'std',round(std,6),'N',len(frame))

if __name__=='__main__': main()
