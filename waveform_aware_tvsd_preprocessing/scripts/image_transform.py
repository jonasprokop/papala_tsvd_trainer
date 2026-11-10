"""Full-field THINGS image preprocessing (no crop, no augmentation).

Resize the entire image with PIL bicubic interpolation to the chosen model size,
convert to float [0,1], optionally convert to grayscale using documented weights.
Grayscale choice is NOT asserted to be the TVSD experimental display transform.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image

WEIGHTS = {
    'gray-bt601':np.array([0.2989,0.5870,0.1140],dtype=np.float32),
    'gray-skimage':np.array([0.2125,0.7154,0.0721],dtype=np.float32), # the scikit-image default, which is the ITU-R BT.709 weights
}
DEFAULT_COLOR_MODE = 'gray-bt601'  # MATLAB rgb2gray / ITU-R BT.601 weights
MODES = (DEFAULT_COLOR_MODE, 'gray-skimage', 'rgb')


def image_hwc(path: Path, image_size: int, mode: str = DEFAULT_COLOR_MODE) -> np.ndarray:
    if mode not in MODES:raise ValueError(f'Mode must be one of {MODES}')
    if int(image_size) < 1:raise ValueError('image_size must be positive')
    with Image.open(path) as im:
        im=im.convert('RGB').resize((int(image_size),int(image_size)),Image.Resampling.BICUBIC)
        rgb=np.asarray(im,dtype=np.float32)/255.0
    if mode=='rgb':return rgb
    return (rgb@WEIGHTS[mode])[...,None].astype(np.float32,copy=False)


def image_chw_normalized(path: Path, *, image_size: int,
                         mean: list[float], std: list[float],
                         mode: str = DEFAULT_COLOR_MODE) -> np.ndarray:
    x=image_hwc(path,image_size,mode)
    m=np.asarray(mean,dtype=np.float32).reshape(1,1,-1)
    s=np.asarray(std,dtype=np.float32).reshape(1,1,-1)
    if (m.shape[-1]!=x.shape[-1] or len(s.reshape(-1))!=x.shape[-1]
            or not np.isfinite(m).all() or not np.isfinite(s).all() or (s<=0).any()):
        raise ValueError('Normalization must match mode channels and be finite/positive')
    x=(x-m)/s
    if x.shape[-1]==1: x=np.repeat(x,3,axis=-1)
    return np.ascontiguousarray(x.transpose(2,0,1),dtype=np.float32)
