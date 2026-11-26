"""Papale V1 MUA / ZWM full-token ridge probe, without feature reduction.

Uses every 1024 x 768 ZWM token coordinate and every one of Papale's 22,248
training stimuli. The expensive Gram matrix and LOOCV are computed on a large
CUDA GPU, preserving the *sklearn RidgeCV* per-target alpha selection used by
BBScore's RidgeMetric(mode='sklearn', subsample_features_for_alpha=None).

The released BBScore TVSDAssembly and scoring functions are loaded verbatim
from a pinned local checkout. The original published experiment configuration
and RNG state are not public, so this is an independently reproducible
implementation of the public method, NOT an asserted reproduction of 0.95.

Phases: init -> extract (sharded) -> gram (H100 NVL) -> fit (H100 NVL) -> report.
The checkpointed raw spatial chunks are FLOAT32, preserving all bfloat16 ZWM
outputs without rounding or reduction.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from sinz_model.runtime import load_config, sha256_file
from sinz_model.tvsd_data import TVSDTrainingBundle
from .benchmark_probe import BB_SCORE_ALPHAS
from .benchmark_v1 import LAYERS, _atomic_json, _atomic_npz
from .train import _images_index, _safe_image


ORIGINAL_SCORE_SEED = 42  # Explicit; paper's run-time RNG state is unavailable.
DEFAULT_GPU_FEATURE_BLOCK = 4096
MIN_GPU_HEADROOM_GIB = 8
SPATIAL_WIDTH = 1024*768


def _source_function(path, function, *, extras=None):
    """Compile ONLY the verbatim original function (no unrelated imports)."""
    tree = ast.parse(Path(path).read_text(encoding='utf-8'), filename=str(path))
    matches = [node for node in tree.body if
               isinstance(node, ast.FunctionDef) and node.name == function]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one {function} in {path}')
    from scipy.stats import pearsonr
    namespace = {'np': np, 'pearsonr': pearsonr}
    namespace.update(extras or {})
    code = ast.fix_missing_locations(ast.Module(body=matches, type_ignores=[]))
    exec(compile(code, str(path), 'exec'), namespace)
    return namespace[function]


def _source_paths(bbscore_root, mua_root):
    b = Path(bbscore_root)
    m = Path(mua_root)
    return dict(ridge=b/'metrics/ridge.py', scorer=b/'metrics/utils.py',
                split_half=b/'data/utils.py',
                f_mua=m/'monkeyF/THINGS_normMUA.mat',
                n_mua=m/'monkeyN/THINGS_normMUA.mat')


def _protocol(cfg, bundle, *, layer_index, chunk_size, bbscore_root, mua_root):
    bc = cfg['model']['backbone']
    if (bundle.info.get('version') != 'tvsd_papale_normmua_v1'
            or bundle.widths != {'F':512,'N':512}
            or bc.get('kind')!='zwm' or bc.get('input_mode')!='rgb'
            or bc.get('model_ref')!='awwkl/zwm-bvd-170m/model.pt'):
        raise ValueError('Requires original 1024-channel Papale V1 MUA and RGB ZWM-170M')
    if not 0<=layer_index<len(LAYERS) or chunk_size<1:
        raise ValueError('Invalid layer or cache chunk size')
    train_ids = sorted(set(bundle.full.rows.stimulus_id))
    test_ids = sorted(set(bundle.test.rows.stimulus_id))
    if len(train_ids)!=22248 or len(test_ids)!=100 or set(train_ids)&set(test_ids):
        raise ValueError('Unexpected original Papale image identities')
    index = _images_index(bundle.df)
    images = {}
    for stim in train_ids+test_ids:
        rel = index[stim]
        stat = _safe_image(Path(cfg['data']['image_root']),rel).stat()
        images[stim] = [rel,stat.st_size,stat.st_mtime_ns]
    paths = _source_paths(bbscore_root, mua_root)
    for key, file in paths.items():
        if not file.is_file():
            raise FileNotFoundError(f'Missing pinned original {key}: {file}')
    model = Path(bc['repo_root'])/'out'/bc['model_ref']
    predictor = Path(bc['repo_root'])/'zwm/zwm_predictor.py'
    base = dict(schema='zwm-papale-mua-exact-full-tokens-v1',
                source_dataset=bundle.info['version'],
                dataset_fingerprint=bundle.fingerprint,
                layer=LAYERS[layer_index],layer_index=layer_index,
                train_ids=train_ids,test_ids=test_ids,images=images,
                width=SPATIAL_WIDTH,chunk_size=chunk_size,
                checkpoint_sha256=sha256_file(str(model)),
                predictor_sha256=sha256_file(str(predictor)),
                alpha_grid=list(BB_SCORE_ALPHAS),source_config=cfg,
                source_paths={key:str(path) for key,path in paths.items()},
                source_sha256={key:sha256_file(str(path)) for key,path in paths.items()},
                note='No pooling, dimensionality reduction, subsampling, or outer K-fold; '
                     'sklearn RidgeCV GCV LOOCV equivalent; source BBScore scorer')
    # The adapter records the source MATLAB hashes: reject a silently changed source.
    declared = bundle.info.get('frozen_input_sha256', {})
    for animal, prefix in (('F','f'), ('N','n')):
        declared_digest = declared.get(f'{animal}_mua')
        if declared_digest and declared_digest != base['source_sha256'][f'{prefix}_mua']:
            raise ValueError(f'{animal} MUA release differs from imported frozen dataset')
    return base


def _chunk_spec(protocol):
    ids = protocol['train_ids']+protocol['test_ids']
    width = protocol['chunk_size']
    for start in range(0,len(ids),width):
        yield start,ids[start:start+width]


def _read_chunk(path, expected, width, *, dtype=np.float32):
    if not path.is_file():
        raise FileNotFoundError(f'Missing spatial chunk: {path}')
    with np.load(path,allow_pickle=False) as source:
        ids = source['stimulus_ids'].astype(str).tolist()
        features = source['features']
    if (ids!=expected or features.shape!=(len(expected),width)
            or features.dtype!=np.dtype(dtype) or not np.isfinite(features).all()):
        raise ValueError(f'Corrupt or mismatched original spatial chunk: {path}')
    return features


def _extract(cfg,protocol,out,*,shards,shard_index):
    from .backbones import ZWMFeatures

    if shards<1 or not 0<=shard_index<shards:
        raise ValueError('Invalid extraction shard')
    folder = out/'spatial_chunks'
    folder.mkdir(parents=True,exist_ok=True)
    model = None
    try:
        for number,(start,ids) in enumerate(_chunk_spec(protocol)):
            if number%shards!=shard_index:
                continue
            path = folder/f'{start:07d}.npz'
            if path.exists():
                _read_chunk(path,ids,protocol['width'])
                continue
            if model is None:
                bc=cfg['model']['backbone']
                model=ZWMFeatures(bc['model_ref'],repo_root=bc['repo_root'],
                                  layer=protocol['layer'],pooling='flatten')
            features=[]
            for stimulus in ids:
                relative=protocol['images'][stimulus][0]
                with Image.open(_safe_image(Path(cfg['data']['image_root']),relative)) as image:
                    # This calls the original ZWM RGB resize -> center crop ->
                    # ImageNet transform, and captures all spatial tokens.
                    array=np.asarray(model.extract(image.convert('RGB')),dtype=np.float32)
                if (array.shape!=(1,protocol['width']) or
                        not np.isfinite(array).all()):
                    raise ValueError(f'Invalid full spatial activation: {stimulus}: {array.shape}')
                features.append(array[0])
            _atomic_npz(path,stimulus_ids=np.asarray(ids),features=np.stack(features))
            print(f'EXTRACT full spatial: shard={shard_index}, '
                  f'{start+len(ids)}/{len(protocol["train_ids"])+len(protocol["test_ids"])}',
                  flush=True)
    finally:
        if model is not None:
            model.close()


def _manifest(out, proposed, *, create):
    path = out/'protocol.json'
    if not path.exists():
        if not create:
            raise FileNotFoundError('Run --phase init first')
        _atomic_json(path, proposed)
    previous = json.loads(path.read_text())
    if previous != json.loads(json.dumps(proposed)):
        raise ValueError('Frozen protocol mismatch; use a fresh --output directory')
    return previous


def _all_chunks(protocol, out):
    for start, ids in _chunk_spec(protocol):
        yield start, ids, out/'spatial_chunks'/f'{start:07d}.npz'


def _atomic_npy(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix='.'+path.name+'.', suffix='.npy',
                                     dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
        try:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    os.replace(tmp, path)


def _gram(protocol, out, *, block_width):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError('Full spatial Gram needs an NVIDIA CUDA GPU')
    if block_width < 128 or block_width > protocol['width']:
        raise ValueError('Invalid feature block width')
    done = out/'exact/gram_manifest.json'
    if done.exists():
        metadata = json.loads(done.read_text())
        if (metadata.get('protocol_sha256') != sha256_file(str(out/'protocol.json'))
                or metadata['n_train'] != len(protocol['train_ids'])
                or metadata['width'] != protocol['width']):
            raise ValueError('Stale full-kernel checkpoint')
        for key, shape in (('train', (metadata['n_train'],)*2),
                           ('test', (len(protocol['test_ids']),metadata['n_train']))):
            source = np.load(out/'exact'/f'gram_{key}.npy', mmap_mode='r')
            if source.shape != shape or source.dtype != np.float64:
                raise ValueError(f'Corrupt gram_{key} checkpoint')
        print('Restored completed exact full-feature Gram', flush=True)
        return

    # Refuse a partial feature extraction before claiming an expensive H100.
    for _, ids, path in _all_chunks(protocol, out):
        _read_chunk(path, ids, protocol['width'], dtype=np.float32)
    n_train, n_test = len(protocol['train_ids']),len(protocol['test_ids'])
    d, n = protocol['width'], n_train+n_test
    device = torch.device('cuda:0')
    free, total = torch.cuda.mem_get_info(device)
    features_bytes = n * d * 4
    # Float64 kernel, eigensolver/allocator allowance, and one fp64 feature tile.
    additional_bytes = 2*n_train*n_train*8 + 2*n_train*block_width*8
    required = features_bytes + additional_bytes + MIN_GPU_HEADROOM_GIB*2**30
    if free < required:
        raise MemoryError(f'Full-feature GPU fit needs {required/2**30:.1f} GiB '
                          f'free (currently {free/2**30:.1f} of {total/2**30:.1f} GiB); '
                          'use an H100 NVL 94GB, and do not run another GPU task there')
    print(f'Gram: N={n_train} train, {n_test} test, D={d}; '
          f'GPU free={free/2**30:.1f} GiB; float64 blocked kernel', flush=True)
    x = torch.empty((n,d), device=device, dtype=torch.float32)
    for start, ids, path in _all_chunks(protocol, out):
        features = _read_chunk(path, ids, d, dtype=np.float32)
        x[start:start+len(ids)] = torch.from_numpy(features).to(
            device=device, dtype=torch.float32)
        if start % (protocol['chunk_size']*100) == 0:
            print(f'Loaded full features {start+len(ids)}/{n}', flush=True)

    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        k_train = torch.zeros((n_train,n_train),device=device,dtype=torch.float64)
        k_test = torch.zeros((n_test,n_train),device=device,dtype=torch.float64)
        for offset in range(0,d,block_width):
            stop = min(offset+block_width,d)
            train = x[:n_train,offset:stop].double()
            test = x[n_train:,offset:stop].double()
            mean = train.mean(dim=0,keepdim=True)
            train.sub_(mean)
            test.sub_(mean)
            k_train.addmm_(train,train.T)
            k_test.addmm_(test,train.T)
            del train, test, mean
            if offset // block_width % 10 == 0:
                print(f'Full float64 Gram: {stop}/{d} dimensions', flush=True)
        del x
        torch.cuda.synchronize()
        if not torch.isfinite(k_train).all() or not torch.isfinite(k_test).all():
            raise FloatingPointError('Nonfinite Gram matrix')
        # Save only after *all* blocks complete, atomically and independently.
        folder = out/'exact'
        _atomic_npy(folder/'gram_train.npy', k_train.cpu().numpy())
        _atomic_npy(folder/'gram_test.npy', k_test.cpu().numpy())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    _atomic_json(done, dict(protocol_sha256=sha256_file(str(out/'protocol.json')),
                            n_train=n_train, n_test=n_test, width=d,
                            dtype='float64', block_width=block_width,
                            method='centered full spatial Gram, no reduction'))
    print('Completed full Gram checkpoint', flush=True)


def _target_matrix(protocol, *, mua_root):
    """Read original released DOUBLE arrays; preserve their original test means."""
    import h5py

    train_ids, test_ids = protocol['train_ids'], protocol['test_ids']
    if (set(train_ids) != {f'train:{i+1}' for i in range(22248)}
            or set(test_ids) != {f'test:{i+1}' for i in range(100)}):
        raise ValueError('Unexpected Papale stimulus IDs; cannot align original MATLAB rows')
    tr_idx = np.asarray([int(i.split(':')[1])-1 for i in train_ids])
    te_idx = np.asarray([int(i.split(':')[1])-1 for i in test_ids])
    y = np.empty((len(tr_idx),1024),dtype=np.float64)
    test_mean = np.empty((len(te_idx),1024),dtype=np.float64)
    repeated = np.empty((30,len(te_idx),1024),dtype=np.float64)
    for animal,(lo,hi) in {'F':(0,512),'N':(512,1024)}.items():
        path = Path(mua_root)/f'monkey{animal}/THINGS_normMUA.mat'
        with h5py.File(path,'r',driver='stdio') as f:
            train = np.asarray(f['train_MUA'][:,0:512],dtype=np.float64)
            test = np.asarray(f['test_MUA'][:,0:512],dtype=np.float64)
            reps = np.asarray(f['test_MUA_reps'][:,:,0:512],dtype=np.float64)
        if (train.shape!=(22248,512) or test.shape!=(100,512)
                or reps.shape!=(30,100,512)):
            raise ValueError(f'Unexpected original monkey {animal} MUA shapes')
        y[:,lo:hi] = train[tr_idx]
        test_mean[:,lo:hi] = test[te_idx]
        repeated[:,:,lo:hi] = reps[:,te_idx]
    if not (np.isfinite(y).all() and np.isfinite(test_mean).all()
            and np.isfinite(repeated).all()):
        raise ValueError('Nonfinite original MUA entries')
    if not np.allclose(repeated.mean(axis=0),test_mean,rtol=3e-4,atol=3e-4):
        raise ValueError('Released test mean is not consistent with test repetitions')
    return y,test_mean,repeated


def _loocv_select_numpy(eigenvalues, eigenvectors, targets, alphas):
    """Reference NumPy formula for RidgeCV(gcv_mode='eigen', alpha_per_target=True).

    Centered kernel includes a zero eigendirection for the intercept. Residual
    operator Q_alpha = alpha*(K+alpha I)^-1 - 11^T/n. LOO residual_i =
    (Q_alpha y)_i / (Q_alpha)_ii. Return one alpha per target.
    """
    eigenvalues = np.maximum(eigenvalues,0.)
    centered = targets-targets.mean(axis=0)
    uy = eigenvectors.T@centered
    u2 = eigenvectors**2
    errors = []
    for a in alphas:
        q = a/(eigenvalues+a)
        diag = u2@q-1/len(targets)
        loo = (eigenvectors@(q[:,None]*uy))/diag[:,None]
        errors.append((loo**2).mean(axis=0))
    return np.asarray(alphas)[np.asarray(errors).argmin(axis=0)]


def _loocv_select_gpu(eigval,eigvec,targets,alphas):
    import torch

    n,p = targets.shape
    uy = eigvec.T @ (targets-targets.mean(dim=0,keepdim=True))
    u2 = eigvec.square()
    mse = []
    for a in alphas:
        weight = float(a)/(eigval+float(a))
        diag = u2@weight - 1.0/n
        # A nonpositive influence diagonal means the LOO estimate is unstable.
        if bool((diag <= 1.e-14).any()):
            mse.append(torch.full((p,),float('inf'),device=targets.device,
                                  dtype=torch.float64))
            continue
        residual = (eigvec@(weight[:,None]*uy)) / diag[:,None]
        err = residual.square().mean(dim=0)
        mse.append(torch.where(torch.isfinite(err),err,float('inf')))
    scores = torch.stack(mse)
    if bool(torch.isinf(scores).all(dim=0).any()):
        raise FloatingPointError('LOOCV failed for at least one target')
    indices = scores.argmin(dim=0)
    selected = torch.as_tensor(alphas,dtype=torch.float64,
                               device=targets.device)[indices]
    return selected, uy, scores


def _score_original(protocol, *, pred, targets, reps, source_paths):
    """Apply verbatim public BBScore split-half and Pearson metric functions."""
    half = _source_function(source_paths['split_half'],'split_half_consistency')
    pearson = _source_function(source_paths['scorer'],'pearson_correlation_scorer')
    # TVSDAssembly.prepare_data computes the same split half *twice*: once
    # while preparing train data, and again while preparing test data.
    # The upstream RNG is global/unseeded; using 42 is a documented choice.
    rng = np.random.RandomState(ORIGINAL_SCORE_SEED)
    reliability = None
    for _ in (True,False):
        reliabilities = [half(reps[:,:,lo:hi],n_splits=50,rng=rng)
                         for lo,hi in ((0,512),(512,1024))]
        reliability = np.concatenate(reliabilities)
    raw = np.asarray([np.corrcoef(targets[:,i],pred[:,i])[0,1]
                      for i in range(1024)])
    bbscore = np.asarray([pearson(targets[:,i],pred[:,i])
                          for i in range(1024)])
    if not (np.isfinite(reliability).all() and np.isfinite(bbscore).all()):
        raise FloatingPointError('Original BBScore scorer or ceiling is nonfinite')
    # Exactly BaseMetric.__init__: replace all ceilings <=1e-6 by 1e-6.
    floor = np.maximum(reliability,1.e-6)
    corrected = bbscore/floor
    return dict(raw_pearson=raw, bbscore_pearson=bbscore,
                split_half_reliability=reliability,noise_ceiling=floor,
                noise_corrected_pearson=corrected,
                n_repeated_images=np.full(1024,100,dtype=np.int32))


def _fit(protocol, out, *, mua_root):
    import torch

    folder = out/'exact'
    status = json.loads((folder/'gram_manifest.json').read_text())
    if status['protocol_sha256'] != sha256_file(str(out/'protocol.json')):
        raise ValueError('Full Gram protocol mismatch')
    if not torch.cuda.is_available():
        raise RuntimeError('Full spatial eigendecomposition needs CUDA')
    device = torch.device('cuda:0')
    print('Reading original Papale MUA without neural renormalization',flush=True)
    y_np,test_np,reps = _target_matrix(protocol,mua_root=mua_root)
    k_np = np.load(folder/'gram_train.npy',mmap_mode='r')
    cross_np = np.load(folder/'gram_test.npy',mmap_mode='r')
    n = len(protocol['train_ids'])
    if (k_np.shape != (n,n) or k_np.dtype != np.float64
            or cross_np.shape != (len(protocol['test_ids']),n)
            or cross_np.dtype != np.float64):
        raise ValueError('Invalid full Gram shapes/dtypes')
    print(f'GPU eigen decomposition of {n} x {n} DOUBLE kernel',flush=True)
    gram = torch.from_numpy(np.array(k_np)).to(device)
    # eigval clamps numerical zero only; report any significant negative modes.
    vals,vec = torch.linalg.eigh(gram)
    del gram
    lowest,highest = float(vals[0].item()),float(vals[-1].item())
    if lowest < -max(1.e-5,1.e-8*highest):
        raise FloatingPointError(f'Kernel has significant negative eigenvalue: {lowest}')
    vals.clamp_(min=0.)
    y = torch.from_numpy(y_np).to(device)
    alphas = tuple(float(a) for a in protocol['alpha_grid'])
    selected,uy,_ = _loocv_select_gpu(vals,vec,y,alphas)
    center = y.mean(dim=0,keepdim=True)
    coefficients = vec @ (uy/(vals[:,None]+selected[None,:]))
    del y,uy
    cross = torch.from_numpy(np.array(cross_np)).to(device)
    pred = (cross@coefficients+center).cpu().numpy()
    chosen = selected.cpu().numpy()
    if not np.isfinite(pred).all():
        raise FloatingPointError('Nonfinite full-space prediction')
    original = _score_original(protocol,pred=pred,targets=test_np,reps=reps,
                               source_paths=protocol['source_paths'])
    _atomic_npz(folder/'predictions.npz',prediction=pred,response=test_np,
                response_repetitions=reps,alpha_per_channel=chosen,
                stimulus_ids=np.asarray(protocol['test_ids']),**original)
    median = float(np.median(original['noise_corrected_pearson']))
    result = dict(schema=protocol['schema'],layer=protocol['layer'],
                  n_train_images=n, n_test_images=len(protocol['test_ids']),
                  spatial_features=protocol['width'],reduction='none',
                  ridge='Full-data sklearn RidgeCV per-target GCV-LOOCV equivalent',
                  alpha_grid=alphas, alpha_counts={str(float(a)):int((chosen==a).sum())
                                                   for a in alphas},
                  original_bbscore_median=median,
                  raw_pearson_median=float(np.median(original['raw_pearson'])),
                  n_channels=1024,ceiling_floor_channels=int((original['split_half_reliability']<=1.e-6).sum()),
                  score_rng_seed=ORIGINAL_SCORE_SEED,
                  published_reference=0.95,
                  caveat='Original private published config and RNG seed unavailable; '
                         'this reproduces the public BBScore algorithm with explicit RNG seed')
    _atomic_json(folder/'summary.json',result)
    print(json.dumps(result,indent=2),flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase',choices=('init','extract','gram','fit','report'),required=True)
    ap.add_argument('--config',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--bbscore-root',required=True)
    ap.add_argument('--mua-root',required=True)
    ap.add_argument('--layer-index',type=int,default=3)
    ap.add_argument('--chunk-size',type=int,default=16)
    ap.add_argument('--block-width',type=int,default=DEFAULT_GPU_FEATURE_BLOCK)
    ap.add_argument('--shards',type=int,default=1)
    ap.add_argument('--shard-index',type=int,default=0)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    bundle = TVSDTrainingBundle(cfg)
    out = Path(args.output)
    proposed = _protocol(cfg,bundle,layer_index=args.layer_index,
                         chunk_size=args.chunk_size,bbscore_root=args.bbscore_root,
                         mua_root=args.mua_root)
    protocol = _manifest(out,proposed,create=args.phase=='init')
    if args.phase=='init':
        print(f'Frozen EXACT MUA: {len(protocol["train_ids"])} train, '
              f'{len(protocol["test_ids"])} test, {protocol["width"]} full features')
    elif args.phase=='extract':
        _extract(cfg,protocol,out,shards=args.shards,shard_index=args.shard_index)
    elif args.phase=='gram':
        _gram(protocol,out,block_width=args.block_width)
    elif args.phase=='fit':
        _fit(protocol,out,mua_root=args.mua_root)
    else:
        print((out/'exact/summary.json').read_text())


if __name__=='__main__':
    main()
