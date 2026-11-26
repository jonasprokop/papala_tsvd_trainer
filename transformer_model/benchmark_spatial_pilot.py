"""Bounded ZWM MUA pilot: mean tokens vs all spatial tokens.

Uses the *unmodified* BBScore TorchRidgeCV class from its source checkout.
This is a pooling diagnostic, NOT the exact published BBScore sklearn protocol:
TorchRidgeCV selects one alpha by holdout validation, not per-target LOOCV.
No existing feature bank, layer result, or project registry is modified.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from sinz_model.runtime import load_config, sha256_file
from sinz_model.tvsd_data import TVSDTrainingBundle
from .benchmark_probe import BB_SCORE_ALPHAS
from .benchmark_stats import (bbscore_floor_corrected_scores,
                              finite_summary, score_repeated_stimuli)
from .benchmark_v1 import LAYERS, _atomic_json, _atomic_npz
from .train import _images_index, _safe_image


DEFAULT_LAYER = 3  # transformer.h.6.ln_1: best existing MUA layer
FEATURE_DIM = 768  # ZWM-170M ln_1 hidden width


def _load_original_torch_ridge(utils_path):
    """Execute only the upstream class definition; BBScore imports JEPA at top level.

    This uses BBScore's own source bytes, without maintaining a derivative
    implementation or importing unrelated neural backbones from metrics/utils.py.
    Pin its source SHA-256 in the pilot protocol.
    """
    import torch

    source = Path(utils_path)
    module = ast.parse(source.read_text(encoding='utf-8'), filename=str(source))
    matches = [node for node in module.body
               if isinstance(node, ast.ClassDef) and node.name == 'TorchRidgeCV']
    if len(matches) != 1:
        raise RuntimeError('Expected exactly one BBScore TorchRidgeCV definition')
    # Current upstream class depends only on torch, numpy and typing.List.
    namespace = dict(torch=torch, np=np, List=List)
    unit = ast.fix_missing_locations(ast.Module(body=matches, type_ignores=[]))
    exec(compile(unit, str(source), 'exec'), namespace)
    model = namespace['TorchRidgeCV']
    if not all(hasattr(model, attr) for attr in ('fit', 'predict')):
        raise RuntimeError('Unexpected BBScore TorchRidgeCV interface')
    return model


def _plan(cfg, bundle, n_train, seed, layer_index, utils_path, chunk_size):
    if bundle.info.get('version') != 'tvsd_papale_normmua_v1':
        raise ValueError('Spatial pilot is exclusively for original Papale normalized MUA')
    if bundle.widths != {'F': 512, 'N': 512}:
        raise ValueError('Expected 512 F + 512 N V1 MUA channels')
    bc = cfg['model']['backbone']
    if (bc['kind'], bc['model_ref'], bc['input_mode']) != (
            'zwm', 'awwkl/zwm-bvd-170m/model.pt', 'rgb'):
        raise ValueError('Expected published ZWM-170M with original RGB images')
    if not 0 <= layer_index < len(LAYERS):
        raise ValueError('Invalid layer index')
    if n_train < 32 or n_train > 2048:
        raise ValueError('Pilot requires 32..2048 training images (full run needs another design)')
    index = _images_index(bundle.df)
    train_ids = sorted(set(bundle.full.rows.stimulus_id))
    test_ids = sorted(set(bundle.test.rows.stimulus_id))
    if n_train > len(train_ids) or set(train_ids) & set(test_ids):
        raise ValueError('Insufficient training stimuli or train/test leakage')
    rng = np.random.default_rng(seed)
    selected = sorted(rng.choice(train_ids, n_train, replace=False).tolist())
    paths = {}
    for stimulus in selected + test_ids:
        source = _safe_image(Path(cfg['data']['image_root']), index[stimulus])
        stat = source.stat()
        paths[stimulus] = [index[stimulus], stat.st_size, stat.st_mtime_ns]
    model_file = Path(bc['repo_root']) / 'out' / bc['model_ref']
    predictor_file = Path(bc['repo_root']) / 'zwm' / 'zwm_predictor.py'
    utils_path = Path(utils_path)
    if not utils_path.is_file():
        raise FileNotFoundError('Clone BBScore and pass --bbscore-utils metrics/utils.py')
    protocol = dict(schema='zwm-spatial-pilot-v1', source_dataset=bundle.info['version'],
                    dataset_fingerprint=bundle.fingerprint,
                    layer=LAYERS[layer_index], layer_index=layer_index,
                    train_ids=selected, test_ids=test_ids, image_sources=paths,
                    feature_dim=FEATURE_DIM, seed=seed, n_train=n_train,
                    chunk_size=chunk_size,
                    source_config=cfg,
                    weights_sha256=sha256_file(str(model_file)),
                    predictor_sha256=sha256_file(str(predictor_file)),
                    bbscore_utils_sha256=sha256_file(str(utils_path)),
                    bbscore_source=str(utils_path.resolve()),
                    ridge_method='unmodified BBScore metrics/utils.py TorchRidgeCV; '
                                 'one alpha chosen by random 10% holdout, not sklearn per-target GCV',
                    alphas=list(BB_SCORE_ALPHAS))
    return protocol


def _prepare_protocol(path, value):
    path = Path(path)
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != json.loads(json.dumps(value)):
            raise ValueError('Existing spatial pilot has different parameters/source files; use a NEW output directory')
    else:
        _atomic_json(path, value)


def _feature_chunks(out, protocol, cfg, *, extract):
    names = protocol['train_ids'] + protocol['test_ids']
    chunk_size = int(protocol['chunk_size'])
    feature_dir = out / 'spatial_features'
    if extract:
        feature_dir.mkdir(parents=True, exist_ok=True)
    extractor = None
    try:
        for start in range(0, len(names), chunk_size):
            expected = names[start:start + chunk_size]
            path = feature_dir / f'{start:06d}.npz'
            if path.exists():
                with np.load(path, allow_pickle=False) as data:
                    got = data['stimulus_ids'].astype(str).tolist()
                    vectors = np.asarray(data['features'], dtype=np.float32)
                if (got != expected or vectors.ndim != 2 or
                        vectors.shape[0] != len(got) or vectors.shape[1] % FEATURE_DIM or
                        not np.isfinite(vectors).all()):
                    raise ValueError(f'Stale or malformed spatial features: {path}')
            elif not extract:
                raise FileNotFoundError(f'Missing features {path}; run extract first')
            else:
                if extractor is None:
                    from .backbones import ZWMFeatures
                    bc = cfg['model']['backbone']
                    extractor = ZWMFeatures(bc['model_ref'], repo_root=bc['repo_root'],
                                            layer=protocol['layer'], pooling='flatten')
                vectors = []
                for stimulus in expected:
                    relative = protocol['image_sources'][stimulus][0]
                    with Image.open(_safe_image(Path(cfg['data']['image_root']), relative)) as image:
                        vector = extractor.extract(image.convert('RGB'))
                    if vector.shape[0] != 1 or vector.shape[1] % FEATURE_DIM:
                        raise ValueError(f'Unexpected spatial ZWM activation shape {vector.shape}')
                    vectors.append(np.asarray(vector[0], dtype=np.float32))
                vectors = np.stack(vectors)
                if not np.isfinite(vectors).all():
                    raise ValueError(f'Nonfinite ZWM activations for {expected}')
                _atomic_npz(path, stimulus_ids=np.asarray(expected), features=vectors)
                print(f'Spatial activations: {start + len(expected)}/{len(names)}', flush=True)
            yield expected, vectors
    finally:
        if extractor is not None:
            extractor.close()


def _load_features(out, protocol, cfg):
    names = protocol['train_ids'] + protocol['test_ids']
    features = None
    position = 0
    for expected, block in _feature_chunks(out, protocol, cfg, extract=False):
        if features is None:
            print(f'Flattened ZWM width: {block.shape[1]} '
                  f'({block.shape[1] // FEATURE_DIM} tokens x {FEATURE_DIM})', flush=True)
            features = np.empty((len(names), block.shape[1]), dtype=np.float32)
        if block.shape[1] != features.shape[1]:
            raise ValueError('Inconsistent ZWM feature widths across chunks')
        features[position:position + len(expected)] = block
        position += len(expected)
    if position != len(names):
        raise AssertionError('Missing pilot features')
    return features[:protocol['n_train']], features[protocol['n_train']:]


def _score_summary(pooling, alpha, scoring, n_train):
    floor = bbscore_floor_corrected_scores(scoring['bbscore_pearson'],
                                            scoring['split_half_reliability'])
    return dict(pooling=pooling, chosen_alpha=alpha, n_train_images=n_train,
                raw=finite_summary(scoring['raw_pearson']),
                corrected=finite_summary(scoring['noise_corrected_pearson']),
                bbscore_floor=finite_summary(floor))


def _fit_one_pooling(model_class, name, x_train, x_test, bundle, protocol):
    """Fit F/N independently (their response masks have disjoint columns)."""
    import torch

    selected = set(protocol['train_ids'])
    ordered_train = {name: i for i, name in enumerate(protocol['train_ids'])}
    test_ids = {name: i for i, name in enumerate(protocol['test_ids'])}
    prediction = np.zeros((len(protocol['test_ids']), 1024), dtype=np.float32)
    alpha = {}
    for animal, (lo, hi) in {'F': (0, 512), 'N': (512, 1024)}.items():
        rows = bundle.full.rows
        chosen = np.flatnonzero(rows.animal.eq(animal).to_numpy() &
                                    rows.stimulus_id.isin(selected).to_numpy())
        stimuli = rows.iloc[chosen].stimulus_id.to_numpy()
        if (len(stimuli) != len(selected) or len(set(stimuli)) != len(stimuli) or
                set(stimuli) != selected):
            raise ValueError(f'MUA pilot requires exactly one {animal} response per selected image')
        observed = bundle.full.mask[np.ix_(chosen, np.arange(lo, hi))]
        if not observed.all():
            raise ValueError(f'Missing {animal} MUA responses in selected images')
        targets = bundle.full.responses[np.ix_(chosen, np.arange(lo, hi))]
        inputs = x_train[[ordered_train[s] for s in stimuli]]
        # The BBScore class draws its own held-out split; make the pilot repeatable.
        torch.manual_seed(protocol['seed'])
        fit = model_class(alphas=list(BB_SCORE_ALPHAS), device='cuda')
        print(f'Fitting {name}/{animal}: X={inputs.shape}, Y={targets.shape}', flush=True)
        fit.fit(inputs, targets)
        predicted = np.asarray(fit.predict(x_test), dtype=np.float32)
        if predicted.shape != (len(test_ids), 512) or not np.isfinite(predicted).all():
            raise ValueError(f'Invalid BBScore {name}/{animal} predictions')
        prediction[:, lo:hi] = predicted
        alpha[animal] = float(fit.best_alpha)
        del fit
        torch.cuda.empty_cache()
    test = bundle.test
    test_prediction = prediction[[test_ids[s] for s in test.rows.stimulus_id]]
    scoring = score_repeated_stimuli(test_prediction, test.responses, test.mask,
                                     test.image_ids, splits=50, seed=42)
    summary = _score_summary(name, alpha, scoring, len(selected))
    return summary, test_prediction, scoring


def _run_fit(out, protocol, cfg, bundle):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError('BBScore TorchRidgeCV pilot requires GPU with CUDA')
    ridge_class = _load_original_torch_ridge(protocol['bbscore_source'])
    flat_train, flat_test = _load_features(out, protocol, cfg)
    n_train = len(flat_train)
    width = flat_train.shape[1]
    # Extra GPU copies and coefficient recovery require headroom beyond the raw arrays.
    free, _ = torch.cuda.mem_get_info()
    if (n_train + len(flat_test)) * width * 4 * 3 > free:
        raise MemoryError('GPU memory too small for this pilot; choose fewer train images')
    mean_train = flat_train.reshape(n_train, -1, FEATURE_DIM).mean(axis=1)
    mean_test = flat_test.reshape(len(flat_test), -1, FEATURE_DIM).mean(axis=1)
    results = {}
    for pooling, train, test in [('mean', mean_train, mean_test),
                                 ('flatten', flat_train, flat_test)]:
        saved = out / f'{pooling}_predictions.npz'
        if saved.exists():
            with np.load(saved, allow_pickle=False) as previous:
                if (not np.array_equal(previous['response'], bundle.test.responses) or
                        not np.array_equal(previous['mask'], bundle.test.mask) or
                        not np.array_equal(previous['image_ids'], bundle.test.image_ids) or
                        not np.array_equal(previous['stimulus_ids'],
                                           bundle.test.rows.stimulus_id.to_numpy(dtype=str))):
                    raise ValueError(f'Saved pilot predictions have different test data: {saved}')
                alpha = dict(F=float(previous['alpha_F']), N=float(previous['alpha_N']))
                scores = {key:previous[key] for key in (
                    'raw_pearson', 'bbscore_pearson', 'split_half_reliability',
                    'noise_ceiling', 'noise_corrected_pearson', 'n_repeated_images')}
            summary = _score_summary(pooling, alpha, scores, n_train)
            print(f'Restored completed {pooling} fit from {saved}', flush=True)
        else:
            summary, predictions, scores = _fit_one_pooling(
                ridge_class, pooling, train, test, bundle, protocol)
            _atomic_npz(saved, prediction=predictions, response=bundle.test.responses,
                        mask=bundle.test.mask, image_ids=bundle.test.image_ids,
                        stimulus_ids=bundle.test.rows.stimulus_id.to_numpy(dtype=str),
                        alpha_F=np.asarray(summary['chosen_alpha']['F']),
                        alpha_N=np.asarray(summary['chosen_alpha']['N']),
                        **scores)
        results[pooling] = summary
        print(json.dumps(summary, indent=2), flush=True)
    _atomic_json(out / 'summary.json', dict(protocol='spatial-pilot-v1',
                                           methods=results, published_reference=0.95,
                                           note='Same images and GPU ridge for both pooling choices. '
                                                'Not a faithful BBScore per-target sklearn RidgeCV fit.'))
    print('Mean vs flatten BBScore medians:',
          results['mean']['bbscore_floor']['median'],
          results['flatten']['bbscore_floor']['median'], flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('extract', 'fit'), required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--bbscore-utils', required=True,
                        help='Upstream BBScore checkout metrics/utils.py')
    parser.add_argument('--n-train', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--layer-index', type=int, default=DEFAULT_LAYER)
    parser.add_argument('--chunk-size', type=int, default=8)
    args = parser.parse_args(argv)
    if args.chunk_size < 1:
        parser.error('--chunk-size must be positive')
    cfg = load_config(args.config)
    bundle = TVSDTrainingBundle(cfg)
    protocol = _plan(cfg, bundle, args.n_train, args.seed,
                     args.layer_index, args.bbscore_utils, args.chunk_size)
    out = Path(args.output)
    _prepare_protocol(out / 'protocol.json', protocol)
    if args.phase == 'extract':
        for _ in _feature_chunks(out, protocol, cfg, extract=True):
            pass
    else:
        _run_fit(out, protocol, cfg, bundle)


if __name__ == '__main__':
    main()
