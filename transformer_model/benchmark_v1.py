"""V1-only ZWM neural-alignment benchmark (F and N, same neuron axis).

Three explicit stages keep MetaCentrum use bounded:
  extract: ONE GPU forward per unique THINGS image, all 12 layers, chunk-resumable;
  fit:     ONE selected layer on CPU, official 21 alpha values and 10 CV folds;
  report:  aggregate the 12 held-out test scores, without fitting anything.

No stage reads or writes the project SQLite registry. The 1,173 tracked-neuron
cohort and its frozen train/test partition differ from BBScore's original
channel-level data; see transformer_model/BENCHMARK_V1.md for limitations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from sinz_model.runtime import load_config, sha256_file
from sinz_model.tvsd_data import TVSDTrainingBundle
from .benchmark_stats import finite_summary, score_repeated_stimuli
from .train import _images_index, _neuron_order, _safe_image
from .benchmark_probe import BB_SCORE_ALPHAS, fit_masked_bbscore


# awwkl/ZWM is a 24-block model. BBScore's public default RidgeMetric grid:
# github.com/neuroailab/bbscore_public/blob/main/metrics/ridge.py
LAYERS = tuple(f'transformer.h.{i}.ln_1' for i in range(0, 24, 2))
ALPHAS = BB_SCORE_ALPHAS
N_FOLDS = 10
N_CEILING_SPLITS = 50
FOLD_SEED = 42


def _atomic_npz(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as handle:
        temporary = Path(handle.name)
        try:
            np.savez_compressed(handle, **values)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def _atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode='w', suffix='.json',
                                     encoding='utf-8', delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(obj, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def _experiment(cfg, bundle, output_dir):
    dc = cfg['data']; bc = cfg['model']['backbone']
    if bc['kind'] != 'zwm' or cfg['model']['family'] != 'transformer_neural':
        raise ValueError('Expected a ZWM transformer model')
    if bc['model_ref'] != 'awwkl/zwm-bvd-170m/model.pt':
        raise ValueError('This comparison targets published ZWM 170M weights')
    if bc['input_mode'] != 'rgb':
        raise ValueError('BBScore presents original RGB images; set input_mode: rgb')
    info = bundle.info
    expected = {
        'tvsd_tracked_neurons_masked_v3': {'F': 804, 'N': 369},
        'tvsd_papale_normmua_v1': {'F': 512, 'N': 512},
    }
    version = info.get('version')
    if version not in expected or bundle.widths != expected[version]:
        raise ValueError('Unsupported or inconsistent V1 TVSD cohort')
    if version == 'tvsd_papale_normmua_v1' and (
            info.get('signal_type') != 'normalized_MUA' or info.get('area') != 'V1'):
        raise ValueError('Papale MUA adapter is not a V1 normalized-MUA cohort')
    index = _images_index(bundle.df)
    img_root = Path(dc['image_root'])
    source = {}
    for name, relative in index.items():
        path = _safe_image(img_root, relative)
        stat = path.stat()
        source[name] = [relative, stat.st_size, stat.st_mtime_ns]
    spec = dict(schema=1, implementation='bbscore-ridgecv-v1',
                dataset_fingerprint=bundle.fingerprint,
                weights_sha256=sha256_file(str(Path(bc['repo_root']) / 'out' / bc['model_ref'])),
                predictor_sha256=sha256_file(str(Path(bc['repo_root']) / 'zwm' / 'zwm_predictor.py')),
                zwm_ref=bc['model_ref'], layers=LAYERS, pooling=bc.get('pooling','mean'),
                input_mode=bc['input_mode'], image_root=str(img_root.resolve()),
                images=source)
    fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    out = Path(output_dir)
    manifest = out / 'protocol.json'
    protocol = dict(cache_fingerprint=fingerprint, benchmark='TVSDV1', animals=['F','N'],
                    n_neurons=sum(bundle.widths.values()), layers=LAYERS, alphas=ALPHAS,
                    inner_cv_folds=N_FOLDS, ceiling_splits=N_CEILING_SPLITS,
                    alpha_grid_source='neuroailab/bbscore_public metrics/ridge.py',
                    source_dataset=version,
                    dataset_fingerprint=bundle.fingerprint,
                    source_weights_sha256=spec['weights_sha256'],
                    source_predictor_sha256=spec['predictor_sha256'],
                    source_config=cfg)
    if manifest.is_file():
        previous = json.loads(manifest.read_text())
        if previous != json.loads(json.dumps(protocol)):
            raise ValueError('Existing benchmark directory uses different data/settings')
    else:
        _atomic_json(manifest, protocol)
    return out, index, fingerprint


def _chunks(out, index, key, cfg, *, extract):
    """Load a validated, resumable 3D layer bank without a giant extra copy."""
    folder = out / 'features' / key
    folder.mkdir(parents=True, exist_ok=True)
    stimuli = sorted(index)
    bc = cfg['model']['backbone']
    chunk_size = int(bc.get('cache_chunk_size', 256))
    if chunk_size < 1:
        raise ValueError('cache_chunk_size must be positive')
    extractor = None
    try:
        for start in range(0, len(stimuli), chunk_size):
            names = stimuli[start:start + chunk_size]
            path = folder / f'{start:07d}.npz'
            if path.is_file():
                with np.load(path, allow_pickle=False) as data:
                    features = data['features']
                    ids = data['stimulus_ids'].astype(str).tolist()
                if (ids != names or features.ndim != 3 or
                        features.shape[:2] != (len(names), len(LAYERS)) or
                        not np.isfinite(features).all()):
                    raise RuntimeError(f'Corrupt layer-bank chunk: {path}')
            elif not extract:
                raise FileNotFoundError(f'Run extract phase first; missing {path}')
            else:
                if extractor is None:
                    from .backbones import ZWMLayerBank
                    extractor = ZWMLayerBank(bc['model_ref'], repo_root=bc['repo_root'],
                                             layers=LAYERS, pooling=bc.get('pooling','mean'))
                rows = []
                for name in names:
                    with Image.open(_safe_image(Path(cfg['data']['image_root']),index[name])) as img:
                        frame = img.convert('RGB')
                        layers = extractor.extract(frame)
                    values = np.stack([np.asarray(layer[0], dtype=np.float32)
                                       for layer in layers])
                    if values.shape[0] != len(LAYERS) or not np.isfinite(values).all():
                        raise ValueError(f'Nonfinite or missing layers for {name}')
                    rows.append(values)
                features = np.stack(rows)
                _atomic_npz(path, stimulus_ids=np.asarray(names), features=features)
                print(f'ZWM layer bank: {min(start+chunk_size,len(stimuli))}/{len(stimuli)}',flush=True)
            yield names, features
    finally:
        if extractor is not None:
            extractor.close()


def _load_layer(out, index, key, cfg, layer_idx):
    mapping = {}
    for names, vectors in _chunks(out, index, key, cfg, extract=False):
        mapping.update(zip(names, vectors[:, layer_idx, :]))
    return mapping


def _fit_layer(cfg, bundle, out, index, key, layer_idx):
    layer = LAYERS[layer_idx]
    dest = out / 'layers' / f'{layer_idx:02d}.npz'
    predictions_path = out / 'predictions' / f'{layer_idx:02d}.npz'
    if dest.exists():
        if not predictions_path.exists():
            # Existing completed layers predate the trial-level prediction
            # handoff. Replay their saved readout without fitting again.
            with np.load(dest, allow_pickle=False) as previous:
                weight = previous['weight']
                bias = previous['bias']
                saved_ids = previous['neuron_ids'].astype(str)
            neuron_ids = np.asarray(_neuron_order(cfg, bundle)).astype(str)
            if not np.array_equal(saved_ids, neuron_ids):
                raise ValueError(f'Saved layer has a different neuron order: {dest}')
            features = _load_layer(out, index, key, cfg, layer_idx)
            test = bundle.test
            x_test = np.stack([features[s] for s in test.rows.stimulus_id])
            prediction = x_test @ weight.T + bias
            _save_test_predictions(predictions_path, layer, test, prediction, neuron_ids)
            print(f'Replayed held-out predictions from saved {layer} fit; no refitting')
        else:
            print(f'Already fitted {layer}; leaving result unchanged')
        return
    features = _load_layer(out, index, key, cfg, layer_idx)
    train = bundle.full
    test = bundle.test
    x = np.stack([features[s] for s in train.rows.stimulus_id])
    progress_dir = out / 'group_progress' / f'{layer_idx:02d}'

    def persist_group(number, columns, values):
        path = progress_dir / f'{number:04d}.npz'
        if values is None:
            if not path.is_file():
                return None
            with np.load(path, allow_pickle=False) as data:
                if not np.array_equal(data['neurons'], columns):
                    raise RuntimeError(f'Stale mask group saved at {path}')
                cached = {name:data[name] for name in ('weight','bias','alpha','cv_r')}
            if (cached['weight'].shape != (len(columns),x.shape[1]) or
                    any(cached[field].shape != (len(columns),)
                        for field in ('bias','alpha','cv_r')) or
                    not np.isfinite(cached['weight']).all() or
                    not np.isfinite(cached['bias']).all()):
                raise RuntimeError(f'Invalid saved ridge group at {path}')
            return cached
        _atomic_npz(path, neurons=columns, **values)

    # Exact public BBScore alpha grid, raw features, RidgeCV with one alpha per
    # neuron. Ten outer stimulus-group folds are retained as a separate metric.
    fit = fit_masked_bbscore(x, train.responses, train.mask,
                            train.rows.stimulus_id.to_numpy(), alphas=ALPHAS,
                            n_folds=N_FOLDS, seed=FOLD_SEED,
                            persist_group=persist_group)
    x_test = np.stack([features[s] for s in test.rows.stimulus_id])
    prediction = x_test @ fit['weight'].T + fit['bias']
    scoring = score_repeated_stimuli(prediction, test.responses, test.mask,
                                      test.image_ids, splits=N_CEILING_SPLITS,
                                      seed=FOLD_SEED)
    neuron_ids = np.asarray(_neuron_order(cfg, bundle)).astype(str)
    # The learned linear map is an independent artifact for Classical Experiments.
    payload = dict(layer=np.asarray(layer), neuron_ids=neuron_ids,
                   **fit,
                   **scoring)
    # Persist trial-level inputs for all protocol scorers. Write this BEFORE
    # the layer completion marker so interrupted jobs can safely resume.
    _save_test_predictions(predictions_path, layer, test, prediction, neuron_ids)
    _atomic_npz(dest, **payload)
    summary = finite_summary(scoring['noise_corrected_pearson'])
    print(f'Finished {layer}: score={summary["mean"]}, valid={summary["n_valid"]}',flush=True)


def _save_test_predictions(path, layer, test, prediction, neuron_ids):
    """Lossless held-out replay for Cadena, Brain-Score and measured MUA reliability.

    Group IDs describe the readout columns; session IDs describe presentations.
    Neither may be inferred from the other's axis.
    """
    neuron_ids = np.asarray(neuron_ids).astype(str)
    prediction = np.asarray(prediction)
    if prediction.shape != test.responses.shape or prediction.shape != test.mask.shape:
        raise ValueError('Held-out predictions and neural trial arrays disagree')
    if (len(neuron_ids) != prediction.shape[1] or
            len(set(neuron_ids)) != len(neuron_ids)):
        raise ValueError('Held-out neuron IDs do not match prediction columns')
    groups = np.asarray([name.split('/')[0] for name in neuron_ids])
    if not np.isin(groups, ('F', 'N')).all():
        raise ValueError('Expected F/N neuron IDs for TVSD V1')
    if len(test.rows) != prediction.shape[0] or len(test.image_ids) != prediction.shape[0]:
        raise ValueError('Test stimulus metadata and prediction rows disagree')
    values = dict(layer=np.asarray(layer), prediction=prediction,
                  response=test.responses, mask=test.mask, image_id=test.image_ids,
                  neuron_ids=neuron_ids, groups=groups,
                  session=test.rows.session.to_numpy(dtype=str))
    if path.exists():
        # An interrupted run may have written predictions but not its layer
        # completion marker. Reuse only an exact match; never silently replace.
        with np.load(path, allow_pickle=False) as previous:
            identical = set(previous.files) == set(values)
            if identical:
                for name, expected in values.items():
                    actual = previous[name]
                    same = (np.array_equal(actual, expected, equal_nan=True)
                            if expected.dtype.kind in 'fc' else np.array_equal(actual, expected))
                    if not same:
                        identical = False
                        break
        if not identical:
            raise FileExistsError(f'Existing held-out predictions differ: {path}')
        return
    _atomic_npz(path, **values)


def _report(out):
    rows = []
    protocol = json.loads((out/'protocol.json').read_text())
    expected_neurons = int(protocol['n_neurons'])
    for i, layer in enumerate(LAYERS):
        path = out / 'layers' / f'{i:02d}.npz'
        if not path.is_file():
            raise FileNotFoundError(f'Missing {layer} fit; run --phase fit --layer-index {i}')
        with np.load(path, allow_pickle=False) as data:
            ids = data['neuron_ids'].astype(str)
            score = data['noise_corrected_pearson']
            raw = data['raw_pearson']
            n_images = data['n_repeated_images']
            if len(ids) != expected_neurons or len(set(ids)) != expected_neurons:
                raise ValueError(f'Invalid neuron cohort for {layer}')
        rows.append(dict(layer=layer, noise_corrected=finite_summary(score),
                         raw_pearson=finite_summary(raw),
                         eligible_neurons=int(np.count_nonzero(n_images >= 20))))
    eligible = [row for row in rows if row['noise_corrected']['mean'] is not None]
    if not eligible:
        raise RuntimeError('No valid split-half-corrected V1 scores')
    best = max(eligible, key=lambda item:item['noise_corrected']['mean'])
    report = dict(benchmark='TVSDV1', model='ZWM 170M',
                  source_dataset=protocol['source_dataset'],
                  published_best_layer_score=0.95,
                  published_value_source='Aw et al., Table S25',
                  best_layer=best['layer'], best_score=best['noise_corrected']['mean'],
                  layers=rows, caveat=('Original BBScore normalized MUA targets; '
                  'the exact ZWM feature wrapper/pooling is still unverified.' if
                  protocol['source_dataset'] == 'tvsd_papale_normmua_v1' else
                  'Tracked sorted units, not the original BBScore MUA targets. '
                  'Pooling and paper layer selection remain unverified.'))
    _atomic_json(out/'summary.json', report)
    print(json.dumps(dict(best_layer=report['best_layer'],best_score=report['best_score'],
                          published=0.95), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['smoke','extract','fit','report'], required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--image', help='One actual THINGS image for --phase smoke')
    parser.add_argument('--layer-index', type=int,
                        help='0..11; required for fit; each CPU job fits one layer')
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.phase == 'smoke':
        if not args.image:
            parser.error('--phase smoke requires --image')
        bc = cfg['model']['backbone']
        from .backbones import ZWMLayerBank
        bank = ZWMLayerBank(bc['model_ref'],repo_root=bc['repo_root'],
                            layers=LAYERS,pooling=bc.get('pooling','mean'))
        try:
            with Image.open(args.image) as src:
                image = src.convert('RGB')
            result = bank.extract(image)
            if (len(result) != len(LAYERS) or
                    any(v.ndim != 2 or v.shape[0] != 1 or
                        not np.isfinite(v).all() for v in result)):
                raise RuntimeError('Invalid multi-layer ZWM forward outputs')
            print(json.dumps({'passed':True,'layers':list(LAYERS),
                              'shapes':[list(v.shape) for v in result]}))
        finally:
            bank.close()
        return
    if args.phase == 'report':
        _report(Path(args.output))
        return
    if args.phase == 'fit' and (args.layer_index is None or
                                 not 0 <= args.layer_index < len(LAYERS)):
        parser.error('--phase fit requires --layer-index 0..11')
    bundle = TVSDTrainingBundle(cfg)
    out, index, key = _experiment(cfg, bundle, args.output)
    if args.phase == 'extract':
        for _ in _chunks(out,index,key,cfg,extract=True):
            pass
        print('All 12 layers cached; no ridge fitting or registry writes performed')
    else:
        _fit_layer(cfg,bundle,out,index,key,args.layer_index)


if __name__ == '__main__':
    main()
