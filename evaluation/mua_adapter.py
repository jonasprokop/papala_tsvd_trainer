"""Prepare Papale's released, *normalized MUA* as an independent TVSD cohort.

The original Papale ``norm_MUA.m`` performs per-day, per-channel normalization
before saving ``THINGS_normMUA.mat``. BBScore reads those arrays directly.
This adapter NEVER renormalizes their neural responses and NEVER reads KS4
tracks. It packs the already-normalized responses into the existing frozen
TVSD presentation format, with one stable channel column per animal.

Inputs are the two HDF5/v7.3 ``THINGS_normMUA.mat`` files and BBScore's
``things_imgs_monkey{F,N}.mat`` stimulus metadata files. For each area, channel
indices follow ``bbscore_public/data/TVSD.py:TVSDAssembly._get_channels_for_region``.

Output is a NEW model_dataset directory. It intentionally has no image
``normalization.json``: calibrate that separately on its own train partition.
MUA neural values are signed z-scores; Sinz's Poisson loss is NOT valid on them.
Ridge evaluation is usable without changing the existing tracked-neuron data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd


# Exact original channel layout, NOT the sorted-unit population.
CHANNELS = {
    'V1': {'F': (0, 512), 'N': (0, 512)},
    'V4': {'F': (832, 1024), 'N': (512, 768)},
    'IT': {'F': (512, 832), 'N': (768, 1024)},
}
WINDOW_MS = {'V1': (25, 125), 'V4': (50, 150), 'IT': (75, 175)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _paths_from_metadata(file, split: str) -> list[str]:
    """Decode BBScore's reference-based MATLAB THINGS image-path arrays."""
    field = file[f'{split}_imgs']
    refs = np.asarray(field['things_path']).ravel()
    paths = []
    for ref in refs:
        chars = np.asarray(field[ref]).ravel()
        path = ''.join(chr(int(char)) for char in chars).replace('\\', '/')
        parts = Path(path)
        if parts.is_absolute() or not parts.parts or '..' in parts.parts:
            raise ValueError(f'Unsafe source THINGS path: {path!r}')
        paths.append(parts.as_posix())
    if len(set(paths)) != len(paths):
        raise ValueError(f'Non-unique {split} images in original TVSD metadata')
    return paths


def _read_monkey(mua_path: Path, metadata_path: Path, *, animal: str,
                 region: str):
    try:
        import h5py
    except ImportError as exc:
        raise ImportError('Papale MUA .mat files require h5py (Python 3.10 runtime)') from exc
    start, stop = CHANNELS[region][animal]
    with h5py.File(metadata_path, 'r') as meta:
        train_paths = _paths_from_metadata(meta, 'train')
        test_paths = _paths_from_metadata(meta, 'test')
    with h5py.File(mua_path, 'r') as f:
        # MATLAB v7.3 HDF5 dimensions are already in BBScore's access order:
        # [image, channel] and [repeat, image, channel], respectively.
        # Keep the source's double precision (MATLAB); do not quantize scores.
        train = np.asarray(f['train_MUA'][:, start:stop], dtype=np.float64)
        test = np.asarray(f['test_MUA'][:, start:stop], dtype=np.float64)
        reps = np.asarray(f['test_MUA_reps'][:, :, start:stop], dtype=np.float64)
        measured_rel = None
        if 'reliab' in f:
            rel = np.asarray(f['reliab'], dtype=np.float64)
            # Original MATLAB reliab[channel, pair] is transposed by HDF5.
            if rel.ndim == 2 and rel.shape[1] == 1024:
                rel = rel[:, start:stop].T
            elif rel.ndim == 2 and rel.shape[0] == 1024:
                rel = rel[start:stop, :]
            else:
                raise ValueError(f'Unexpected Papale reliab shape: {rel.shape}')
            # Brain-Score package_papale2025.py: reliability.mean(axis=1), a plain
            # mean, so a channel with any NaN pair is NaN and fails the > 0.3
            # filter. nanmean is kept separately for inspection only.
            # np.nanmean raises RuntimeWarning on channels with all pairs
            # missing. Keep that inspection statistic without making such a
            # channel abort an otherwise valid MUA import under -Werror.
            observed_pairs = ~np.isnan(rel)
            counts = observed_pairs.sum(axis=1)
            inspection_mean = np.full(len(rel), np.nan, dtype=np.float64)
            np.divide(np.where(observed_pairs, rel, 0.0).sum(axis=1), counts,
                      out=inspection_mean, where=counts > 0)
            measured_rel = dict(mean=rel.mean(axis=1), nanmean=inspection_mean,
                                n_pairs=int(rel.shape[1]))
        measured_oracle = None
        if 'oracle' in f:
            orc = np.asarray(f['oracle'], dtype=np.float64)
            # Brain-Score uses `oracle` per electrode; accept [1024], [1,1024], [1024,1].
            if orc.size == 1024:
                measured_oracle = orc.reshape(-1)[start:stop]
            else:
                raise ValueError(f'Unexpected Papale oracle shape: {orc.shape}')
    if train.shape != (len(train_paths), stop-start):
        raise ValueError(f'{animal}: train_MUA and stimulus metadata disagree: {train.shape}')
    if test.shape != (len(test_paths), stop-start) or reps.ndim != 3 or (
            reps.shape[1:] != test.shape or reps.shape[0] < 2):
        raise ValueError(f'{animal}: test_MUA/repetitions and stimulus metadata disagree')
    if not np.allclose(np.nanmean(reps, axis=0), test, rtol=3e-4, atol=3e-4,
                       equal_nan=True):
        raise ValueError(f'{animal}: average of released repetitions differs from test_MUA')
    return train_paths, test_paths, train, test, reps, (measured_rel, measured_oracle)


def _assign_partitions(n_train: int, *, seed: int, val_fraction: float):
    if not 0 < val_fraction < 1 or n_train < 3:
        raise ValueError('Need at least 3 training images and 0 < val_fraction < 1')
    # Identical rule to frozen KS4 dataset: sorted image IDs, RNG permutation.
    ids = np.asarray([f'train:{i+1}' for i in range(n_train)])
    ids.sort()
    n_val = round(val_fraction*n_train)
    if not 0 < n_val < n_train:
        raise ValueError('Validation selection has no train or validation images')
    selected = set(np.random.default_rng(seed).permutation(ids)[:n_val])
    return {i+1: ('val' if f'train:{i+1}' in selected else 'train')
            for i in range(n_train)}


def _split_sha256(rows: pd.DataFrame) -> str:
    cols = ['stimulus_id', 'partition', 'image_relative_path']
    unique = rows[cols].drop_duplicates().sort_values(cols, kind='stable')
    if unique.stimulus_id.duplicated().any():
        raise ValueError('Different images share a stimulus ID')
    payload = json.dumps(unique.values.tolist(), ensure_ascii=False,
                         separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _pack_monkey(folder: Path, *, animal: str, region: str, train_paths,
                 test_paths, train, reps, partitions):
    """Build row-level test repeats; keep normalized signed MUA untouched."""
    session = f'macaque{animal}_TVSD_normMUA_{region}'
    n_train = len(train_paths)
    n_test = len(test_paths)
    n_reps, _, width = reps.shape
    neural = np.concatenate((train, reps.transpose(1, 0, 2).reshape(-1, width)))
    observed = np.isfinite(neural)
    if not observed.all():
        # Existing masked-data loaders require zero in missing cells.
        neural = np.where(observed, neural, 0.0)
    records = []
    for i, path in enumerate(train_paths, 1):
        records.append(dict(animal=animal, session=session, trial_index=i-1,
                            source_row=i-1, stimulus_id=f'train:{i}',
                            partition=partitions[i], split='train', image_index=i,
                            image_relative_path=path))
    cursor = n_train
    for i, path in enumerate(test_paths, 1):
        for rep in range(n_reps):
            records.append(dict(animal=animal, session=session, trial_index=cursor,
                                source_row=cursor, stimulus_id=f'test:{i}',
                                partition='test', split='test', image_index=i,
                                image_relative_path=path))
            cursor += 1
    if neural.shape != (len(records), width):
        raise AssertionError('MUA response/presentation ordering mismatch')
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / f'{session}.npz', counts=neural,
                        observed=observed, neuron_indices=np.arange(width, dtype=np.int32),
                        source_rows=np.arange(len(records), dtype=np.int64),
                        stimulus_ids=np.asarray([r['stimulus_id'] for r in records]),
                        response_window_ms=np.asarray(WINDOW_MS[region], dtype=float))
    return records, int(observed.sum()), int((~observed).sum())


def prepare(*, f_mua: Path, n_mua: Path, f_metadata: Path, n_metadata: Path,
            image_root: Path, output_dir: Path, region: str = 'V1',
            val_fraction: float = 0.1, seed: int = 8000, check_images=True):
    """Create an immutable new model_dataset. An existing output is NEVER touched."""
    if region not in CHANNELS:
        raise ValueError('region must be V1, V4, or IT')
    output_dir = Path(output_dir)
    sources = {f'{a}_{k}': Path(path).resolve() for a, k, path in (
        ('F', 'mua', f_mua), ('N', 'mua', n_mua),
        ('F', 'metadata', f_metadata), ('N', 'metadata', n_metadata))}
    for name, path in sources.items():
        if not path.is_file():
            raise FileNotFoundError(f'{name} not found: {path}')
    if output_dir.exists():
        raise FileExistsError(f'Output already exists: {output_dir}')
    image_root = Path(image_root).resolve()
    both = {}
    for animal in ('F', 'N'):
        both[animal] = _read_monkey(sources[f'{animal}_mua'],
                                    sources[f'{animal}_metadata'], animal=animal,
                                    region=region)
    f_train, f_test = both['F'][:2]
    if (both['N'][0] != f_train or both['N'][1] != f_test):
        raise ValueError('F and N stimulus paths/order differ; BBScore stacks '
                         'them as matching rows, so a shared image index is unsafe')
    if set(f_train) & set(f_test):
        raise ValueError('Paper train and test stimulus images overlap')
    if check_images:
        for rel in f_train + f_test:
            if not (image_root/rel).is_file():
                raise FileNotFoundError(f'Original THINGS image missing: {image_root/rel}')
    partitions = _assign_partitions(len(f_train), seed=seed,
                                    val_fraction=val_fraction)
    digests = {name: _sha256(path) for name, path in sources.items()}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # Keep partial failures outside the final versioned location.
    temporary = Path(tempfile.mkdtemp(prefix=f'.{output_dir.name}.',
                                      dir=output_dir.parent))
    try:
        records = []
        observed_count, missing_count = 0, 0
        indices, unit_rows = [], []
        for animal in ('F', 'N'):
            trp, tep, train, _, reps, (measured, oracle) = both[animal]
            local, observed, missing = _pack_monkey(
                temporary/'sessions', animal=animal, region=region,
                train_paths=trp, test_paths=tep, train=train, reps=reps,
                partitions=partitions)
            records.extend(local)
            observed_count += observed
            missing_count += missing
            for j in range(train.shape[1]):
                neuron_id = f'{animal}/{j}'
                channel = CHANNELS[region][animal][0] + j
                record = dict(animal=animal, neuron_index=j, neuron_id=neuron_id,
                              original_channel_index=channel,
                              signal_type='Papale_normalized_MUA',
                              papale_reliab_measured=(float(measured['mean'][j])
                                                      if measured is not None else np.nan),
                              papale_reliab_measured_nanmean=(float(measured['nanmean'][j])
                                                              if measured is not None else np.nan),
                              papale_reliab_n_pairs=(measured['n_pairs']
                                                     if measured is not None else 0),
                              papale_oracle_measured=(float(oracle[j])
                                                      if oracle is not None else np.nan))
                indices.append(record)
                unit_rows.append(dict(**record, session=f'macaque{animal}_TVSD_normMUA_{region}',
                                      unit_instance_id=f'{animal}/MUA/{channel}'))
        rows = pd.DataFrame.from_records(records)
        rows.to_csv(temporary/'sample_index.csv', index=False)
        pd.DataFrame(indices).to_csv(temporary/'neuron_index.csv', index=False)
        pd.DataFrame(unit_rows).to_csv(temporary/'unit_to_neuron.csv', index=False)
        n_neurons = {a: CHANNELS[region][a][1]-CHANNELS[region][a][0]
                     for a in ('F', 'N')}
        info = dict(version='tvsd_papale_normmua_v1', signal_type='normalized_MUA',
                    original_release='THINGS_normMUA.mat', area=region,
                    n_neurons=n_neurons, n_selected_unit_instances=sum(n_neurons.values()),
                    n_sessions=2, n_presentations=len(rows),
                    n_observed_response_entries=observed_count,
                    n_missing_response_entries=missing_count,
                    split_sha256=_split_sha256(rows), split_seed=int(seed),
                    validation_fraction=float(val_fraction),
                    image_root=str(image_root),
                    partition_images={p: int(rows.loc[rows.partition.eq(p),'stimulus_id'].nunique())
                                      for p in ('train','val','test')},
                    response_window_ms=list(WINDOW_MS[region]),
                    window_definition='start < time <= end (Papale norm_MUA.m)',
                    frozen_input_sha256=digests,
                    provenance='MUA channel responses: normalized in Papale norm_MUA.m, '
                               'loaded unchanged; NEVER pooled or spike-sorted')
        (temporary/'dataset_info.json').write_text(json.dumps(info, indent=2)+'\n')
        # Race/overwrite guard: no replacement of another experiment's files.
        if output_dir.exists():
            raise FileExistsError(output_dir)
        temporary.rename(output_dir)
        return info
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--f-mua', type=Path, required=True)
    p.add_argument('--n-mua', type=Path, required=True)
    p.add_argument('--f-metadata', type=Path, required=True)
    p.add_argument('--n-metadata', type=Path, required=True)
    p.add_argument('--image-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--region', choices=tuple(CHANNELS), default='V1')
    p.add_argument('--seed', type=int, default=8000)
    p.add_argument('--val-fraction', type=float, default=0.1)
    p.add_argument('--skip-image-check', action='store_true')
    a = p.parse_args()
    info = prepare(f_mua=a.f_mua, n_mua=a.n_mua,
                   f_metadata=a.f_metadata, n_metadata=a.n_metadata,
                   image_root=a.image_root, output_dir=a.output_dir,
                   region=a.region, seed=a.seed, val_fraction=a.val_fraction,
                   check_images=not a.skip_image_check)
    print(json.dumps(dict(output=str(a.output_dir), version=info['version'],
                          area=info['area'], n_neurons=info['n_neurons'],
                          n_images=info['partition_images'],
                          missing=info['n_missing_response_entries']), indent=2))
    print('Next: calibrate this dataset\'s train-image normalization.json. '
          'Sinz on this cohort needs training.loss.kind=mse and '
          'model.output.activation=identity (configs/tvsd_sinz_mua_v1.yaml); '
          'Poisson is invalid on signed z-scores.')


if __name__ == '__main__':
    main()
