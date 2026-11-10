#!/usr/bin/env python3
"""Build and load the TVSD THINGS -> KS4 masked-response dataset.

The selected neuronal union is provided by final_trimmed/unit_to_readout.csv.
No neuron is inferred here. A missing response is ALWAYS masked, never a zero target.

Inside the existing project container:
    python scripts/tvsd_dataset.py prepare
    python scripts/tvsd_dataset.py check --animal F --partition train
    python scripts/tvsd_dataset.py check --animal N --partition test

As a Python module:
    from tvsd_dataset import TVSDDataset
    train = TVSDDataset(animal='F', partition='train', image_size=224)
    batch = next(iter(torch.utils.data.DataLoader(train, batch_size=8)))

Dependencies: numpy, pandas, Pillow, torch. No torchvision required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

ROOT = Path('/project/data/processed/ks4_v1')
MAP_DIR = ROOT / 'identity_diagnostics/joint_labeled/selection/final_trimmed'
IMAGE_ROOT = Path('/project/data/assets/THINGS')
DEFAULT_PREPARED = ROOT / 'model_dataset_v1'


def _require_columns(df, cols, label):
    missing = set(cols).difference(df.columns)
    if missing:
        raise ValueError(f'{label}: missing columns: {sorted(missing)}')


def _safe_image(root: Path, name: str) -> Path:
    """Reject absolute/traversing paths even when the manifest is altered."""
    rel = Path(str(name))
    if rel.is_absolute() or not rel.parts or '..' in rel.parts or str(rel) == '.':
        raise ValueError(f'Unsafe/empty image path: {name!r}')
    path = (root / rel).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Image outside image root: {name!r}')
    return path


def _image_partition(stimulus_ids, *, val_fraction: float, seed: int):
    """Fixed image-level validation across animals/sessions; never split test images."""
    if not 0 < val_fraction < 1:
        raise ValueError('val_fraction must be strictly between 0 and 1')
    train_ids = sorted(s for s in set(stimulus_ids) if s.startswith('train:'))
    rng = np.random.default_rng(seed)
    n_val = int(round(val_fraction * len(train_ids)))
    if not 0 < n_val < len(train_ids):
        raise ValueError('Invalid validation size')
    val_ids = set(np.array(train_ids, dtype=object)[rng.permutation(len(train_ids))[:n_val]])
    return {s: ('test' if s.startswith('test:') else 'val' if s in val_ids else 'train')
            for s in set(stimulus_ids)}


def _write_npz_atomic(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    try:
        with tmp.open('wb') as f:
            np.savez_compressed(f, **arrays)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def prepare_dataset(
    *, processed_root: Path = ROOT, map_dir: Path = MAP_DIR,
    image_root: Path = IMAGE_ROOT, output_dir: Path = DEFAULT_PREPARED,
    val_fraction: float = 0.1, seed: int = 8000,
    expected_sessions: int = 45, overwrite: bool = False,
    check_images: bool = True,
):
    """Pack selected response columns; NEVER copy full ragged spike arrays.

    Validates exact session/UUID alignment, stimulus identity and observation masks.
    Does not modify source NPZs or final neuron assignments.
    """
    processed_root, map_dir, image_root, output_dir = map(Path, (
        processed_root, map_dir, image_root, output_dir))
    if output_dir.resolve() in (processed_root.resolve(), map_dir.resolve()):
        raise ValueError('Output must be a separate dataset directory')
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f'{output_dir} already exists. Use --overwrite to rebuild.')
    unit_map = pd.read_csv(map_dir / 'unit_to_readout.csv', low_memory=False)
    _require_columns(unit_map, ['unit_instance_id', 'session', 'animal', 'readout_index',
                                'readout_id', 'train_presentations', 'test_presentations'], 'unit map')
    if unit_map.isna().any().any():
        raise ValueError('Unit map contains missing fields')
    if unit_map.unit_instance_id.duplicated().any():
        raise ValueError('One selected unit_instance_id occurs more than once')
    if unit_map.duplicated(['session', 'animal', 'readout_index']).any():
        raise ValueError('Two simultaneously recorded units assigned to the same model readout')
    if not unit_map.animal.isin(['F', 'N']).all():
        raise ValueError('Only monkeys F and N are expected')
    if not (unit_map.animal + '/' + unit_map.readout_index.astype(str) == unit_map.readout_id).all():
        raise ValueError('Inconsistent animal/readout_id in map')
    n_readouts = {}
    for animal, g in unit_map.groupby('animal'):
        inds = np.sort(g.readout_index.unique())
        if not np.array_equal(inds, np.arange(len(inds))):
            raise ValueError(f'{animal}: readout indices are not 0..N-1')
        n_readouts[animal] = int(len(inds))
    readout_meta_path = map_dir / 'readout_index.csv'
    if readout_meta_path.exists():
        readout_meta = pd.read_csv(readout_meta_path, low_memory=False)
        _require_columns(readout_meta, ['animal','readout_index','readout_id'], 'readout_index')
        if readout_meta.duplicated(['animal','readout_index']).any():
            raise ValueError('Duplicate readout index in readout_index.csv')
        observed_pairs = set(zip(unit_map.animal,unit_map.readout_index))
        meta_pairs = set(zip(readout_meta.animal,readout_meta.readout_index))
        if observed_pairs != meta_pairs:
            raise ValueError('unit_to_readout.csv and readout_index.csv disagree')

    stimulus = pd.read_csv(processed_root / 'images/stimulus_manifest.csv', low_memory=False)
    pres = pd.read_csv(processed_root / 'images/presentation_to_stimulus.csv', low_memory=False)
    _require_columns(stimulus, ['stimulus_id','split','image_index','resolution_status',
                                'image_relative_path'], 'stimulus manifest')
    _require_columns(pres, ['session','source_row','stimulus_id','split','image_index'],
                     'presentation map')
    if stimulus.stimulus_id.duplicated().any() or pres.duplicated(['session','source_row']).any():
        raise ValueError('Duplicate stimulus identity or session/source_row')
    if not stimulus.resolution_status.eq('resolved').all():
        raise ValueError('Image manifest has unresolved images')
    if stimulus.image_relative_path.isna().any():
        raise ValueError('Missing resolved image path')
    if not stimulus.stimulus_id.eq(stimulus.split + ':' + stimulus.image_index.astype(str)).all():
        raise ValueError('Stimulus manifest split/index mismatch')
    if not pres.stimulus_id.eq(pres.split + ':' + pres.image_index.astype(str)).all():
        raise ValueError('Presentation split/index mismatch')
    if set(pres.stimulus_id) != set(stimulus.stimulus_id):
        raise ValueError('Mismatch between stimuli used in presentations and image manifest')
    if check_images:
        if not image_root.is_dir():
            raise FileNotFoundError(f'Image root does not exist: {image_root}')
        missing = [s for s in stimulus.image_relative_path
                   if not _safe_image(image_root, s).is_file()]
        if missing:
            raise FileNotFoundError(f'{len(missing)} stimulus images missing; first: {missing[:3]}')
    partition = _image_partition(stimulus.stimulus_id, val_fraction=val_fraction, seed=seed)
    stimulus_paths = stimulus.set_index('stimulus_id')['image_relative_path'].to_dict()
    sessions = sorted(pres.session.unique())
    if len(sessions) != expected_sessions or set(sessions) != set(unit_map.session.unique()):
        raise ValueError(f'Session coverage mismatch: {len(sessions)} presentations, '
                         f'{unit_map.session.nunique()} with readouts; expected {expected_sessions}')
    if pres.session.str.extract(r'^macaque([FN])_TVSD_')[0].isna().any():
        raise ValueError('Unexpected session naming in image manifest')
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    actual_counts = {}
    observed_train_pairs = 0
    observed_test_pairs = 0
    for k, session in enumerate(sessions, start=1):
        selected = unit_map.loc[unit_map.session.eq(session)].copy()
        if selected.empty or selected.animal.nunique() != 1:
            raise ValueError(f'{session}: no units or mixed animals')
        animal = str(selected.animal.iloc[0])
        if not session.startswith(f'macaque{animal}_'):
            raise ValueError(f'{session}: monkey does not match session metadata')
        per_session = pres.loc[pres.session.eq(session)].set_index('source_row')
        original = processed_root / 'sessions' / f'{session}.npz'
        if not original.is_file():
            raise FileNotFoundError(original)
        # Important: selectively decompress only the relevant NPZ members.
        # Do not touch spike_times_s, spike_offsets or waveform arrays.
        with np.load(original, allow_pickle=False) as npz:
            source_rows = npz['trial_source_rows'].astype(np.int64)
            original_ids = npz['unit_instance_ids'].astype(str)
            trial_split = npz['split'].astype(str)
            trial_image_index = npz['image_index'].astype(np.int64)
            response_window = npz['response_window_ms'].astype(float)
            if len(set(original_ids)) != len(original_ids):
                raise ValueError(f'{session}: duplicate source unit IDs')
            if len(set(source_rows)) != len(source_rows):
                raise ValueError(f'{session}: duplicate source trial rows')
            missing_ids = set(selected.unit_instance_id) - set(original_ids)
            if missing_ids:
                raise ValueError(f'{session}: mapped unit missing from NPZ: {next(iter(missing_ids))}')
            columns = {name: i for i,name in enumerate(original_ids)}
            selected = selected.sort_values('readout_index', kind='stable')
            col_indices = np.array([columns[x] for x in selected.unit_instance_id], dtype=int)
            local_readout_indices = selected.readout_index.to_numpy(dtype=np.int32)
            local_counts = npz['counts'][:, col_indices]
            local_observed = npz['observed'][:, col_indices]
        if not np.array_equal(response_window, [40.0, 160.0]):
            raise ValueError(f'{session}: unexpected spike window: {response_window}')
        if (local_counts.shape != local_observed.shape or
            local_counts.shape[0] != len(source_rows) or
            local_counts.shape[1] != len(local_readout_indices)):
            raise ValueError(f'{session}: malformed neural response arrays')
        if (local_counts[~local_observed] != 0).any():
            raise ValueError(f'{session}: unobserved targets have nonzero values')
        if not per_session.index.is_unique or set(per_session.index) != set(source_rows):
            raise ValueError(f'{session}: mismatch in presentation rows')
        ordered = per_session.loc[source_rows]
        if not np.array_equal(trial_split, ordered.split.to_numpy(dtype=str)):
            raise ValueError(f'{session}: trial split order differs from NPZ')
        if not np.array_equal(trial_image_index, ordered.image_index.to_numpy(dtype=np.int64)):
            raise ValueError(f'{session}: image index differs from NPZ')
        assigned_ids = ordered.stimulus_id.to_numpy(dtype=str)
        # Verify per-session train/test presentation support per unit.
        local_train = int(np.count_nonzero(trial_split == 'train'))
        local_test = int(np.count_nonzero(trial_split == 'test'))
        if not (selected.train_presentations.to_numpy() == local_train).all():
            raise ValueError(f'{session}: selected units disagree on training presentation count')
        if not (selected.test_presentations.to_numpy() == local_test).all():
            raise ValueError(f'{session}: selected units disagree on test presentation count')
        for i, (source_row, st_id) in enumerate(zip(source_rows, assigned_ids)):
            records.append(dict(animal=animal, session=session, trial_index=i,
                                source_row=int(source_row), stimulus_id=st_id,
                                split=str(trial_split[i]), partition=partition[st_id],
                                image_index=int(trial_image_index[i]),
                                image_relative_path=stimulus_paths[st_id]))
        actual_counts[session] = [local_train, local_test, len(local_readout_indices)]
        observed_train_pairs += int(local_observed[trial_split == 'train'].sum())
        observed_test_pairs += int(local_observed[trial_split == 'test'].sum())
        _write_npz_atomic(output_dir / 'sessions' / f'{session}.npz',
                          counts=local_counts.astype(np.uint16, copy=False),
                          observed=local_observed.astype(bool, copy=False),
                          readout_indices=local_readout_indices,
                          source_rows=source_rows,
                          stimulus_ids=assigned_ids,
                          response_window_ms=response_window)
        print(f'{k:2d}/{len(sessions)} {session}: {len(source_rows)} presentations, '
              f'{len(local_readout_indices)} selected units', flush=True)
    index = pd.DataFrame.from_records(records)
    index.to_csv(output_dir / 'sample_index.csv', index=False)
    # Copy the exact inputs into this immutable-by-convention dataset version.
    inputs = {
        'unit_to_readout.csv': map_dir / 'unit_to_readout.csv',
        'stimulus_manifest.csv': processed_root / 'images/stimulus_manifest.csv',
        'presentation_to_stimulus.csv': processed_root / 'images/presentation_to_stimulus.csv',
    }
    if readout_meta_path.is_file():
        inputs['readout_index.csv'] = readout_meta_path
    digests = {}
    for name, path in inputs.items():
        shutil.copyfile(path, output_dir / name)
        digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary = dict(version='tvsd_masked_v1', n_readouts=n_readouts,
                   n_selected_unit_instances=int(len(unit_map)),
                   n_sessions=len(sessions), n_presentations=len(index),
                   n_nominal_train_response_entries=int(unit_map.train_presentations.sum()),
                   n_observed_train_response_entries=observed_train_pairs,
                   n_observed_test_response_entries=observed_test_pairs,
                   frozen_input_sha256=digests,
                   response_window_ms=[40.0,160.0], validation_fraction=val_fraction,
                   split_seed=seed, image_root=str(image_root.resolve()),
                   session_counts=actual_counts,
                   # Holdout image IDs are shared across both animals/sessions.
                   partition_images={x: int(index.loc[index.partition.eq(x),'stimulus_id'].nunique())
                                     for x in ('train','val','test')},
                   partition_presentations=index.partition.value_counts().to_dict(),
                   preprocessing='RGB, PIL bicubic resize to image_size x image_size, float [0,1]; '
                                 'no stimulus geometry/normalization equivalence assumed')
    (output_dir / 'dataset_info.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('\nDataset built:', output_dir)
    print('Readouts:', n_readouts, 'Selected KS4 unit instances:', len(unit_map))
    print('Nominal/observed train response entries:',
          summary['n_nominal_train_response_entries'], observed_train_pairs)
    print('Partition images:', summary['partition_images'])
    print('Partition presentations:', summary['partition_presentations'])
    print('No spike trains, source files, or readout identities modified.')
    return summary


class TVSDDataset(Dataset):
    """Per-animal image-presentation dataset with fixed dense readout target axis.

    Output dict:
      image            float32 [3,H,W], RGB [0,1] (no unverified presentation correction)
      responses        float32 [n_readouts_for_animal], spikes in [40,160) ms
      response_mask    bool    [n_readouts_for_animal]; absent readouts are false
      stimulus_id      string  'train:N' or 'test:N'
      session          string
      source_row       int
      animal           string

    Use separate dataloaders for F and N; shared core may be trained by alternating them.
    """
    def __init__(self, *, animal: str, partition: str,
                 prepared_root: Path = DEFAULT_PREPARED,
                 image_root: Optional[Path] = None, image_size: int = 224,
                 load_responses: bool = True):
        self.root = Path(prepared_root)
        if animal not in ('F','N') or partition not in ('train','val','test'):
            raise ValueError('animal must be F/N; partition must be train/val/test')
        if int(image_size) != image_size or image_size < 1:
            raise ValueError('image_size must be a positive integer')
        self.animal, self.partition, self.image_size = animal, partition, int(image_size)
        info = json.loads((self.root / 'dataset_info.json').read_text())
        self.n_readouts = int(info['n_readouts'][animal])
        self.image_root = Path(image_root) if image_root is not None else Path(info['image_root'])
        self.samples = pd.read_csv(self.root / 'sample_index.csv', low_memory=False)
        self.samples = self.samples.loc[
            self.samples.animal.eq(animal) & self.samples.partition.eq(partition)].reset_index(drop=True)
        if self.samples.empty:
            raise ValueError(f'No samples: {animal} {partition}')
        self._session_data: Dict[str, Dict[str,np.ndarray]] = {}
        if load_responses:
            for session in sorted(self.samples.session.unique()):
                with np.load(self.root / 'sessions' / f'{session}.npz', allow_pickle=False) as data:
                    counts = data['counts']
                    observed = data['observed']
                    indices = data['readout_indices'].astype(np.int64)
                if len(set(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= self.n_readouts):
                    raise ValueError(f'Invalid local readout indices for {session}')
                if counts.shape != observed.shape or counts.shape[1] != len(indices):
                    raise ValueError(f'Malformed responses for {session}')
                self._session_data[session] = dict(counts=counts, observed=observed, indices=indices)
        self.load_responses = load_responses

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        row = self.samples.iloc[int(i)]
        path = _safe_image(self.image_root, row.image_relative_path)
        with Image.open(path) as im:
            im = im.convert('RGB')
            im = im.resize((self.image_size, self.image_size), resample=Image.Resampling.BICUBIC)
            pixels = np.array(im, dtype=np.uint8, copy=True)
        image = torch.from_numpy(pixels).permute(2,0,1).float().div_(255.0)
        responses = torch.zeros(self.n_readouts, dtype=torch.float32)
        response_mask = torch.zeros(self.n_readouts, dtype=torch.bool)
        if self.load_responses:
            data = self._session_data[row.session]
            j = int(row.trial_index)
            valid = data['observed'][j]
            pos = data['indices'][valid]
            responses[pos] = torch.from_numpy(data['counts'][j,valid].astype(np.float32))
            response_mask[pos] = True
        return dict(image=image, responses=responses, response_mask=response_mask,
                    stimulus_id=str(row.stimulus_id), session=str(row.session),
                    source_row=int(row.source_row), animal=self.animal)


def check_dataset(*, animal: str, partition: str, prepared_root: Path,
                  image_root: Optional[Path], image_size: int, batch_size: int):
    ds = TVSDDataset(animal=animal, partition=partition, prepared_root=prepared_root,
                     image_root=image_root, image_size=image_size)
    batch = next(iter(DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)))
    x, y, mask = batch['image'], batch['responses'], batch['response_mask']
    if (x.shape != (len(y),3,image_size,image_size) or
        y.shape != mask.shape or y.shape[1] != ds.n_readouts):
        raise AssertionError('Unexpected batch shape')
    if mask.dtype != torch.bool or not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise AssertionError('Invalid pixel/response data')
    if (y[~mask] != 0).any() or x.min() < 0 or x.max() > 1 or not mask.any():
        raise AssertionError('Invalid unobserved values, pixel range or empty mask')
    print(f'PASS {animal}/{partition}: {len(ds):,} presentations, {ds.n_readouts} readouts')
    print('Image batch:', tuple(x.shape), 'range', (round(float(x.min()),4), round(float(x.max()),4)))
    print('Responses:', tuple(y.shape), 'mask:', tuple(mask.shape))
    print('Observed fraction in batch:', round(float(mask.float().mean()),4))
    print('Mean observed spike count [40,160) ms:', round(float(y[mask].mean()),4))
    print('First stimulus:', batch['stimulus_id'][0], 'session:', batch['session'][0])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    cmd = p.add_subparsers(dest='cmd', required=True)
    prep = cmd.add_parser('prepare', help='Build the compact response packs and global presentation index')
    prep.add_argument('--processed-root', type=Path, default=ROOT)
    prep.add_argument('--map-dir', type=Path, default=MAP_DIR)
    prep.add_argument('--images-root', type=Path, default=IMAGE_ROOT)
    prep.add_argument('--output-dir', type=Path, default=DEFAULT_PREPARED)
    prep.add_argument('--val-fraction', type=float, default=0.1)
    prep.add_argument('--seed', type=int, default=8000)
    prep.add_argument('--expected-sessions', type=int, default=45)
    prep.add_argument('--overwrite', action='store_true')
    prep.add_argument('--skip-image-file-check', action='store_true')
    check = cmd.add_parser('check', help='Decode a batch and verify mask/readout layout')
    check.add_argument('--animal', choices=['F','N'], default='F')
    check.add_argument('--partition', choices=['train','val','test'], default='train')
    check.add_argument('--prepared-root', type=Path, default=DEFAULT_PREPARED)
    check.add_argument('--images-root', type=Path, default=None)
    check.add_argument('--image-size', type=int, default=224)
    check.add_argument('--batch-size', type=int, default=8)
    args = p.parse_args()
    if args.cmd == 'prepare':
        prepare_dataset(processed_root=args.processed_root,map_dir=args.map_dir,
                        image_root=args.images_root,output_dir=args.output_dir,
                        val_fraction=args.val_fraction,seed=args.seed,
                        expected_sessions=args.expected_sessions,overwrite=args.overwrite,
                        check_images=not args.skip_image_file_check)
    else:
        check_dataset(animal=args.animal,partition=args.partition,
                      prepared_root=args.prepared_root,image_root=args.images_root,
                      image_size=args.image_size,batch_size=args.batch_size)


if __name__ == '__main__':
    main()
