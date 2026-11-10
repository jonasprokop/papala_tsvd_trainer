#!/usr/bin/env python3
"""Build and load the TVSD THINGS -> KS4 masked-response dataset.

Each SELECTED provisional longitudinal neuron receives one stable per-animal
column index, with original KS4 unit UUIDs retained in unit_to_neuron.csv.
Identity matches are proposals, not ground truth. Absent responses are masked.

Inside the existing project container:
    python scripts/dataset.py prepare
    python scripts/dataset.py check --animal F --partition train
    python scripts/dataset.py check --animal N --partition test

As a Python module:
    from dataset import TVSDDataset
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
from image_transform import image_chw_normalized, MODES
from constants import EXPECTED_SESSIONS
import torch
from torch.utils.data import Dataset, DataLoader

ROOT = Path('/project/data/processed/tvsd_waveform_aware_v1')
IMAGE_ROOT = Path('/project/data/assets/THINGS')
DEFAULT_PREPARED = ROOT / 'model_dataset'


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
    *, processed_root: Path = ROOT,
    image_root: Path = IMAGE_ROOT, output_dir: Path = DEFAULT_PREPARED,
    val_fraction: float = 0.1, seed: int = 8000,
    expected_sessions: int = EXPECTED_SESSIONS, overwrite: bool = False,
    check_images: bool = True, selection_dir: Path | None = None,
):
    """Pack responses by selected longitudinal tracks, independently of readout architecture.

    Selection is an explicit, separate, auditable step; here no new identity
    decision is made. Track index is never a Gaussian readout parameter.
    """
    processed_root, image_root, output_dir = map(Path, (
        processed_root, image_root, output_dir))
    if output_dir.resolve() == processed_root.resolve():
        raise ValueError('Output must be a separate dataset directory')
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f'{output_dir} already exists. Use --overwrite to rebuild.')
    inventory = pd.read_csv(processed_root / 'unit_instances.csv', low_memory=False)
    _require_columns(inventory, ['unit_instance_id', 'session', 'animal'], 'KS4 inventory')
    if not inventory.unit_instance_id.is_unique:
        raise ValueError('Duplicate KS4 unit_instance_id')
    selection_dir = Path(selection_dir) if selection_dir else processed_root / 'identity_diagnostics/joint/selection'
    membership_path = selection_dir / 'unit_to_neuron.csv'
    neuron_path = selection_dir / 'neuron_index.csv'
    if not membership_path.is_file() or not neuron_path.is_file():
        raise FileNotFoundError(f'Missing tracked neuron mapping in {selection_dir}; run select_tracks.py first')
    selection_config_path = selection_dir / 'selection_config.json'
    if not selection_config_path.is_file():
        raise FileNotFoundError(f'Missing selection_config.json in {selection_dir}; run select_tracks.py')
    selection_config = json.loads(selection_config_path.read_text())
    for key, relative in [
        ('source_membership_sha256', 'candidate_joint_unit_membership.csv'),
        ('source_quality_sha256', 'candidate_joint_track_quality.csv')]:
        path = processed_root / 'identity_diagnostics/joint' / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != selection_config.get(key):
            raise ValueError(f'Selected neurons are stale relative to {relative}; rerun select_tracks.py')
    matching_config_path = processed_root / 'identity_diagnostics/joint/matching_config.json'
    if selection_config.get('matching_config_sha256'):
        if (not matching_config_path.is_file() or
            hashlib.sha256(matching_config_path.read_bytes()).hexdigest() !=
                selection_config['matching_config_sha256']):
            raise ValueError('Matching weights/config changed since selection; rerun select_tracks.py')
    units = pd.read_csv(membership_path, low_memory=False)
    neurons = pd.read_csv(neuron_path, low_memory=False)
    _require_columns(units, ['unit_instance_id','session','animal',
                             'candidate_joint_track_id','neuron_index','neuron_id'], 'unit_to_neuron')
    _require_columns(neurons, ['animal','candidate_joint_track_id','neuron_index','neuron_id'], 'neuron_index')
    if units.empty or neurons.empty or units.isna().any().any() or neurons[['animal','neuron_index','neuron_id']].isna().any().any():
        raise ValueError('Empty or invalid selected neuron mapping')
    if not units.unit_instance_id.is_unique or units.duplicated(['session','animal','neuron_index']).any():
        raise ValueError('Duplicate unit or two simultaneous units mapped to one neuron')
    if not set(units.unit_instance_id).issubset(set(inventory.unit_instance_id)):
        raise ValueError('Mapped unit missing from original inventory')
    joined = units.merge(inventory[['unit_instance_id','session','animal']],on='unit_instance_id',
                         suffixes=('','_original'), validate='one_to_one')
    if not ((joined.session==joined.session_original)&(joined.animal==joined.animal_original)).all():
        raise ValueError('Tracked neuron mapping disagrees with original session/animal')
    if neurons.duplicated(['animal','neuron_index']).any() or neurons.duplicated(['animal','candidate_joint_track_id']).any():
        raise ValueError('Duplicate neuron_index in neuron selection')
    if set(zip(units.animal,units.neuron_index)) != set(zip(neurons.animal,neurons.neuron_index)):
        raise ValueError('Selected neurons and unit-to-neuron mapping disagree')
    if (units.groupby(['animal','neuron_index']).candidate_joint_track_id.nunique()>1).any():
        raise ValueError('Different tracks mapped to one neuron index')
    if not (units.animal+'/'+units.neuron_index.astype(str)==units.neuron_id).all():
        raise ValueError('Inconsistent neuron IDs')
    n_neurons = {}
    for animal, g in neurons.groupby('animal'):
        ids=np.sort(g.neuron_index.to_numpy(dtype=int))
        if not np.array_equal(ids,np.arange(len(ids))):
            raise ValueError(f'{animal}: neuron indices must be 0..N-1')
        n_neurons[str(animal)]=int(len(ids))
    units = units.sort_values(['animal','session','neuron_index','unit_instance_id'],kind='stable')
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
    all_sessions = sorted(pres.session.unique())
    if expected_sessions and len(all_sessions) != expected_sessions:
        raise ValueError(f'Expected {expected_sessions} presentation sessions, got {len(all_sessions)}')
    if set(inventory.session.unique()) != set(all_sessions):
        raise ValueError('KS4 inventory sessions differ from presentation manifest')
    sessions = sorted(all_sessions)
    if pres.session.str.extract(r'^macaque([FN])_TVSD_')[0].isna().any():
        raise ValueError('Unexpected session naming in image manifest')
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        # A rebuilt split/unit inventory requires calibration again, even if
        # the new split happens to have the same number of images.
        (output_dir / 'normalization.json').unlink(missing_ok=True)
    records = []
    actual_counts = {}
    observed_train_pairs = 0
    observed_test_pairs = 0
    common_window = None
    for k, session in enumerate(sessions, start=1):
        selected = units.loc[units.session.eq(session)].copy()
        if selected.empty:
            print(f'{k:2d}/{len(sessions)} {session}: no selected neuron; skipping session', flush=True)
            continue
        if selected.animal.nunique() != 1:
            raise ValueError(f'{session}: mixed animals in selected mapping')
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
                raise ValueError(f'{session}: indexed unit missing from NPZ: {next(iter(missing_ids))}')
            columns = {name: i for i,name in enumerate(original_ids)}
            selected = selected.sort_values('neuron_index', kind='stable')
            col_indices = np.array([columns[x] for x in selected.unit_instance_id], dtype=int)
            local_neuron_indices = selected.neuron_index.to_numpy(dtype=np.int32)
            local_counts = npz['counts'][:, col_indices]
            local_observed = npz['observed'][:, col_indices]
        if (response_window.shape != (2,) or not np.isfinite(response_window).all()
                or response_window[0] < 0 or response_window[0] >= response_window[1]):
            raise ValueError(f'{session}: invalid spike window: {response_window}')
        if common_window is None:
            common_window = response_window.copy()
        elif not np.array_equal(response_window, common_window):
            raise ValueError(f'{session}: inconsistent spike window {response_window}; '
                             f'expected {common_window} from first session')
        if (local_counts.shape != local_observed.shape or
            local_counts.shape[0] != len(source_rows) or
            local_counts.shape[1] != len(local_neuron_indices)):
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
        # These presentation counts come from actual trial arrays, not model selection.
        local_train = int(np.count_nonzero(trial_split == 'train'))
        local_test = int(np.count_nonzero(trial_split == 'test'))
        for i, (source_row, st_id) in enumerate(zip(source_rows, assigned_ids)):
            records.append(dict(animal=animal, session=session, trial_index=i,
                                source_row=int(source_row), stimulus_id=st_id,
                                split=str(trial_split[i]), partition=partition[st_id],
                                image_index=int(trial_image_index[i]),
                                image_relative_path=stimulus_paths[st_id]))
        actual_counts[session] = [local_train, local_test, len(local_neuron_indices)]
        observed_train_pairs += int(local_observed[trial_split == 'train'].sum())
        observed_test_pairs += int(local_observed[trial_split == 'test'].sum())
        _write_npz_atomic(output_dir / 'sessions' / f'{session}.npz',
                          counts=local_counts.astype(np.uint16, copy=False),
                          observed=local_observed.astype(bool, copy=False),
                          neuron_indices=local_neuron_indices,
                          source_rows=source_rows,
                          stimulus_ids=assigned_ids,
                          response_window_ms=response_window)
        print(f'{k:2d}/{len(sessions)} {session}: {len(source_rows)} presentations, '
              f'{len(local_neuron_indices)} tracked neuron observations', flush=True)
    index = pd.DataFrame.from_records(records)
    split_sha256 = _split_sha256(index)
    index.to_csv(output_dir / 'sample_index.csv', index=False)
    neurons.to_csv(output_dir / 'neuron_index.csv', index=False)
    # Preserve source provenance and allow downstream model-specific selection.
    inputs = {
        'unit_instances.csv': processed_root / 'unit_instances.csv',
        'stimulus_manifest.csv': processed_root / 'images/stimulus_manifest.csv',
        'presentation_to_stimulus.csv': processed_root / 'images/presentation_to_stimulus.csv',
    }
    inputs['unit_to_neuron.csv'] = membership_path
    inputs['neuron_index.csv'] = neuron_path
    config_path = selection_dir / 'selection_config.json'
    if config_path.is_file():
        inputs['selection_config.json'] = config_path
    if matching_config_path.is_file():
        inputs['matching_config.json'] = matching_config_path
    digests = {}
    for name, path in inputs.items():
        shutil.copyfile(path, output_dir / name)
        digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary = dict(version='tvsd_tracked_neurons_masked_v3', n_neurons=n_neurons,
                   n_source_sessions=len(all_sessions),
                   n_selected_unit_instances=int(len(units)),
                   n_original_unit_instances=int(len(inventory)),
                   n_sessions=len(sessions), n_presentations=len(index),
                   n_nominal_train_response_entries=int(sum(v[0]*v[2] for v in actual_counts.values())),
                   n_observed_train_response_entries=observed_train_pairs,
                   n_observed_test_response_entries=observed_test_pairs,
                   frozen_input_sha256=digests,
                   response_window_ms=common_window.tolist(), split_sha256=split_sha256,
                   validation_fraction=val_fraction,
                   split_seed=seed, image_root=str(image_root.resolve()),
                   session_counts=actual_counts,
                   partition_images={x: int(index.loc[index.partition.eq(x),'stimulus_id'].nunique())
                                     for x in ('train','val','test')},
                   partition_presentations=index.partition.value_counts().to_dict(),
                   preprocessing='Full image, PIL bicubic resize, MATLAB/BT.601 grayscale default, '
                                 'train-only global z-score; no center crop. Other modes explicit.',
                   identity_note='Columns are selected provisional longitudinal neurons; original unit UUIDs preserved in unit_to_neuron.csv')
    (output_dir / 'dataset_info.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('\nDataset built:', output_dir)
    print('Selected tracked neurons by animal:', n_neurons, 'Unit instances:', len(units))
    print('Nominal/observed train response entries:',
          summary['n_nominal_train_response_entries'], observed_train_pairs)
    print('Partition images:', summary['partition_images'])
    print('Partition presentations:', summary['partition_presentations'])
    print('No KS4 recordings or provisional identities modified.')
    return summary


def _split_sha256(index: pd.DataFrame) -> str:
    """Fingerprint actual image/split/path mapping, independent of presentations.

    This deliberately includes the complete frozen split, not just the number
    of images, and detects changes to stimulus-to-file mapping.
    """
    required = ['stimulus_id', 'partition', 'image_relative_path']
    _require_columns(index, required, 'presentation index')
    unique = index[required].drop_duplicates().sort_values(required, kind='stable')
    if unique.stimulus_id.duplicated().any() or unique[required].isna().any().any():
        raise ValueError('Non-unique or missing image split/path assignments')
    text = json.dumps(unique.values.tolist(), ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


class TVSDDataset(Dataset):
    """Per-animal image-presentation dataset with stable tracked-neuron columns.

    Output dict:
      image            float32 [3,H,W], train-global z-scored full image
      responses        float32 [n_neurons_for_animal], counts in the configured window
      response_mask    bool    [n_neurons_for_animal]; absent unit instances are false
      stimulus_id      string  'train:N' or 'test:N'
      session          string
      source_row       int
      animal           string

    Use one DataLoader per animal, or filter by session downstream.
    The neuron axis does not assume or require a Gaussian readout.
    """
    def __init__(self, *, animal: str, partition: str,
                 prepared_root: Path = DEFAULT_PREPARED,
                 image_root: Optional[Path] = None, image_size: int = 224,
                 load_responses: bool = True, normalization_path: Optional[Path] = None):
        self.root = Path(prepared_root)
        if animal not in ('F','N') or partition not in ('train','val','test'):
            raise ValueError('animal must be F/N; partition must be train/val/test')
        if int(image_size) != image_size or image_size < 1:
            raise ValueError('image_size must be a positive integer')
        self.animal, self.partition, self.image_size = animal, partition, int(image_size)
        info = json.loads((self.root / 'dataset_info.json').read_text())
        if animal not in info['n_neurons']:
            raise ValueError(f'No selected neurons for animal {animal} in this prepared dataset')
        self.n_neurons = int(info['n_neurons'][animal])
        self.image_root = Path(image_root) if image_root is not None else Path(info['image_root'])
        norm_path = Path(normalization_path) if normalization_path is not None else self.root/'normalization.json'
        norm = json.loads(norm_path.read_text())
        # This is intentionally checked even for an explicit normalization_path:
        # mismatched calibration and neural/image partitions must not go unnoticed.
        expected_sha = info.get('split_sha256')
        if not expected_sha or norm.get('split_sha256') != expected_sha:
            raise ValueError('Normalization split fingerprint missing/mismatched; '
                             'recalibrate on this prepared dataset')
        if norm.get('partial', False):
            raise ValueError('Partial normalization is for diagnostics only')
        if norm['image_size'] != self.image_size:
            raise ValueError('Input image_size differs from fitted normalization')
        self.color_mode = norm['color_mode']
        if self.color_mode not in MODES:
            raise ValueError('Unknown color mode in normalization file')
        self.mean,self.std = norm['mean'],norm['std']
        self.samples = pd.read_csv(self.root / 'sample_index.csv', low_memory=False)
        if _split_sha256(self.samples) != expected_sha:
            raise ValueError('Prepared sample_index.csv differs from its frozen split fingerprint')
        self.samples = self.samples.loc[
            self.samples.animal.eq(animal) & self.samples.partition.eq(partition)].reset_index(drop=True)
        if self.samples.empty:
            raise ValueError(f'No samples: {animal} {partition}')
        self._session_data: Dict[str, Dict[str,np.ndarray]] = {}
        if load_responses:
            for session in sorted(self.samples.session.unique()):
                with np.load(self.root / 'sessions' / f'{session}.npz', allow_pickle=False) as data:
                    if not np.array_equal(data['response_window_ms'], info['response_window_ms']):
                        raise ValueError(f'{session}: prepared response window disagrees with dataset_info')
                    counts = data['counts']
                    observed = data['observed']
                    indices = data['neuron_indices'].astype(np.int64)
                if len(set(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= self.n_neurons):
                    raise ValueError(f'Invalid local tracked neuron indices for {session}')
                if counts.shape != observed.shape or counts.shape[1] != len(indices):
                    raise ValueError(f'Malformed responses for {session}')
                self._session_data[session] = dict(counts=counts, observed=observed, indices=indices)
        self.load_responses = load_responses

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        row = self.samples.iloc[int(i)]
        path = _safe_image(self.image_root, row.image_relative_path)
        image = torch.from_numpy(image_chw_normalized(
            path, image_size=self.image_size, mode=self.color_mode,
            mean=self.mean, std=self.std))
        responses = torch.zeros(self.n_neurons, dtype=torch.float32)
        response_mask = torch.zeros(self.n_neurons, dtype=torch.bool)
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
        y.shape != mask.shape or y.shape[1] != ds.n_neurons):
        raise AssertionError('Unexpected batch shape')
    if mask.dtype != torch.bool or not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise AssertionError('Invalid pixel/response data')
    if (y[~mask] != 0).any() or not mask.any():
        raise AssertionError('Invalid unobserved values or empty mask')
    print(f'PASS {animal}/{partition}: {len(ds):,} presentations, {ds.n_neurons} tracked neurons')
    print('Image batch:', tuple(x.shape), 'range', (round(float(x.min()),4), round(float(x.max()),4)))
    print('Responses:', tuple(y.shape), 'mask:', tuple(mask.shape))
    print('Observed fraction in batch:', round(float(mask.float().mean()),4))
    window = json.loads((Path(prepared_root) / 'dataset_info.json').read_text())['response_window_ms']
    print(f'Mean observed spike count [{window[0]:g},{window[1]:g}) ms:', round(float(y[mask].mean()),4))
    print('First stimulus:', batch['stimulus_id'][0], 'session:', batch['session'][0])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    cmd = p.add_subparsers(dest='cmd', required=True)
    prep = cmd.add_parser('prepare', help='Build the compact response packs and global presentation index')
    prep.add_argument('--processed-root', type=Path, default=ROOT)
    prep.add_argument('--images-root', type=Path, default=IMAGE_ROOT)
    prep.add_argument('--output-dir', type=Path, default=DEFAULT_PREPARED)
    prep.add_argument('--val-fraction', type=float, default=0.1)
    prep.add_argument('--seed', type=int, default=8000)
    prep.add_argument('--expected-sessions', type=int, default=EXPECTED_SESSIONS)
    prep.add_argument('--overwrite', action='store_true')
    prep.add_argument('--skip-image-file-check', action='store_true')
    prep.add_argument('--selection-dir', type=Path,
                      help='Directory from select_tracks.py, defaults to root/identity_diagnostics/joint/selection')
    check = cmd.add_parser('check', help='Decode a batch and verify mask/unit layout')
    check.add_argument('--animal', choices=['F','N'], default='F')
    check.add_argument('--partition', choices=['train','val','test'], default='train')
    check.add_argument('--prepared-root', type=Path, default=DEFAULT_PREPARED)
    check.add_argument('--images-root', type=Path, default=None)
    check.add_argument('--image-size', type=int, default=224)
    check.add_argument('--batch-size', type=int, default=8)
    args = p.parse_args()
    if args.cmd == 'prepare':
        prepare_dataset(processed_root=args.processed_root,
                        image_root=args.images_root,output_dir=args.output_dir,
                        val_fraction=args.val_fraction,seed=args.seed,
                        expected_sessions=args.expected_sessions,overwrite=args.overwrite,
                        check_images=not args.skip_image_file_check, selection_dir=args.selection_dir)
    else:
        check_dataset(animal=args.animal,partition=args.partition,
                      prepared_root=args.prepared_root,image_root=args.images_root,
                      image_size=args.image_size,batch_size=args.batch_size)


if __name__ == '__main__':
    main()
