#!/usr/bin/env python3
"""Export strict-V1 filtered KS4 NIX units, raw spikes and stimulus response counts.

No waveform classification is calculated. No cross-session neuronal identity is inferred.
Katka's previously exported labels are preserved independently and attached only
by exact (session, array, nix_name) equality.

Reads original TVSD trial metadata and all selected NIX files directly. One session
at a time; outputs compressed ragged spike times plus a trials x units count matrix.

Run in the TVSD Docker container:
    python scripts/build_ks4_v1.py --session macaqueF_TVSD_20240112_B1
    python scripts/build_ks4_v1.py --all

Requires numpy, pandas, h5py. Never reads Katka's huge pickles.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

DEFAULT_MANIFEST = Path('/project/data/reports/tvsd_manifest.csv')
DEFAULT_LABEL_DIR = Path('/project/data/reports/katka_labels')
DEFAULT_OUT = Path('/project/data/processed/ks4_v1')
WAVEFORM_SAMPLES = 90
LABEL_PATTERN = re.compile(r'monkey([FN])_all_arrays_date_(\d{8}_B\d+)_v1_labels\.csv$')


def as_string(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode('utf-8')
    return str(value)


def prop(properties, name, default=None):
    if name not in properties:
        return default
    arr = np.asarray(properties[name][()])
    if arr.size == 0:
        return default
    return arr.reshape(-1)[0] if arr.size == 1 else arr.reshape(-1)


def waveform(properties, name):
    value = prop(properties, name)
    if value is None:
        return np.full(WAVEFORM_SAMPLES, np.nan, dtype=np.float32)
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        arr = np.array([], dtype=np.float32)
    if arr.size != WAVEFORM_SAMPLES or not np.isfinite(arr).all():
        return np.full(WAVEFORM_SAMPLES, np.nan, dtype=np.float32)
    return arr


def read_nix_units(nix_path, array_id):
    """Yield the complete spike train plus metadata for each NIX unit."""
    with h5py.File(nix_path, 'r') as f:
        blocks = [(n, g) for n, g in f['metadata'].items() if n.startswith('neo.block.')]
        if len(blocks) != 1:
            raise ValueError(f'{nix_path}: expected exactly one metadata block, got {len(blocks)}')
        sections = blocks[0][1]['sections']
        segments = [g for n, g in sections.items() if n.startswith('neo.segment.')]
        if len(segments) != 1:
            raise ValueError(f'{nix_path}: expected exactly one segment, got {len(segments)}')

        block_name = blocks[0][0]
        arrays = f['data'][block_name]['data_arrays']
        for name, unit in sorted(segments[0]['sections'].items()):
            if not name.startswith('neo.spiketrain.'):
                continue
            p = unit['properties']
            uid = as_string(prop(p, 'nix_name'))
            if uid != name:
                raise ValueError(f'{nix_path}: UUID mismatch {uid} vs {name}')
            data_name = uid + '.times'
            if data_name not in arrays:
                raise ValueError(f'{nix_path}: missing data array {data_name}')
            times = np.asarray(arrays[data_name]['data'][()], dtype=np.float64).reshape(-1)
            if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
                raise ValueError(f'{nix_path}, {uid}: nonfinite or unsorted spike timestamps')
            electrode = prop(p, 'Electrode_ID')
            if electrode is None:
                raise ValueError(f'{nix_path}, {uid}: missing Electrode_ID')
            area = as_string(prop(p, 'Area', 'V1'))
            if area != 'V1':
                raise ValueError(f'{nix_path}, {uid}: expected Area V1, found {area}')
            annotated_array = prop(p, 'Array_ID')
            if annotated_array is not None and int(annotated_array) != int(array_id):
                raise ValueError(f'{nix_path}, {uid}: Array_ID does not match manifest')
            t_start = float(prop(p, 't_start', np.nan))
            t_stop = float(prop(p, 't_stop', np.nan))
            if not (np.isfinite(t_start) and np.isfinite(t_stop) and t_start <= t_stop):
                raise ValueError(f'{nix_path}, {uid}: invalid observation bounds')
            yield {
                't_start_s': t_start,
                't_stop_s': t_stop,
                'nix_name': uid,
                'electrode_id': int(electrode),
                'train_order_original': int(prop(p, 'train_order_original', -1)),
                'firing_rate_hz': float(prop(p, 'firing_rate', np.nan)),
                'waveform_zscored': waveform(p, 'avg_wf_zscored'),
                'waveform_raw': waveform(p, 'avg_wf'),
                'spike_times': times,
            }


def read_trials(recording_dir, session, window_ms):
    path = recording_dir / f'{session}_trial_metadata.csv'
    if not path.is_file():
        raise FileNotFoundError(f'Missing trial metadata: {path}')
    df = pd.read_csv(path)
    needed = {'Trial_start_s', 'Success', 'train_or_test_dataset',
              'THINGS_Train_Image_Index', 'THINGS_Test_Image_Index'}
    if missing := needed - set(df.columns):
        raise ValueError(f'{path}: missing columns: {sorted(missing)}')

    success = pd.to_numeric(df['Success'], errors='coerce').eq(1)
    split = df['train_or_test_dataset'].astype('string').str.strip().str.lower()
    invalid = success & ~split.isin(['train', 'test']).fillna(False)
    if invalid.any():
        raise ValueError(f'{path}: {invalid.sum()} successful trials with unknown split')
    indices_train = pd.to_numeric(df['THINGS_Train_Image_Index'], errors='coerce')
    indices_test = pd.to_numeric(df['THINGS_Test_Image_Index'], errors='coerce')
    selected = np.where(split[success].eq('train').to_numpy(),
                        indices_train[success].to_numpy(), indices_test[success].to_numpy())
    onset = pd.to_numeric(df.loc[success, 'Trial_start_s'], errors='coerce').to_numpy(dtype=float)
    if len(onset) == 0 or not np.isfinite(onset).all() or (np.diff(onset) <= 0).any():
        raise ValueError(f'{path}: invalid/unsorted successful trial onsets')
    if not np.isfinite(selected).all() or (selected < 0).any() or (selected != np.floor(selected)).any():
        raise ValueError(f'{path}: invalid image indices')
    if (onset + window_ms[0] / 1000 < 0).any():
        raise ValueError(f'{path}: negative response window start')

    trials = pd.DataFrame({
        'session': session,
        'source_row': np.flatnonzero(success.to_numpy()),
        'onset_s': onset,
        'split': split[success].to_numpy(dtype=str),
        'image_index': selected.astype(np.int64),
    })
    for c in ['THINGS_image_path', 'THINGS_image_category', 'NIN_local_path']:
        if c in df.columns:
            trials[c] = df.loc[success, c].to_numpy()
    return trials


def load_katka_labels(label_dir):
    """Preserve all recovered original labels, including ones absent from KS4."""
    frames = []
    for path in sorted(label_dir.glob('*_v1_labels.csv')):
        match = LABEL_PATTERN.fullmatch(path.name)
        if not match:
            raise ValueError(f'Unexpected Katka CSV filename: {path.name}')
        monkey, recording = match.groups()
        session = f'macaque{monkey}_TVSD_{recording}'
        df = pd.read_csv(path)
        required = {'cell_name', 'array', 'area', 'final_class'}
        if missing := required - set(df.columns):
            raise ValueError(f'{path}: missing {sorted(missing)}')
        if not df['area'].eq('V1').all():
            raise ValueError(f'{path}: contains non-V1 rows')
        df = df.copy()
        df['session'] = session
        df['animal'] = monkey
        df['katka_source_csv'] = path.name
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=['session', 'array', 'cell_name', 'final_class',
                                     'animal', 'katka_source_csv'])
    reference = pd.concat(frames, ignore_index=True)
    if reference.duplicated(['session', 'array', 'cell_name']).any():
        raise ValueError('Duplicate session/array/UUID in original Katka reference')
    return reference


def atomically_save_npz(path, **arrays):
    tmp = path.with_suffix('.npz.tmp')
    try:
        with tmp.open('wb') as f:
            np.savez_compressed(f, **arrays)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def process_session(session, rows, output, window_ms):
    source_path = Path(rows.iloc[0]['source_path'])
    recording_dir = source_path.parent.parent
    trials = read_trials(recording_dir, session, window_ms)
    onset = trials['onset_s'].to_numpy(dtype=np.float64)
    left = onset + window_ms[0] / 1000
    right = onset + window_ms[1] / 1000

    metadata, columns, masks, wfs, raw_wfs, trains = [], [], [], [], [], []
    for row in rows.itertuples(index=False):
        if not Path(row.source_path).is_file():
            raise FileNotFoundError(row.source_path)
        count_array = 0
        for unit in read_nix_units(row.source_path, row.array):
            uid = unit['nix_name']
            key = f'{session}/A{int(row.array)}/{uid}'
            spikes = unit['spike_times']
            observed = (left >= unit['t_start_s']) & (right <= unit['t_stop_s'])
            count = np.searchsorted(spikes, right, side='left') - np.searchsorted(
                spikes, left, side='left')
            count[~observed] = 0  # explicitly unobserved, NEVER a zero-spike target
            if np.any(count > np.iinfo(np.uint16).max):
                raise ValueError(f'Excess spikes per bin for {key}')
            columns.append(count.astype(np.uint16))
            masks.append(observed)
            wfs.append(unit['waveform_zscored'])
            raw_wfs.append(unit['waveform_raw'])
            trains.append(spikes)
            metadata.append({
                'unit_instance_id': key,
                'animal': session.split('_')[0].replace('macaque', ''),
                'session': session,
                'array': int(row.array),
                'nix_name': uid,
                'electrode_id': unit['electrode_id'],
                'train_order_original': unit['train_order_original'],
                'firing_rate_hz': unit['firing_rate_hz'],
                'n_spikes_total': len(spikes),
                't_start_s': unit['t_start_s'],
                't_stop_s': unit['t_stop_s'],
                'waveform_valid': bool(np.isfinite(unit['waveform_zscored']).all()),
            })
            count_array += 1
        if count_array != int(row.units):
            raise ValueError(f'{row.source_path}: {count_array} units, manifest says {row.units}')

    if not metadata:
        raise ValueError(f'No KS4 spike trains in {session}')
    info = pd.DataFrame(metadata)
    if info['unit_instance_id'].duplicated().any():
        raise ValueError(f'{session}: duplicate unit identifiers')
    counts = np.column_stack(columns)
    observed = np.column_stack(masks)
    if not observed.any():
        raise ValueError(f'{session}: no observed unit/trial pairs; check alignment')
    if not counts.any():
        raise ValueError(f'{session}: every response count is zero; check alignment')
    offsets = np.concatenate(([0], np.cumsum([len(s) for s in trains], dtype=np.int64)))
    all_times = np.concatenate(trains).astype(np.float64)
    if len(all_times) != offsets[-1]:
        raise AssertionError('Ragged spike offsets incorrect')

    (output / 'sessions').mkdir(parents=True, exist_ok=True)
    (output / 'trials').mkdir(parents=True, exist_ok=True)
    (output / 'unit_instances_by_session').mkdir(parents=True, exist_ok=True)
    atomically_save_npz(
        output / 'sessions' / f'{session}.npz',
        counts=counts,
        observed=observed,  # validity mask; false entries must not enter the loss
        spike_times_s=all_times,
        spike_offsets=offsets,
        unit_instance_ids=info['unit_instance_id'].to_numpy(dtype=str),
        waveforms_zscored=np.stack(wfs),
        waveforms_raw=np.stack(raw_wfs),
        trial_source_rows=trials['source_row'].to_numpy(dtype=np.int64),
        onset_s=onset,
        split=trials['split'].to_numpy(dtype=str),
        image_index=trials['image_index'].to_numpy(dtype=np.int64),
        response_window_ms=np.asarray(window_ms, dtype=np.float64),
    )
    info.to_csv(output / 'unit_instances_by_session' / f'{session}.csv', index=False)
    trials.to_csv(output / 'trials' / f'{session}.csv', index=False)
    nonzero = np.count_nonzero(counts) / np.count_nonzero(observed)
    fraction_observed = np.mean(observed)
    print(f'{session}: {counts.shape[0]} images × {counts.shape[1]} units; '
          f'{sum(info.waveform_valid)} valid waveforms; '
          f'{nonzero:.1%} nonzero among observed; {fraction_observed:.1%} observed; '
          f'{len(all_times):,} spikes', flush=True)


def rebuild_inventory(manifest, output, reference, window_ms):
    sessions = sorted(manifest['session'].unique())
    present = []
    for s in sessions:
        csv = output / 'unit_instances_by_session' / f'{s}.csv'
        npz = output / 'sessions' / f'{s}.npz'
        if not csv.exists() or not npz.exists():
            continue
        with np.load(npz) as saved:
            if not np.array_equal(saved['response_window_ms'], window_ms):
                raise RuntimeError(f'{s}: inconsistent response window; use --overwrite')
            if saved['counts'].shape[1] != len(saved['unit_instance_ids']):
                raise RuntimeError(f'{s}: response/unit shape mismatch')
            if saved['observed'].shape != saved['counts'].shape:
                raise RuntimeError(f'{s}: observation mask shape mismatch')
        present.append(pd.read_csv(csv))

    if not present:
        raise RuntimeError('No completed sessions')
    units = pd.concat(present, ignore_index=True)
    if units.duplicated('unit_instance_id').any():
        raise ValueError('Duplicate unit_instance_id across sessions')
    expected = int(manifest.loc[manifest.session.isin(units.session.unique()), 'units'].sum())
    if len(units) != expected:
        raise ValueError(f'Processed {len(units)} units, expected {expected} from manifest')

    reference.to_csv(output / 'katka_reference_units.csv', index=False)
    if len(reference):
        right = reference[['session', 'array', 'cell_name', 'final_class',
                           'katka_source_csv']].rename(columns={'cell_name': 'nix_name'})
        units = units.merge(right, on=['session', 'array', 'nix_name'],
                            how='left', validate='one_to_one')
    else:
        units['final_class'] = pd.NA
        units['katka_source_csv'] = pd.NA
    units['has_katka_label'] = units['final_class'].notna()
    # No cross-session identities should be inferred at this step.
    units['longitudinal_neuron_id'] = pd.NA

    done = units['session'].nunique()
    filename = 'unit_instances.csv' if done == len(sessions) else 'unit_instances_partial.csv'
    units.to_csv(output / filename, index=False)
    print(f'\nInventory: {done}/{len(sessions)} sessions, {len(units):,} KS4 unit instances; '
          f'{units.has_katka_label.sum():,} exact Katka labels')
    print(f'Original Katka reference: {len(reference):,} rows from '
          f'{reference.session.nunique() if len(reference) else 0} session exports')
    print(f'Saved: {output / filename}; no longitudinal identities assigned')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--session', help='Process one session first')
    mode.add_argument('--all', action='store_true', help='Process/resume all sessions')
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--labels-dir', type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--window-ms', nargs=2, type=float, default=[40, 160],
                        metavar=('START', 'STOP'))
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    start, stop = args.window_ms
    if not (0 <= start < stop):
        parser.error('Response window must have 0 <= START < STOP')
    window_ms = np.asarray([start, stop])

    manifest = pd.read_csv(args.manifest)
    mandatory = {'session', 'array', 'area', 'units', 'source_path'}
    if missing := mandatory - set(manifest.columns):
        raise ValueError(f'Manifest missing {sorted(missing)}')
    manifest = manifest.loc[manifest['area'].eq('V1')].sort_values(['session', 'array'])
    if manifest.empty or manifest.duplicated(['session', 'array']).any():
        raise ValueError('Manifest has no V1 recordings or duplicate arrays')
    all_sessions = sorted(manifest['session'].unique())
    requested = [args.session] if args.session else all_sessions
    if set(requested) - set(all_sessions):
        raise ValueError(f'Session not found in strict-V1 manifest: {requested}')
    reference = load_katka_labels(args.labels_dir)
    args.output_root.mkdir(parents=True, exist_ok=True)

    for session in requested:
        dest = args.output_root / 'sessions' / f'{session}.npz'
        meta = args.output_root / 'unit_instances_by_session' / f'{session}.csv'
        if not args.overwrite and dest.is_file() and meta.is_file():
            with np.load(dest) as saved:
                if not np.array_equal(saved['response_window_ms'], window_ms):
                    raise ValueError(f'{session}: existing response window differs; use --overwrite')
            print(f'SKIP completed: {session}', flush=True)
            continue
        group = manifest[manifest['session'].eq(session)]
        process_session(session, group, args.output_root, window_ms)
    rebuild_inventory(manifest, args.output_root, reference, window_ms)


if __name__ == '__main__':
    main()
