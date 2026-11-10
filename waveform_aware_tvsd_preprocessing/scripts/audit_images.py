#!/usr/bin/env python3
"""Audit TVSD stimulus identities and optionally resolve to a local THINGS image tree.

Reads trial metadata produced by extract_ks4.py, never modifies it.
One stimulus identity is (split, image_index), NEVER just image_index.

Usage:
    python scripts/audit_images.py
    python scripts/audit_images.py --images-root /project/data/assets/THINGS
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path, PurePosixPath
import re

import numpy as np
import pandas as pd
from PIL import Image
from constants import EXPECTED_SESSIONS

SUPPORTED_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
IMAGE_COLUMNS = ('THINGS_image_path', 'NIN_local_path')


def clean_string(x):
    if pd.isna(x):
        return ''
    s = str(x).strip()
    if s.lower() in ('nan', 'none', 'null', '<na>'):
        return ''
    return s


def path_parts(s):
    s = clean_string(s).replace('\\', '/')
    if not s:
        return ()
    # Preserve internal spaces, but normalize ./ and repeated separators.
    parts = [p for p in s.split('/') if p not in ('', '.')]
    return tuple(p for p in parts if p != '..')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--processed-root', type=Path,
                   default=Path('/project/data/processed/tvsd_waveform_aware_v1'))
    p.add_argument('--output-dir', type=Path, default=None)
    p.add_argument('--images-root', type=Path, default=None,
                   help='Extracted THINGS image directory; resolution is optional.')
    p.add_argument('--expected-sessions', type=int, default=EXPECTED_SESSIONS)
    a = p.parse_args()
    trial_dir = a.processed_root / 'trials'
    files = sorted(trial_dir.glob('*.csv'))
    if len(files) != a.expected_sessions:
        raise RuntimeError(f'Found {len(files)} trial CSVs in {trial_dir}; '
                           f'expected {a.expected_sessions}. Check --processed-root.')

    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        required = {'session', 'source_row', 'onset_s', 'split', 'image_index'}
        if missing := required - set(df):
            raise ValueError(f'{f}: missing {sorted(missing)}')
        if len(df) == 0 or not df['session'].astype(str).eq(f.stem).all():
            raise ValueError(f'{f}: session mismatch or empty trials')
        for c in (*IMAGE_COLUMNS, 'THINGS_image_category'):
            if c not in df:
                df[c] = ''
        frames.append(df[['session', 'source_row', 'onset_s', 'split', 'image_index',
                          'THINGS_image_path', 'THINGS_image_category', 'NIN_local_path']])

    obs = pd.concat(frames, ignore_index=True)
    obs['split'] = obs['split'].astype(str).str.strip().str.lower()
    if not obs['split'].isin(('train', 'test')).all():
        raise ValueError('Found split other than train or test')
    index_numeric = pd.to_numeric(obs['image_index'], errors='raise')
    if not np.isfinite(index_numeric).all() or (index_numeric < 0).any() or (index_numeric % 1 != 0).any():
        raise ValueError('Invalid image indices')
    obs['image_index'] = index_numeric.astype('int64')
    for c in (*IMAGE_COLUMNS, 'THINGS_image_category'):
        obs[c] = obs[c].map(clean_string)
    obs['stimulus_id'] = obs['split'] + ':' + obs['image_index'].astype(str)
    if obs.duplicated(['session', 'source_row']).any():
        raise ValueError('Duplicate source_row within a session')

    out = a.output_dir or a.processed_root / 'images'
    out.mkdir(parents=True, exist_ok=True)

    records = []
    conflict_rows = []
    for (split, idx), g in obs.groupby(['split', 'image_index'], sort=True):
        paths = sorted(set(g['THINGS_image_path']) - {''})
        nin_paths = sorted(set(g['NIN_local_path']) - {''})
        categories = sorted(set(g['THINGS_image_category']) - {''})
        record = {
            'stimulus_id': f'{split}:{idx}',
            'split': split,
            'image_index': int(idx),
            'n_presentations': len(g),
            'n_sessions': g['session'].nunique(),
            'THINGS_image_path': paths[0] if len(paths) == 1 else '',
            'NIN_local_path': nin_paths[0] if len(nin_paths) == 1 else '',
            'THINGS_image_category': categories[0] if len(categories) == 1 else '',
            'n_distinct_things_paths': len(paths),
            'n_distinct_nin_paths': len(nin_paths),
            'n_distinct_categories': len(categories),
            'mapping_conflict': len(paths) > 1 or len(categories) > 1,
        }
        records.append(record)
        if record['mapping_conflict']:
            conflict_rows.append({**record,
                                  'all_things_paths': ' | '.join(paths),
                                  'all_categories': ' | '.join(categories)})

    unique = pd.DataFrame.from_records(records)
    conflicts = pd.DataFrame.from_records(conflict_rows)
    unique = unique.sort_values(['split', 'image_index']).reset_index(drop=True)

    # An identical concrete image path should not silently mean two different images.
    path_keys = defaultdict(set)
    for r in unique.itertuples(index=False):
        if r.THINGS_image_path:
            path_keys[r.THINGS_image_path].add(r.stimulus_id)
    reuse = [(p, ' | '.join(sorted(keys))) for p, keys in path_keys.items() if len(keys) > 1]
    pd.DataFrame(reuse, columns=['THINGS_image_path', 'stimulus_ids']).to_csv(
        out / 'reused_image_paths.csv', index=False)

    print(f'Trial CSVs: {len(files)}')
    print(f'Image presentations: {len(obs):,}')
    print('Unique (split, image_index) identities:')
    print(unique.groupby('split').agg(n_images=('image_index', 'size'),
                                      n_presentations=('n_presentations', 'sum'),
                                      min_index=('image_index', 'min'),
                                      max_index=('image_index', 'max')).to_string())
    print(f'Conflicting THINGS paths/categories per image ID: {len(conflicts)}')
    print(f'THINGS paths reused for multiple IDs: {len(reuse)}')
    print(f'Unique IDs missing THINGS_image_path: {(unique.THINGS_image_path == "").sum():,}')
    print('\nExample rows:')
    print(unique[['stimulus_id', 'n_presentations', 'THINGS_image_path',
                  'NIN_local_path']].groupby(unique['split'], sort=True).head(4).to_string(index=False,
                                                                                            max_colwidth=90))
    unique.to_csv(out / 'stimulus_manifest.csv', index=False)
    obs[['session', 'source_row', 'stimulus_id', 'split', 'image_index']].to_csv(
        out / 'presentation_to_stimulus.csv', index=False)
    conflicts.to_csv(out / 'mapping_conflicts.csv', index=False)

    if a.images_root is not None:
        root = a.images_root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(f'Images directory does not exist: {root}')
        print(f'\nIndexing local image files under {root} ...', flush=True)
        images = sorted(f for f in root.rglob('*') if f.is_file()
                        and f.suffix.lower() in SUPPORTED_EXT)
        print(f'Local image files: {len(images):,}')
        by_name = defaultdict(list)
        by_tail = defaultdict(list)
        for f in images:
            rel = f.relative_to(root).as_posix()
            parts = path_parts(rel)
            by_name[f.name.lower()].append(rel)
            for n in range(2, min(len(parts), 5) + 1):
                by_tail['/'.join(parts[-n:]).lower()].append(rel)
        statuses, paths_out, ambiguous = [], [], []
        for row in unique.itertuples(index=False):
            if row.mapping_conflict:
                statuses.append('metadata_conflict')
                paths_out.append('')
                continue
            candidates = []
            for field in IMAGE_COLUMNS:
                parts = path_parts(getattr(row, field))
                if parts:
                    candidates.append(parts)
            match = None
            ambiguous_set = set()
            # Prefer longest matching suffix; never resolve ambiguous basenames by arbitrary ordering.
            for parts in candidates:
                for n in range(min(len(parts), 5), 1, -1):
                    vals = set(by_tail.get('/'.join(parts[-n:]).lower(), []))
                    if len(vals) == 1:
                        match = next(iter(vals))
                        break
                    ambiguous_set.update(vals)
                if match:
                    break
            if not match:
                for parts in candidates:
                    vals = set(by_name.get(parts[-1].lower(), []))
                    if len(vals) == 1:
                        match = next(iter(vals))
                        break
                    ambiguous_set.update(vals)
            if match:
                statuses.append('resolved')
                paths_out.append(match)
            elif ambiguous_set:
                statuses.append('ambiguous')
                paths_out.append('')
                ambiguous.append((row.stimulus_id, ' | '.join(sorted(ambiguous_set)[:20])))
            else:
                statuses.append('not_found')
                paths_out.append('')
        unique['resolution_status'] = statuses
        unique['image_relative_path'] = paths_out
        unique.to_csv(out / 'stimulus_manifest.csv', index=False)
        pd.DataFrame(ambiguous, columns=['stimulus_id', 'candidate_paths']).to_csv(
            out / 'ambiguous_image_matches.csv', index=False)
        print('\nImage resolution:')
        print(unique['resolution_status'].value_counts().to_string())
        print('Image root:', root)
        nonsquare = []
        for rel in sorted(set(unique.loc[unique.resolution_status.eq('resolved'),
                                         'image_relative_path'])):
            with Image.open(root / rel) as im:
                w, h = im.size
            if w != h:
                nonsquare.append(dict(image_relative_path=rel, width=w, height=h))
        pd.DataFrame(nonsquare, columns=['image_relative_path', 'width', 'height']).to_csv(
            out / 'nonsquare_images.csv', index=False)
        if nonsquare:
            raise ValueError(f'{len(nonsquare)} non-square THINGS images found; '
                             '224x224 resize would distort aspect ratio. '
                             'See images/nonsquare_images.csv')
        print('Original image shapes: all resolved files are square.')
        if unique['resolution_status'].eq('resolved').all() and not reuse:
            print('All stimulus IDs resolved. Ready for a separate pixel/transform QA.')
        else:
            print('Not all images uniquely resolved. Do not train until collisions/missing files are addressed.')

    print(f'\nSaved reports under: {out}')
    print('No recording files, images or neuronal identities were modified.')


if __name__ == '__main__':
    main()
