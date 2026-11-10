#!/usr/bin/env python3
"""Read-only adjacent-session waveform matching diagnostics for filtered V1 KS4 units.

Never assigns cross-session neuron IDs or combines unit responses. Candidates
must be from the same animal, array and electrode in *adjacent recordings*;
even a high waveform correlation is not evidence of identity on its own.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ROOT = Path('/project/data/processed/ks4_v1')


def order(session):
    m = re.fullmatch(r'macaque([FN])_TVSD_(\d{8})_B(\d+)', session)
    if not m:
        raise ValueError(f'Unexpected session: {session}')
    return m.group(1), m.group(2), int(m.group(3))


def correlation(x, y):
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return float('nan')
    a = x.astype(np.float64) - np.mean(x)
    b = y.astype(np.float64) - np.mean(y)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm else float('nan')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    table_path = args.root / 'unit_instances.csv'
    if not table_path.is_file():
        raise FileNotFoundError(f'Full unit inventory required: {table_path}. Run extractor --all first.')
    table = pd.read_csv(table_path)
    if not table['unit_instance_id'].is_unique:
        raise ValueError('Duplicate unit_instance_id')
    if table['longitudinal_neuron_id'].notna().any():
        raise RuntimeError('Inventory already has assigned identities; this diagnostic expects none')

    waveforms, raw_waveforms = {}, {}
    for session in sorted(table.session.unique()):
        with np.load(args.root / 'sessions' / f'{session}.npz') as a:
            ids = a['unit_instance_ids']
            if len(ids) != len(set(ids)):
                raise ValueError(f'{session}: duplicate IDs in .npz')
            for i, uid in enumerate(ids):
                if uid in waveforms:
                    raise ValueError(f'Unit ID repeated across files: {uid}')
                waveforms[uid] = a['waveforms_zscored'][i]
                raw_waveforms[uid] = a['waveforms_raw'][i]
    if set(waveforms) != set(table['unit_instance_id']):
        raise ValueError('Waveform NPZ files and unit inventory do not agree')

    candidates = []
    for animal, by_animal in table.groupby('animal', sort=True):
        all_sessions = sorted(by_animal.session.unique(), key=order)
        # Only compare consecutive full recordings; do not bridge missing sessions.
        for sa, sb in zip(all_sessions, all_sessions[1:]):
            a_sess = by_animal.loc[by_animal.session.eq(sa)]
            b_sess = by_animal.loc[by_animal.session.eq(sb)]
            a_grouped = a_sess.groupby(['array', 'electrode_id'])
            b_grouped = b_sess.groupby(['array', 'electrode_id'])
            electrodes = set(a_grouped.groups) & set(b_grouped.groups)
            for array, electrode in sorted(electrodes):
                a = a_grouped.get_group((array, electrode)).reset_index(drop=True)
                b = b_grouped.get_group((array, electrode)).reset_index(drop=True)
                av = a.unit_instance_id.tolist()
                bv = b.unit_instance_id.tolist()
                corrs = np.array([[correlation(waveforms[x], waveforms[y])
                                   for y in bv] for x in av], dtype=float)
                for i, x in enumerate(av):
                    valid = np.flatnonzero(np.isfinite(corrs[i]))
                    if not len(valid):
                        continue
                    ranked = valid[np.argsort(corrs[i, valid])[::-1]]
                    j = int(ranked[0]); y = bv[j]
                    competitor_a = float(corrs[i, ranked[1]]) if len(ranked) > 1 else np.nan
                    other_rows = np.flatnonzero(np.isfinite(corrs[:, j]))
                    best_row = int(other_rows[np.argmax(corrs[other_rows, j])])
                    runner_rows = [r for r in other_rows if r != i]
                    competitor_b = (float(np.max(corrs[runner_rows, j]))
                                    if runner_rows else np.nan)
                    ra, rb = raw_waveforms[x], raw_waveforms[y]
                    amp_a = float(np.ptp(ra)) if np.isfinite(ra).all() else np.nan
                    amp_b = float(np.ptp(rb)) if np.isfinite(rb).all() else np.nan
                    fr_a = float(a.iloc[i].firing_rate_hz)
                    fr_b = float(b.iloc[j].firing_rate_hz)
                    candidates.append({
                        'animal': animal, 'array': int(array),
                        'electrode_id': int(electrode),
                        'session_a': sa, 'session_b': sb,
                        'unit_a': x, 'unit_b': y,
                        'waveform_r': float(corrs[i, j]),
                        'second_best_r_for_a': competitor_a,
                        'second_best_r_for_b': competitor_b,
                        'margin_a': (float(corrs[i, j] - competitor_a)
                                     if np.isfinite(competitor_a) else np.nan),
                        'margin_b': (float(corrs[i, j] - competitor_b)
                                     if np.isfinite(competitor_b) else np.nan),
                        'n_units_a_on_electrode': len(av),
                        'n_units_b_on_electrode': len(bv),
                        'mutual_nearest': bool(best_row == i),
                        'amplitude_ratio_b_a': (amp_b / amp_a if np.isfinite(amp_a)
                                                and amp_a > 0 else np.nan),
                        'firing_rate_ratio_b_a': (fr_b / fr_a if np.isfinite(fr_a)
                                                  and fr_a > 0 else np.nan),
                        'katka_class_a': a.iloc[i].get('final_class', np.nan),
                        'katka_class_b': b.iloc[j].get('final_class', np.nan),
                    })

    output = args.root / 'waveform_match_candidates.csv'
    pd.DataFrame(candidates).to_csv(output, index=False)
    print(f'Saved {len(candidates):,} candidates: {output}')
    if candidates:
        df = pd.DataFrame(candidates)
        print('\nCorrelations for mutual nearest waveform candidates:')
        print(df.loc[df.mutual_nearest, 'waveform_r'].describe().to_string())
        print('\nTop candidate pairs (NOT confirmed identities):')
        cols = ['animal','array','electrode_id','session_a','session_b',
                'waveform_r','margin_a','margin_b','mutual_nearest']
        print(df.sort_values('waveform_r', ascending=False)[cols].head(12).to_string(index=False))
    print('\nNo neuron identities were assigned or merged.')


if __name__ == '__main__':
    main()
