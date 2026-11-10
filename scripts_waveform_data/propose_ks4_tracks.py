#!/usr/bin/env python3
"""Exploratory longitudinal KS4 unit tracking, with hard within-session exclusion.

READ ONLY with respect to unit_instances.csv and session NPZs. A track is a
*proposal*, never proof of biological identity. Every unit instance is retained;
Katka's existing labels are left intact. Inspect diagnostics before using these
tracks as the model readout mapping.

Docker usage:
    python scripts/propose_ks4_tracks.py
    python scripts/propose_ks4_tracks.py --thresholds 0.97 0.98 0.99 0.995 \
        --consistency-slack 0.03 --proposal-threshold 0.98

Inputs: /project/data/processed/ks4_v1/unit_instances.csv
        /project/data/processed/ks4_v1/sessions/*.npz
Outputs: /project/data/processed/ks4_v1/identity_diagnostics/*
Dependencies: numpy and pandas only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/project/data/processed/ks4_v1')


def normalized_waveforms(root: Path, table: pd.DataFrame) -> np.ndarray:
    """One centered, L2-normalized waveform per unit, in inventory row order."""
    waveforms: dict[str, np.ndarray] = {}
    for session in sorted(table.session.unique()):
        path = root / 'sessions' / f'{session}.npz'
        with np.load(path) as d:
            ids = d['unit_instance_ids'].astype(str)
            wfs = d['waveforms_zscored'].astype(np.float64)
            if len(ids) != len(wfs) or len(set(ids)) != len(ids):
                raise ValueError(f'{session}: inconsistent/duplicate waveforms')
            if set(ids) & waveforms.keys():
                raise ValueError(f'{session}: duplicate global unit ID')
            waveforms.update(zip(ids, wfs))
    ids = table.unit_instance_id.astype(str).tolist()
    if set(ids) != set(waveforms):
        raise ValueError('Unit inventory does not match NPZ unit IDs')
    x = np.stack([waveforms[uid] for uid in ids])
    x -= x.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(x, axis=1)
    good = np.isfinite(x).all(axis=1) & (norm > 0)
    x[good] /= norm[good, None]
    x[~good] = np.nan
    print(f'Waveforms: {int(good.sum()):,}/{len(good):,} valid')
    return x


def site_correlations(x: np.ndarray) -> np.ndarray:
    # Dot product of centered/unit-L2-normalized waveforms = Pearson r.
    c = x @ x.T
    np.clip(c, -1.0, 1.0, out=c)
    return c


def make_tracks(group: pd.DataFrame, c: np.ndarray, threshold: float,
                consistency_slack: float) -> list[list[int]]:
    """Constrained agglomeration on one physical electrode.

    Merge the most similar units first. Reject merges if any recording would
    contain two units from the same proposed track, or if the *worst* waveform
    similarity across the merged tracks is < threshold - consistency_slack.
    The second condition limits chaining while allowing gradual waveform drift.
    """
    n = len(group)
    sessions = group['session'].to_numpy()
    parents = list(range(n))
    members = {i: [i] for i in range(n)}
    occupied = {i: {sessions[i]} for i in range(n)}

    def find(i: int) -> int:
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    ii, jj = np.triu_indices(n, 1)
    keep = np.isfinite(c[ii, jj]) & (c[ii, jj] >= threshold) & (sessions[ii] != sessions[jj])
    edges = sorted(zip(c[ii[keep], jj[keep]], ii[keep], jj[keep]), reverse=True)
    floor = threshold - consistency_slack
    for _, i, j in edges:
        a, b = find(int(i)), find(int(j))
        if a == b or (occupied[a] & occupied[b]):
            continue
        cross = c[np.ix_(members[a], members[b])]
        if not np.isfinite(cross).all() or np.min(cross) < floor:
            continue
        if len(members[a]) < len(members[b]):
            a, b = b, a
        parents[b] = a
        members[a].extend(members.pop(b))
        occupied[a].update(occupied.pop(b))
    tracks = list(members.values())
    tracks.sort(key=lambda idx: (min(sessions[idx]), min(idx)))
    assert sum(map(len, tracks)) == n
    assert all(len(set(sessions[idx])) == len(idx) for idx in tracks)
    return tracks


def evaluate(table: pd.DataFrame, x: np.ndarray, thresholds: list[float],
             proposal_threshold: float, slack: float, outdir: Path) -> None:
    groups = []
    negatives = []
    site_summary = []
    for (animal, array, electrode), rows in table.groupby(['animal', 'array', 'electrode_id'], sort=True):
        idx = rows.index.to_numpy(dtype=int)
        g = rows.reset_index(drop=True)
        corr = site_correlations(x[idx])
        ii, jj = np.triu_indices(len(g), 1)
        within = g.session.to_numpy()[ii] == g.session.to_numpy()[jj]
        negatives.extend(corr[ii[within], jj[within]].tolist())
        nmax = int(g.groupby('session').size().max())
        site_summary.append({
            'animal': animal, 'array': int(array), 'electrode_id': int(electrode),
            'n_unit_instances': len(g), 'n_sessions': g.session.nunique(),
            'max_simultaneous_units': nmax,
            'max_correlated_distinct_units': (
                float(np.nanmax(corr[ii[within], jj[within]]))
                if np.isfinite(corr[ii[within], jj[within]]).any() else np.nan
            ),
        })
        groups.append((animal, int(array), int(electrode), idx, g, corr))

    neg = np.asarray(negatives, dtype=float)
    neg = neg[np.isfinite(neg)]
    pd.DataFrame(site_summary).to_csv(outdir / 'electrode_sites.csv', index=False)
    lower_bound = sum(row['max_simultaneous_units'] for row in site_summary)
    print(f'Electrode sites: {len(site_summary)}; hard lower bound: {lower_bound:,} distinct units')
    print('Within-session distinct-unit correlations (negative controls):')
    if len(neg):
        print(pd.Series(neg).describe(percentiles=[.5, .9, .95, .99]).to_string())
    else:
        print('None available')

    thresholds = sorted(set(thresholds + [proposal_threshold]))
    sweep = []
    for t in thresholds:
        assignments = []
        quality = []
        for animal, array, electrode, idx, g, corr in groups:
            tracks = make_tracks(g, corr, t, slack)
            for track_no, members in enumerate(tracks, 1):
                track_id = f'{animal}/A{array}/E{electrode}/T{track_no:03d}'
                sub = g.iloc[members]
                if len(members) > 1:
                    a, b = np.triu_indices(len(members), 1)
                    rs = corr[np.ix_(members, members)][a, b]
                    min_r, median_r = float(np.min(rs)), float(np.median(rs))
                else:
                    min_r = median_r = np.nan
                classes = sorted(sub.final_class.dropna().astype(str).unique()) if 'final_class' in sub else []
                fr = sub.firing_rate_hz.to_numpy(dtype=float)
                fr = fr[np.isfinite(fr) & (fr > 0)]
                days = sorted(set(sub.session.str.extract(r'TVSD_(\d{8})_')[0].dropna()))
                quality.append({
                    'candidate_track_id': track_id, 'animal': animal,
                    'array': array, 'electrode_id': electrode,
                    'n_recordings': len(members), 'n_days': len(days),
                    'min_waveform_r': min_r, 'median_waveform_r': median_r,
                    'firing_rate_max_min_ratio': float(fr.max() / fr.min()) if len(fr) else np.nan,
                    'n_distinct_katka_classes': len(classes),
                    'katka_classes': ';'.join(classes),
                })
                for k in members:
                    assignments.append((g.iloc[k].unit_instance_id, track_id))
        proposed = pd.DataFrame(assignments, columns=['unit_instance_id', 'candidate_track_id'])
        q = pd.DataFrame(quality)
        if not proposed.unit_instance_id.is_unique or len(proposed) != len(table):
            raise AssertionError('Lost or duplicated unit instance in proposed mapping')
        summary = {
            'threshold': t, 'consistency_slack': slack,
            'n_provisional_tracks': len(q),
            'tracks_in_multiple_recordings': int((q.n_recordings > 1).sum()),
            'unit_instances_in_multirecording_tracks': int(q.loc[q.n_recordings > 1, 'n_recordings'].sum()),
            'tracks_spanning_multiple_days': int((q.n_days > 1).sum()),
            'tracks_with_katka_class_disagreement': int((q.n_distinct_katka_classes > 1).sum()),
            'fraction_negative_controls_above_threshold': float((neg >= t).mean()) if len(neg) else np.nan,
        }
        sweep.append(summary)
        print(f't={t:.4f}: {len(q):,} provisional tracks; '
              f'{summary["tracks_spanning_multiple_days"]:,} spanning days; '
              f'{summary["tracks_with_katka_class_disagreement"]:,} conflicting Katka classes')
        if abs(t - proposal_threshold) < 1e-10:
            # This is a *candidate* ID; it is deliberately NOT longitudinal_neuron_id.
            detail = table.merge(proposed, on='unit_instance_id', how='left', validate='one_to_one')
            if detail.candidate_track_id.isna().any():
                raise AssertionError('Missing proposed identity')
            if detail.groupby(['candidate_track_id', 'session']).size().max() != 1:
                raise AssertionError('Same-session collision inside a proposed track')
            detail.to_csv(outdir / 'candidate_unit_track_membership.csv', index=False)
            q.to_csv(outdir / 'candidate_track_quality.csv', index=False)
    pd.DataFrame(sweep).to_csv(outdir / 'track_threshold_sweep.csv', index=False)
    print(f'\nWrote diagnostics to {outdir}')
    print('IMPORTANT: unit_instances.csv and longitudinal_neuron_id remain unchanged.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--thresholds', nargs='+', type=float, default=[0.97, 0.98, 0.99, 0.995])
    parser.add_argument('--proposal-threshold', type=float, default=0.98)
    parser.add_argument('--consistency-slack', type=float, default=0.03)
    args = parser.parse_args()
    if any(t <= -1 or t > 1 for t in args.thresholds + [args.proposal_threshold]):
        parser.error('Thresholds must be in (-1, 1]')
    if args.consistency_slack < 0:
        parser.error('consistency-slack must be nonnegative')
    table = pd.read_csv(args.root / 'unit_instances.csv')
    required = {'unit_instance_id', 'animal', 'array', 'electrode_id', 'session', 'firing_rate_hz'}
    if required - set(table.columns):
        raise ValueError(f'Missing unit metadata columns: {sorted(required - set(table.columns))}')
    if not table.unit_instance_id.is_unique:
        raise ValueError('Duplicate unit IDs in master inventory')
    if 'longitudinal_neuron_id' in table and table.longitudinal_neuron_id.notna().any():
        raise ValueError('Master inventory already has assigned longitudinal identities')
    x = normalized_waveforms(args.root, table)
    outdir = args.root / 'identity_diagnostics'
    outdir.mkdir(parents=True, exist_ok=True)
    evaluate(table, x, args.thresholds, args.proposal_threshold,
             args.consistency_slack, outdir)


if __name__ == '__main__':
    main()
