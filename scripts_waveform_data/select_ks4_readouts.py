#!/usr/bin/env python3
"""Create an auditable conservative training cohort from joint KS4 tracks.

Requires COMPLETED label-aware tracking (track_ks4_joint_labeled.py) and the
previous electrode-sites waveform diagnostic. Does not modify source data.

A valid track is not automatically a good training target. The CORE cohort
requires >=3 distinct recording sessions, >=2000 TRAIN presentations,
consistent waveform polarity, good all-pairs waveform and firing-rate stability,
and no nearly identical simultaneously sorted units on the same electrode
(the latter is a conservative SITE-LEVEL risk veto, not proof of a track error).

All 11,012 original recording-specific units stay represented. Every class
annotation stays at the unit-instance level; no biological identity is invented.

Usage:
    python scripts/select_ks4_readouts.py
    python scripts/select_ks4_readouts.py --min-recordings 4 --min-train 3000
    python scripts/select_ks4_readouts.py --diagnostics /project/data/processed/ks4_v1/identity_diagnostics/joint
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_ROOT = Path('/project/data/processed/ks4_v1')
TRACK = 'candidate_joint_track_id'
KEYS = ['animal', 'array', 'electrode_id']


def require(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'{name}: missing {sorted(missing)}')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    ap.add_argument('--diagnostics', type=Path, default=None)
    ap.add_argument('--output', type=Path, default=None)
    ap.add_argument('--min-recordings', type=int, default=3)
    ap.add_argument('--min-train', type=int, default=2000)
    ap.add_argument('--min-pairwise-r', type=float, default=0.93)
    ap.add_argument('--max-rate-ratio', type=float, default=3.0)
    ap.add_argument('--max-distinct-r', type=float, default=0.98,
                    help='Review entire sites with simultaneously distinct waveforms this similar')
    args = ap.parse_args()
    if args.min_recordings < 2 or args.min_train < 0 or not -1 <= args.min_pairwise_r <= 1:
        ap.error('Invalid minimum-support constraints')
    diag = args.diagnostics or args.root / 'identity_diagnostics' / 'joint_labeled'
    outdir = args.output or diag / 'selection'
    if outdir.resolve() == args.root.resolve():
        ap.error('Output may not overwrite the source dataset')

    original = pd.read_csv(args.root / 'unit_instances_labeled.csv', low_memory=False)
    mapping = pd.read_csv(diag / 'candidate_joint_unit_membership.csv', low_memory=False)
    sites = pd.read_csv(args.root / 'identity_diagnostics' / 'electrode_sites.csv')
    presentations = pd.read_csv(args.root / 'images' / 'presentation_to_stimulus.csv')
    quality = pd.read_csv(diag / 'candidate_joint_track_quality.csv')
    require(original, {'unit_instance_id', 'session', 'animal', 'array', 'electrode_id',
                       'final_class', 'firing_rate_hz'}, 'labeled inventory')
    require(mapping, {'unit_instance_id', TRACK, 'candidate_day_track_id'}, 'joint mapping')
    require(sites, set(KEYS + ['max_simultaneous_units', 'max_correlated_distinct_units']), 'electrode sites')
    require(presentations, {'session', 'split', 'stimulus_id'}, 'stimulus presentations')
    require(quality, {TRACK, 'min_pairwise_waveform_r', 'n_days'}, 'track quality')
    if original.unit_instance_id.duplicated().any() or mapping.unit_instance_id.duplicated().any():
        raise ValueError('Duplicate unit_instance_id in an input')
    if original.final_class.isna().any():
        raise ValueError('Incomplete Katka labels. First run attach_katka_rebuilt.py')
    if set(original.unit_instance_id) != set(mapping.unit_instance_id):
        raise ValueError('Candidate mapping does not contain exactly the labeled source units')
    if quality[TRACK].duplicated().any() or sites.duplicated(KEYS).any():
        raise ValueError('Duplicate track or electrode-site entries')

    # Only borrow the proposed IDs, never stale partial annotations from the joint CSV.
    units = original.merge(mapping[['unit_instance_id', TRACK, 'candidate_day_track_id']],
                           on='unit_instance_id', validate='one_to_one')
    if units.groupby([TRACK, 'session']).size().max() > 1:
        raise ValueError('Two simultaneous units assigned to the same proposed neuron')
    # The recording-specific train/test split is authoritative, not filename suffixes.
    per_session = presentations.groupby(['session', 'split']).size().unstack(fill_value=0)
    for key in ['train', 'test']:
        if key not in per_session:
            per_session[key] = 0
    if not set(units.session).issubset(set(per_session.index)):
        raise ValueError('Missing stimulus presentation metadata for some sessions')
    units['train_presentations'] = units.session.map(per_session['train']).astype(int)
    units['test_presentations'] = units.session.map(per_session['test']).astype(int)

    tr = units.groupby(TRACK, sort=True).agg(
        animal=('animal', 'first'), array=('array', 'first'), electrode_id=('electrode_id','first'),
        n_recordings=('session','nunique'), n_unit_instances=('unit_instance_id','size'),
        n_day_tracks=('candidate_day_track_id','nunique'),
        n_train_presentations=('train_presentations','sum'),
        n_test_presentations=('test_presentations','sum'),
        min_firing_rate=('firing_rate_hz','min'),
        max_firing_rate=('firing_rate_hz','max'),
        n_classes=('final_class','nunique'),
        class_set=('final_class',lambda x: ';'.join(sorted(x.unique()))),
    ).reset_index()
    tr['n_days'] = (units.assign(day=units.session.str.extract(r'TVSD_(\d{8})_')[0])
                    .groupby(TRACK).day.nunique().reindex(tr[TRACK]).to_numpy())
    tr['firing_rate_ratio'] = tr.max_firing_rate / tr.min_firing_rate
    if (~np.isfinite(tr.firing_rate_ratio)).any():
        raise ValueError('Nonfinite firing rates; cannot audit neuron continuity')
    tr['mixed_polarity'] = tr.class_set.map(lambda s: 'UP' in s.split(';') and any(
        x.startswith('DOWN_') for x in s.split(';')))

    tr = tr.merge(quality[[TRACK, 'min_pairwise_waveform_r']], on=TRACK, validate='one_to_one')
    tr = tr.merge(sites[KEYS + ['max_simultaneous_units', 'max_correlated_distinct_units']],
                  on=KEYS, validate='many_to_one')
    tr['high_confusability_site'] = (
        (tr.max_simultaneous_units >= 2) &
        (tr.max_correlated_distinct_units >= args.max_distinct_r)
    ).fillna(False)
    tr['multiple_units_ever_on_site'] = tr.max_simultaneous_units > 1

    # Single and double recording tracks never enter the core, even with high correlation.
    # NaN pairwise waveform score on a singleton fails the waveform criterion.
    tests = {
        'insufficient_recordings': tr.n_recordings < args.min_recordings,
        'insufficient_training_images': tr.n_train_presentations < args.min_train,
        'unstable_waveforms': ~(tr.min_pairwise_waveform_r >= args.min_pairwise_r),
        'unstable_firing_rate': tr.firing_rate_ratio > args.max_rate_ratio,
        'mixed_waveform_polarity': tr.mixed_polarity,
        'ambiguous_multiunit_site': tr.high_confusability_site,
    }
    tr['exclusion_reasons'] = [
        ';'.join(name for name, bad in tests.items() if bool(bad.iloc[i]))
        for i in range(len(tr))
    ]
    tr['include_core'] = tr.exclusion_reasons.eq('')
    tr['selection'] = np.where(tr.include_core, 'core',
                               np.where((tr.n_recordings < args.min_recordings) |
                                        (tr.n_train_presentations < args.min_train),
                                        'insufficient_support', 'review'))
    # Assign model readout indices only to CORE tracks; indices are not biological IDs.
    core = tr.loc[tr.include_core].sort_values(KEYS + [TRACK]).copy()
    core['readout_index'] = core.groupby('animal').cumcount()
    tr = tr.merge(core[[TRACK, 'readout_index']], on=TRACK, how='left', validate='one_to_one')
    units = units.merge(tr[[TRACK, 'selection', 'include_core', 'readout_index',
                            'exclusion_reasons', 'high_confusability_site']],
                        on=TRACK, validate='many_to_one')
    if len(units) != len(original) or units.unit_instance_id.duplicated().any():
        raise AssertionError('Unit inventory changed unexpectedly')
    if tr.loc[tr.include_core].groupby('animal').readout_index.min().lt(0).any():
        raise AssertionError('Bad core readout indices')
    outdir.mkdir(parents=True, exist_ok=True)
    tr.to_csv(outdir / 'readout_selection.csv', index=False)
    units.to_csv(outdir / 'unit_selection.csv', index=False)
    tr.loc[tr.include_core].to_csv(outdir / 'core_readout_index.csv', index=False)
    diag_summary = tr.groupby(['animal','selection']).agg(
        n_tracks=(TRACK, 'size'),
        n_unit_instances=('n_unit_instances','sum'),
        n_train_responses=('n_train_presentations','sum'),
    ).reset_index()
    diag_summary.to_csv(outdir / 'selection_summary.csv', index=False)
    print(f'Original sorted unit observations: {len(units):,}, no recordings created/deleted')
    print('Waveform annotations:', units.final_class.nunique(), 'classes; all retained individually')
    print('Population before filtering:', len(tr), 'provisional tracks')
    print('Selection:\n' + diag_summary.to_string(index=False))
    print('Multiple-unit sites:', tr.multiple_units_ever_on_site.sum(), 'tracks; not automatically excluded')
    print('High-confusability sites:', tr.high_confusability_site.sum(), 'tracks placed under review or excluded')
    print('Output:', outdir)
    print('Original inventories, raw spike data, and joint proposals unchanged.')


if __name__ == '__main__':
    main()
