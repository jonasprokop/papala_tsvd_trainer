#!/usr/bin/env python3
"""Finalize label-aware KS4 neuron readouts, trimming rare width outliers.

This is a CONSERVATIVE alternative to finalize_ks4_readouts.py:
* Retain existing clean 'core' tracks unchanged.
* For a 'core' track containing both Katka DOWN_narrow* and DOWN_wide,
  remove individual original unit-session observations ONLY if the minority
  extreme-width class accounts for <=15% of the track, the opposite class is
  unambiguously more common, and the surviving track passes support, waveform
  width-span and firing-rate checks.
* Otherwise exclude the entire suspicious track. NEVER split or invent a neuron.

All 11,012 original unit observations and classifications are preserved in the
full audit CSV. NPZ spike data, original identities, and original class labels
are not modified. Run after select_ks4_readouts.py.

Examples:
  python scripts/finalize_ks4_readouts_trimmed.py
  python scripts/finalize_ks4_readouts_trimmed.py --min-days 2
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_DIR = Path('/project/data/processed/ks4_v1/identity_diagnostics/joint_labeled/selection')
TRACK = 'candidate_joint_track_id'


def has_extreme_width_conflict(class_set: object) -> bool:
    classes = str(class_set).split(';')
    return 'DOWN_wide' in classes and any(c.startswith('DOWN_narrow') for c in classes)


def width_group(classification: str) -> str:
    if classification.startswith('DOWN_narrow'):
        return 'narrow'
    if classification == 'DOWN_wide':
        return 'wide'
    if classification.startswith('DOWN_medium'):
        return 'medium'
    if classification == 'UP':
        return 'up'
    raise ValueError(f'Unknown Katka classification: {classification!r}')


def attempt_trim(group: pd.DataFrame, max_outlier_fraction: float,
                 max_width_span: float, min_recordings: int,
                 min_train: int, max_rate_ratio: float) -> tuple[pd.DataFrame | None, pd.DataFrame, str]:
    """Return (retained rows or None, rejected outliers, reason).

    A class label is only a signal for an observation-level exclusion; no new
    neuronal IDs are introduced. Nearly balanced narrow/wide classes are not
    resolved by an arbitrary vote and result in full-track exclusion.
    """
    categories = group['rebuilt_final_class'].map(width_group)
    narrow_count = int((categories == 'narrow').sum())
    wide_count = int((categories == 'wide').sum())
    n = len(group)
    if not (narrow_count > 0 and wide_count > 0):
        return group.copy(), group.iloc[:0].copy(), 'already_consistent'
    if narrow_count == wide_count:
        return None, group.iloc[:0].copy(), 'ambiguous_equal_extremes'
    minority = 'narrow' if narrow_count < wide_count else 'wide'
    minority_count = min(narrow_count, wide_count)
    if minority_count / n > max_outlier_fraction + 1e-12:
        return None, group.iloc[:0].copy(), 'too_many_extreme_outliers'
    retained = group.loc[categories != minority].copy()
    dropped = group.loc[categories == minority].copy()
    if retained['session'].nunique() < min_recordings or int(retained['train_presentations'].sum()) < min_train:
        return None, group.iloc[:0].copy(), 'insufficient_support_after_trim'
    widths = pd.to_numeric(retained['width_wf'], errors='coerce')
    if widths.isna().any() or (widths.max() - widths.min()) > max_width_span:
        return None, group.iloc[:0].copy(), 'remaining_widths_inconsistent'
    rates = pd.to_numeric(retained['firing_rate_hz'], errors='coerce')
    if rates.isna().any() or (rates <= 0).any() or rates.max()/rates.min() > max_rate_ratio:
        return None, group.iloc[:0].copy(), 'remaining_rates_inconsistent'
    remaining = set(retained['rebuilt_final_class'].map(width_group))
    if 'wide' in remaining and 'narrow' in remaining:
        raise AssertionError('Narrow/wide conflict remained after trimming')
    if retained.duplicated('session').any():
        raise AssertionError('Same-session collision after trimming')
    return retained, dropped, f'trim_{minority}_outliers'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection-dir', type=Path, default=DEFAULT_DIR)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--min-days', type=int, choices=[1, 2], default=1)
    p.add_argument('--max-outlier-fraction', type=float, default=.15)
    p.add_argument('--max-width-span', type=float, default=9.,
                   help='Maximum remaining trough-to-peak width span in original width_wf sample units')
    p.add_argument('--min-recordings', type=int, default=3)
    p.add_argument('--min-train', type=int, default=2000)
    p.add_argument('--max-rate-ratio', type=float, default=3.)
    args = p.parse_args()
    if not 0 <= args.max_outlier_fraction < .5:
        p.error('--max-outlier-fraction must lie in [0,0.5)')
    root = args.selection_dir
    out = args.output_dir or root / ('final_trimmed' if args.min_days == 1 else 'final_trimmed_cross_day')
    if out.resolve() == root.resolve():
        p.error('Output must not overwrite source selection files')
    tracks = pd.read_csv(root / 'readout_selection.csv', low_memory=False)
    units = pd.read_csv(root / 'unit_selection.csv', low_memory=False)
    for df, key, name in [(tracks, TRACK, 'tracks'), (units, 'unit_instance_id', 'units')]:
        if df[key].duplicated().any() or df[key].isna().any():
            raise ValueError(f'{name}: duplicate or missing {key}')
    if len(units) != 11012 or set(units[TRACK]) != set(tracks[TRACK]):
        raise ValueError('Expected exact original 11,012-unit inventory with matching track IDs')
    if units['rebuilt_final_class'].isna().any() or units.duplicated([TRACK, 'session']).any():
        raise ValueError('Missing Katka classes or simultaneous-unit identity collision')
    if not units['rebuilt_final_class'].map(lambda v: isinstance(v, str)).all():
        raise ValueError('Unrecognized Katka class data')
    if 'readout_index' in units:
        units = units.rename(columns={'readout_index': 'previous_core_readout_index'})
    units['include_final'] = False
    units['final_exclusion_reason'] = ''

    candidate_tracks = tracks.loc[tracks.selection.eq('core')].copy()
    records, trimmed_groups, excluded_tracks, audit = [], [], [], []
    for tr in candidate_tracks.itertuples(index=False):
        tid = getattr(tr, TRACK)
        q = units.loc[units[TRACK].eq(tid)]
        if has_extreme_width_conflict(tr.class_set):
            kept, dropped, reason = attempt_trim(q, args.max_outlier_fraction,
                   args.max_width_span, args.min_recordings, args.min_train, args.max_rate_ratio)
        else:
            kept, dropped, reason = q.copy(), q.iloc[:0].copy(), 'unchanged_core'
        if kept is not None and args.min_days == 2:
            days = kept.session.str.extract(r'TVSD_(\d{8})_')[0].nunique()
            if days < 2:
                kept = None
                dropped = q.iloc[:0].copy()
                reason = 'insufficient_recording_days'
        if kept is None:
            excluded_tracks.append(tid)
            units.loc[units[TRACK].eq(tid), 'final_exclusion_reason'] = reason
            audit.append(dict(candidate_joint_track_id=tid, outcome='exclude_track', reason=reason,
                              before=len(q), after=0, dropped=len(q)))
            continue
        if kept.duplicated('session').any():
            raise ValueError(f'Collision in surviving track: {tid}')
        record = tracks.loc[tracks[TRACK].eq(tid)].iloc[0].to_dict()
        # Update all support and width information to describe the INCLUDED subset.
        record.update(n_recordings=int(kept.session.nunique()),
                      n_unit_instances=int(len(kept)),
                      n_day_tracks=int(kept.candidate_day_track_id.nunique()),
                      n_train_presentations=int(kept.train_presentations.sum()),
                      n_test_presentations=int(kept.test_presentations.sum()),
                      n_days=int(kept.session.str.extract(r'TVSD_(\d{8})_')[0].nunique()),
                      min_firing_rate=float(kept.firing_rate_hz.min()),
                      max_firing_rate=float(kept.firing_rate_hz.max()),
                      firing_rate_ratio=float(kept.firing_rate_hz.max() / kept.firing_rate_hz.min()),
                      n_classes=int(kept.rebuilt_final_class.nunique()),
                      class_set=';'.join(sorted(kept.rebuilt_final_class.unique())),
                      n_excluded_observations=int(len(q)-len(kept)),
                      finalization_method=reason,
                      original_n_unit_instances=int(len(q)),
                      original_n_train_presentations=int(q.train_presentations.sum()))
        if kept.session.nunique() < args.min_recordings or record['n_train_presentations'] < args.min_train:
            raise ValueError(f'Surviving track fails minimum support: {tid}')
        if record['firing_rate_ratio'] > args.max_rate_ratio + 1e-6:
            raise ValueError(f'Surviving track has unstable FR: {tid}')
        if has_extreme_width_conflict(record['class_set']):
            raise ValueError(f'Surviving track still mixes narrow and wide: {tid}')
        records.append(record)
        units.loc[units.unit_instance_id.isin(kept.unit_instance_id), 'include_final'] = True
        if not dropped.empty:
            units.loc[units.unit_instance_id.isin(dropped.unit_instance_id), 'final_exclusion_reason'] = 'trimmed_extreme_width_outlier'
            trimmed_groups.append(dropped)
        audit.append(dict(candidate_joint_track_id=tid, outcome='retain' if reason=='unchanged_core' else 'trim_and_retain',
                          reason=reason, before=len(q), after=len(kept), dropped=len(q)-len(kept)))

    units.loc[~units.include_final & units.final_exclusion_reason.eq(''), 'final_exclusion_reason'] = 'not_in_core_selection'
    selected = pd.DataFrame(records)
    if selected.empty:
        raise ValueError('Nothing selected')
    selected = selected.sort_values(['animal','array','electrode_id',TRACK],kind='stable').reset_index(drop=True)
    selected['readout_index'] = selected.groupby('animal').cumcount().astype('int64')
    selected['readout_id'] = selected.animal + '/' + selected.readout_index.astype(str)
    selected['selection_version'] = 'trimmed_core_v1' if args.min_days == 1 else 'trimmed_cross_day_v1'
    units = units.drop(columns=[c for c in ['readout_index','readout_id'] if c in units.columns])
    units = units.merge(selected[[TRACK,'animal','readout_index','readout_id']].rename(columns={'animal':'readout_animal'}),
                        on=TRACK, how='left', validate='many_to_one')
    # A trimmed observation belongs to an accepted candidate track but MUST NOT have a model output.
    trimmed = ~units.include_final
    units.loc[trimmed, ['readout_index','readout_id']] = [np.nan, pd.NA]
    units['readout_index'] = units['readout_index'].astype('Int64')
    if not (units.loc[units.include_final, 'animal'] == units.loc[units.include_final, 'readout_animal']).all():
        raise ValueError('Cross-animal identity assignment detected')
    if units.loc[~units.include_final, 'readout_index'].notna().any():
        raise ValueError('Excluded observation still mapped to model output')
    included = units.loc[units.include_final].copy()
    if included.duplicated(['session','animal','readout_index']).any():
        raise ValueError('Simultaneous sorted units assigned the same model readout')
    if not np.array_equal(included.groupby(TRACK).train_presentations.sum().reindex(selected[TRACK]).to_numpy(),
                          selected.n_train_presentations.to_numpy()):
        raise ValueError('Training response count discrepancy')
    if len(included) + int((~units.include_final).sum()) != 11012:
        raise AssertionError('Lost a unit observation during selection')
    cols = ['unit_instance_id','animal','session','array','electrode_id','nix_name',
            'rebuilt_final_class', TRACK,'readout_index','readout_id','train_presentations','test_presentations']
    out.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out/'readout_index.csv',index=False)
    included[cols].to_csv(out/'unit_to_readout.csv',index=False)
    units.to_csv(out/'unit_selection_final.csv',index=False)
    pd.DataFrame(audit).to_csv(out/'width_conflict_audit.csv',index=False)
    units.loc[units.final_exclusion_reason.eq('trimmed_extreme_width_outlier')].to_csv(
        out/'trimmed_width_observations.csv', index=False)
    tracks.loc[tracks[TRACK].isin(excluded_tracks)].to_csv(out/'excluded_suspicious_core_tracks.csv',index=False)
    print(f'Original KS4 unit observations: {len(units):,} (all preserved in audit)')
    print(f'Original provisional tracks: {len(tracks):,} (unchanged)')
    print(f'Core tracks before width check: {len(candidate_tracks):,}')
    print(f'Conflicted core tracks: {int(candidate_tracks.class_set.map(has_extreme_width_conflict).sum())}')
    print(f'Conflicted core tracks salvaged by trimming: {len(trimmed_groups):,}')
    print(f'Conflicted core tracks fully excluded: {len(excluded_tracks):,}' if args.min_days == 1 else
          f'Core tracks excluded, including day criterion: {len(excluded_tracks):,}')
    print(f'Individual suspicious observations trimmed: {sum(map(len, trimmed_groups)):,}')
    print(f'Minimum recording days: {args.min_days}')
    print(f'Final readouts: {len(selected):,}; by animal: {selected.groupby("animal").size().to_dict()}')
    print(f'Included KS4 observations: {len(included):,}')
    print(f'Training response entries: {int(selected.n_train_presentations.sum()):,}')
    print(f'Saved: {out}')
    print('No neuron IDs fabricated; no spike data or original labels modified.')

if __name__ == '__main__':
    main()
